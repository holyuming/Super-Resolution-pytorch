import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import glob
import numpy as np
import time
import math
from datetime import datetime
from collections import OrderedDict

from datasets import datasetSR
from torchvision import transforms

from torch.utils.data.dataloader import DataLoader
from models.hat import HAT
from utils import util_calculate_psnr_ssim as util
import torch_optimizer

from thop import profile


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--savedir', type=str, default=None)
    args = parser.parse_args()
    return args

def demo_UHD_fast(img, model):
    # test the image tile by tile
    # print(img.shape) # [1, 3, 2048, 1152] for ali forward data
    scale = 3
    b, c, h, w = img.size()
    tile = min(256, h, w)
    tile_overlap = 0
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*scale, w*scale).type_as(img)
    W = torch.zeros_like(E)
    
    in_patch = []
    # append all 256x256 patches in a batch with size = 135
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch.append(img[..., h_idx:h_idx+tile, w_idx:w_idx+tile].squeeze(0))

    in_patch = torch.stack(in_patch, 0)
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            out_patch_mask = torch.ones_like(out_patch[idx])

            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch[idx])
            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
            
    output = E.div_(W)
    return output


if __name__ == '__main__':
    args = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn(1, 3, 256, 256).to(device)

    # v1 (customized_lightweight_sr HAT), 10G for input [1, 3, 256, 256], PSNR: 32.1134 dB, PSNRY: 34.1501 dB
    model = HAT( 
        img_size=256, patch_size=1, in_chans=3, embed_dim=24, depths=(2, 2, 2), num_heads=(3, 3, 3), 
        window_size=8, compress_ratio=4, squeeze_factor=30, conv_scale=0.1, overlap_ratio=0.5, mlp_ratio=2., 
        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
        ape=False, patch_norm=True, use_checkpoint=False, upscale=3, img_range=1., upsampler='pixelshuffledirect', resi_connection='1conv').to(device)

    # model path
    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')
    print(f'Flops: {macs * 2 * 15 * 60 / 1e12} T, for 720p cut into 15 patches and 60fps.')

    model = model.to(device)
    model = nn.DataParallel(model)

    # model path
    model_checkpoint_path = 'checkpoints/hat_v1.pt'
    model_best_path = 'checkpoints/hat_v1_best.pt'

    # basic setup
    best_psnr = 0
    checkpoint = None
    epochs = args.epochs
    batch_size = args.batch
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch_optimizer.Lamb(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*args.lr)
    epoch = 0

    # Loading from best
    if os.path.exists(model_best_path):
        print(f'Loading state_dict from {model_best_path}')
        checkpoint = torch.load(model_best_path)
        best_psnr = checkpoint['best_psnr']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Best PSNR: ', best_psnr)

    # Loading from checkpoint
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        if input(f'Do you want to load from checkpoint: {model_checkpoint_path} ? (y/n) ') == 'y':
            print(f'Loading state_dict from {model_checkpoint_path}')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']




    # datasets & dataloaders
    # for each training data --> input size = (B, 3, 256, 256), output size = (B, 3, 768, 768)
    # I generate those 256 x 256 sub images w/ utils/img_proc.py
    LQ_train_path = ["/work/u0810886/SR/div2k/LRbicx3/", "/work/u0810886/SR/flickr2k/LRbicx3/"]      # 800 pic + 2650 pic -> 5088 patches + 16377 patches
    GT_train_path = ["/work/u0810886/SR/div2k/original/", "/work/u0810886/SR/flickr2k/original/"]    # 800 pic + 2650 pic -> 5088 patches + 16377 patches

    LQ_valid_path = ["/work/u0810886/SR/Set5/LRbicx3/"]    # 5 pic
    GT_valid_path = ["/work/u0810886/SR/Set5/original/"]   # 5 pic

    train_ds = datasetSR(lq_paths=LQ_train_path, gt_paths=GT_train_path)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=4)

    valid_ds = datasetSR(lq_paths=LQ_valid_path, gt_paths=GT_valid_path, valid=True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=1, num_workers=4)

    # training
    print(f'================================TRAINING===============================')
    if args.train:
        model.train()
        min_loss = None
        time_start = datetime.now()
        # for epoch in range(epochs):
        while(epoch < epochs):
            for idx, data in enumerate(train_dl):
                img_lq = data[0].to(device)
                img_gt = data[1].to(device)
                
                preds   = model(img_lq)
                loss    = criterion(preds, img_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                min_loss = loss if min_loss == None or loss < min_loss else min_loss

                # display current information
                if idx % 100 == 0:
                    time_period = datetime.now() - time_start
                    # ==============================================================
                    # ==============================================================

                    psnrs = []
                    model.eval()
                    for _idx, data in enumerate(valid_dl):
                        img_name = data[0][0]
                        img_lq = data[1].to(device)
                        img_gt = data[2].to(device)
                        with torch.no_grad():
                            _, _, h_old, w_old = img_lq.size()
                            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
                            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
                            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                            output = demo_UHD_fast(img_lq, model)
                            preds = (output[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()

                        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()
                        psnr = util.psnr_tensor(preds, img_gt)
                        psnrs.append(psnr)
                        # print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
                        
                    psnrs = torch.tensor(psnrs)
                    # print(f'\nAverage PSNR: {psnrs.mean()}\n')
                    print(f'Epoch: {str(epoch):4s}, iter: {str(idx):6s}, Loss: {loss:.7f}, time: {str(time_period):.11s}, average psnr: {psnrs.mean()}')

                    # save checkpoint if best
                    if psnrs.mean() > best_psnr:
                        best_psnr = psnrs.mean()
                        print(f'BEST !!!! {best_psnr}')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'best_psnr': best_psnr,
                        }, model_best_path)

                    model.train()
                    # ==============================================================
                    # ==============================================================

            epoch += 1
            scheduler.step()

            # save checkpoint every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, model_checkpoint_path)


        print(f'Minimum Loss: {min_loss}, total training time: {str(datetime.now() - time_start):.11s}')



    # evaluating
    print(f'================================EVALUATION=============================')
    psnrs = []

    print('Original model after training:')
    model.eval()
    for idx, data in enumerate(valid_dl):
        img_name = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = demo_UHD_fast(img_lq, model)
            preds = (output[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()

        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()
        psnr = util.psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean()}\n')


    print('Model with best loss after training:')
    if os.path.exists(model_best_path):
        checkpoint = torch.load(model_best_path)
        print(f'Loading state_dict from {model_best_path}')
        model.load_state_dict(checkpoint['model_state_dict'])  
    model.eval()
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0
    border = 0
    for idx, data in enumerate(valid_dl):
        imgname = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        
        # inference
        with torch.no_grad():
            # pad input image to be a power of 2
            _, _, h_old, w_old = img_lq.shape
            # print(f'w, h : {w_old}, {h_old}')
            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = demo_UHD_fast(img_lq, model)
            output = (output[..., :h_old * args.scale, :w_old * args.scale].clamp(0, 1) * 255).round()
            output = output.squeeze().cpu().numpy().astype(np.uint8)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).cpu().numpy().round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[..., :h_old * args.scale, :w_old * args.scale]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border, input_order='CHW')
            ssim = util.calculate_ssim(output, img_gt, crop_border=border, input_order='CHW')
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True, input_order='CHW')
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True, input_order='CHW')
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            print('Testing {:3s} {:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}.'.
                  format(str(idx), imgname, psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:3d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n-- Average PSNR/SSIM(RGB): {:.4f} dB; {:.4f}'.format(ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.4f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))