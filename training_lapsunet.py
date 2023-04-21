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

from datasets import datasetSR
from torchvision import transforms

from torch.utils.data.dataloader import DataLoader
from models.lapsunet import LapSUNET
from utils import util_calculate_psnr_ssim as util

from thop import profile


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
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
    # print(in_patch.shape)
    
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            # print(idx)
            out_patch_mask = torch.ones_like(out_patch[idx])

            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch[idx])
            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
            
            
    output = E.div_(W)
    return output



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    args = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn(1, 3, 256, 256)

    # our model
    # without shortcut: 6.58G, 31.79
    # model = LapSUNET(dim=12, depth=2, num_heads=3, mlp_ratio=2)
    # with shortcut: 11.02G, 31.95
    # model = LapSUNET(dim=16, depth=2, num_heads=2, mlp_ratio=2)
    # with shortcut, customized num_heads: 8.63G, 
    model = LapSUNET(dim=14, depth=2, num_heads=[2, 4], mlp_ratio=2)

    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')
    print(f'Flops: {macs * 2 * 15 * 60 / 1e9} G, for 720p cut into 15 patches and 60fps.')

    model = model.to(device)
    model = nn.DataParallel(model)

    # model path
    model_save_path = 'pretrained_weight/lapsunet/bullshit_training_lapsunet.pth'
    model_checkpoint_path = 'checkpoints/lapsunet.pt'
    model_best_path = 'checkpoints/lapsunet_best.pt'

    # basic setup
    best_psnr = 0
    checkpoint = None
    epochs = args.epochs
    batch_size = args.batch
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.05*args.lr)
    epoch = 0

    # Loading from best
    if os.path.exists(model_best_path):
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
    LQ_train_path = ['/data/SR/div2k/LRbicx3', '/data/SR/flickr2k/LRbicx3']      # 800 pic + 2650 pic -> 5088 patches + 16377 patches
    GT_train_path = ['/data/SR/div2k/original', '/data/SR/flickr2k/original']    # 800 pic + 2650 pic -> 5088 patches + 16377 patches

    LQ_valid_path = ["/data/SR/Set5/LRbicx3/"]    # 5 pic
    GT_valid_path = ["/data/SR/Set5/original/"]   # 5 pic

    train_ds = datasetSR(lq_paths=LQ_train_path, gt_paths=GT_train_path)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)

    valid_ds = datasetSR(lq_paths=LQ_valid_path, gt_paths=GT_valid_path, valid=True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=1, num_workers=8)

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
    psnrs = []
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
            # save result img
            if args.savedir != None:
                if not os.path.isdir(args.savedir):
                    os.mkdir(args.savedir)
                save_img = (output.clamp(0, 1) * 255).round().data.squeeze().cpu().numpy()
                if save_img.ndim == 3:
                    save_img = np.transpose(save_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                save_img = save_img[:h_old*args.scale, :w_old*args.scale].astype(np.uint8)  # float32 to uint8
                cv2.imwrite(os.path.join(args.savedir, img_name), save_img)

        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()

        psnr = util.psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean()}\n')