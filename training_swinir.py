import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import glob
import numpy as np
import time
from datetime import datetime
import logging

from datasets import datasetSR
from torchvision import transforms

from torch.utils.data.dataloader import DataLoader
from models.swinir import SwinIR
from utils import util_calculate_psnr_ssim as util

from thop import profile


def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--trainset', type=str, default='testsets/Set5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--evalset', type=str, default='testsets/Set5')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # logging.basicConfig(filename='train.log', filemode='w', level=logging.DEBUG)
    args = parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # original model
    # model pretrained weight path = 'pretrained_weight/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth'
    model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')


    # img = torch.randn(16, 3, 256, 256).to(device)
    img = torch.randn(1, 3, 1280, 720).to(device)

    # our model
    # best psnr = 28 on set5, batch = 8
    model = SwinIR(upscale=args.scale, in_chans=3, img_size=256, window_size=8,
                img_range=1., depths=[2, 2, 2], embed_dim=24, num_heads=[2, 2, 2],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device)

    # best psnr = 29.35, batch = 8, lr = 1e-3, epochs = 5
    # best psnr = , batch = 8, lr = 1e-3, epochs = 1, with transform
    model = SwinIR(upscale=args.scale, in_chans=3, img_size=256, window_size=8,
                img_range=1., depths=[2, 2, 2, 2], embed_dim=24, num_heads=[2, 2, 2, 1],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device)

    # best psnr = 29.31, batch = 8, lr = 1e-3, epochs = 5, num_heads[2, 2, 2, 1]
    # best psnr = 20.50, batch = 8, lr = 1e-3, epochs = 1, num_heads[2, 2, 2, 1], with transform(have some issues)
    # best psnr = 31.54, batch = 8, lr = 1e-3, epochs = 50, num_heads[2, 2, 2, 1]
    # best psnr = 31.54, batch = 8, lr = 1e-3, epochs = 15, num_heads[2, 2, 2, 1], div2k, flickr2k
    model = SwinIR(upscale=args.scale, in_chans=3, img_size=256, window_size=8,
                img_range=1., depths=[2, 2, 2, 2], embed_dim=24, num_heads=[2, 2, 2, 1],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device)


    model = model.to(device)
    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')
    logging.info(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')


    # basic setup
    model = model.to(device)
    epochs = 15
    batch_size = 8
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # datasets & dataloaders
    # for each training data --> input size = (B, 3, 256, 256), output size = (B, 3, 768, 768)
    # I generate those 256 x 256 sub images w/ utils/img_proc.py
    LQ_train_path = ['/data/SR/div2k/LRbicx3', '/data/SR/flick2k/LRbicx3']      # 800 pic + 2650 pic -> 5088 patches + 16377 patches
    GT_train_path = ['/data/SR/div2k/original', '/data/SR/flick2k/original']    # 800 pic + 2650 pic -> 5088 patches + 16377 patches

    LQ_valid_path = ["/data/SR/Set5/LRbicx3/"]    # 5 pic
    GT_valid_path = ["/data/SR/Set5/original/"]   # 5 pic

    train_ds = datasetSR(lq_paths=LQ_train_path, gt_paths=GT_train_path)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    valid_ds = datasetSR(lq_paths=LQ_valid_path, gt_paths=GT_valid_path, valid=True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=1)

    model_save_path = 'pretrained_weight/swinir/bullshit_training_swinir_x3_1.pth'

    # load pretrained model (from check point)
    # model = torch.load(model_save_path)

    # training
    print(f'================================TRAINING===============================')
    if args.train:
        model.train()
        min_loss = None
        time_start = datetime.now()
        for epoch in range(epochs):
            for idx, data in enumerate(train_dl):
                img_lq = data[0].to(device)
                img_gt = data[1].to(device)
                
                preds   = model(img_lq)
                loss    = criterion(preds, img_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # save model
                if min_loss == None or loss < min_loss:
                    min_loss = loss
                    torch.save(model, model_save_path)
                    logging.info(f'Epoch: {str(epoch):4s}, iter: {str(idx):5s}, Loss: {loss:.7f}, time: {str(time_period):.11s}')

                # display current information
                if idx % 30 == 0:
                    time_period = datetime.now() - time_start
                    print(f'Epoch: {str(epoch):4s}\t, iter: {str(idx):6s}, Loss: {loss:.7f}, time: {str(time_period):.11s}')

        print(f'Minimum Loss: {min_loss}, total training time: {str(datetime.now() - time_start):.11s}')
        logging.info(f'Minimum Loss: {min_loss}, total training time: {str(datetime.now() - time_start):.11s}')



    # evaluating
    print(f'================================EVALUATION=============================')
    psnrs = []
    window_size = 8

    model = torch.load(model_save_path)
    print('Original model after training:')
    logging.info('Original model after training:')
    model.eval()
    for idx, data in enumerate(valid_dl):
        img_name = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            preds = (model(img_lq)[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()

        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()
        psnr = util.psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        logging.info(f'Tesing: {idx}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean()}\n')
    logging.info(f'\nAverage PSNR: {psnrs.mean()}\n')


    print('Model with best loss after training:')
    print(f'Loading from: {model_save_path}')
    logging.info('Model with best loss after training:')
    model = torch.load(model_save_path)
    model.eval()
    psnrs = []
    for idx, data in enumerate(valid_dl):
        img_name = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            output = model(img_lq)
            preds = (output[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()
            # save result img
            save_img = (output.clamp(0, 1) * 255).round().data.squeeze().cpu().numpy()
            if save_img.ndim == 3:
                save_img = np.transpose(save_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            save_img = save_img[:h_old*args.scale, :w_old*args.scale].astype(np.uint8)  # float32 to uint8
            cv2.imwrite(f'results/bullshit_result/{img_name}', save_img)

        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()

        psnr = util.psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        logging.info(f'Tesing: {idx}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean()}\n')
    logging.info(f'\nAverage PSNR: {psnrs.mean()}\n')
    
    logging.info('Finished !!')

