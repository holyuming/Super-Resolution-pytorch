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
from models.unetSR import UNet
from utils import util_calculate_psnr_ssim as util

from thop import profile


def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--trainset', type=str, default='testsets/Set5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--evalset', type=str, default='testsets/Set5')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--savedir', type=str, default=None)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # img = torch.randn(16, 3, 256, 256).to(device)
    img = torch.randn(1, 3, 1280, 720).to(device)

    # our model
    # best psnr: 30.259, dim=18, lr=0.001 & 0.0001 & 0.00001, batch=4, Adam
    model = UNet(dim=18).to(device)

    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')

    # model path
    model_save_path = 'pretrained_weight/unetSR/bullshit_training_unetSR.pth'
    model_checkpoint_path = 'checkpoints/unetSR.pt'

    # basic setup
    best_psnr = 0
    checkpoint = None
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        best_psnr = checkpoint['best_psnr']
        print('Best PSNR: ', best_psnr)
        if input(f'Do you want to load from checkpoint: {model_checkpoint_path} ? (y/n) ') == 'y':
            print(f'Loading state_dict from {model_checkpoint_path}')
            model.load_state_dict(checkpoint['model_state_dict'])

    epochs = args.epochs
    batch_size = args.batch
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    # datasets & dataloaders
    # for each training data --> input size = (B, 3, 256, 256), output size = (B, 3, 768, 768)
    # I generate those 256 x 256 sub images w/ utils/img_proc.py
    LQ_train_path = ['/data/SR/div2k/LRbicx3', '/data/SR/flickr2k/LRbicx3']      # 800 pic + 2650 pic -> 5088 patches + 16377 patches
    GT_train_path = ['/data/SR/div2k/original', '/data/SR/flickr2k/original']    # 800 pic + 2650 pic -> 5088 patches + 16377 patches

    LQ_valid_path = ["/data/SR/Set5/LRbicx3/"]    # 5 pic
    GT_valid_path = ["/data/SR/Set5/original/"]   # 5 pic

    train_ds = datasetSR(lq_paths=LQ_train_path, gt_paths=GT_train_path)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    valid_ds = datasetSR(lq_paths=LQ_valid_path, gt_paths=GT_valid_path, valid=True)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=1)

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
                            preds = (model(img_lq)[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()

                        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()
                        psnr = util.psnr_tensor(preds, img_gt)
                        psnrs.append(psnr)
                        # print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
                        
                    psnrs = torch.tensor(psnrs)
                    # print(f'\nAverage PSNR: {psnrs.mean()}\n')
                    print(f'Epoch: {str(epoch):4s}, iter: {str(idx):6s}, Loss: {loss:.7f}, time: {str(time_period):.11s}, average psnr: {psnrs.mean()}')

                    # save checkpoint
                    if psnrs.mean() > best_psnr:
                        best_psnr = psnrs.mean()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'best_psnr': best_psnr,
                            'optimizer_state_dict': optimizer.state_dict()
                        }, model_checkpoint_path)

                    model.train()
                    # ==============================================================
                    # ==============================================================

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
            preds = (model(img_lq)[:, :, :h_old*args.scale, :w_old*args.scale].clamp(0, 1) * 255).round()

        img_gt = (img_gt[:, :, :h_old*args.scale, :w_old*args.scale] * 255.).round()
        psnr = util.psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean()}\n')


    print('Model with best loss after training:')
    # print(f'Loading from: {model_save_path}')
    # model = torch.load(model_save_path)

    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        print(f'Loading state_dict from {model_checkpoint_path}')
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

            output = model(img_lq)
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