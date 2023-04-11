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
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--model_path', type=str, default=None, required=True)
    # 
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--savedir', type=str, required=True)
    # 
    parser.add_argument('--tile', type=int, default=256, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=0, help='Overlapping of different tiles')
    args = parser.parse_args()
    return args


# needed to fully understand the code
def test(img_lq, model, args):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        # assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output



def main():
    args = parse()

    # basic setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = torch.randn(1, 3, 256, 256).to(device)
    model = LapSUNET(dim=12, depth=2, num_heads=3, mlp_ratio=2).to(device)
    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')

    # load from best
    model_path = args.model_path
    if os.path.exists(model_path):
        print(f'Loading from {model_path}')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])


    # make savedir
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
        print(f'making directory: {args.savedir}')

    # inference
    print('start testing ->')
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.testset, '*')))):
        imgname, imgext = os.path.splitext(os.path.basename(path))
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{args.savedir}/{imgname}{imgext}', output)

        print('Testing {:d} {:20s}'.format(idx, imgname))
        


if __name__ == '__main__':
    main()