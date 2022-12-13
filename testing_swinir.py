import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time

from thop import profile

from models.swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--training_patch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='pretrained_weight/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth')
    # 
    parser.add_argument('--testset', type=str, default='Set5')
    parser.add_argument('--folder_lq', type=str, default='testsets/Set5/LRbicx3', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/Set5/original', help='input ground-truth test image folder')
    # 
    parser.add_argument('--tile', type=int, default=256, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=0, help='Overlapping of different tiles')
    args = parser.parse_args()
    args.folder_lq = f'testsets/{args.testset}/LRbicx3'
    args.folder_gt = f'testsets/{args.testset}/original'
    return args


def define_model(args):
    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    save_dir = f'results/swinir_{args.testset}_x{args.scale}'
    folder = args.folder_gt
    border = args.scale
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


# needed to fully understand the code
def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
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

    # setup device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # load pretrained weights
    if os.path.exists(args.model_path):
        print(f'Loading model from:\n {args.model_path}\n')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        assert r.status_code == 200
        print(f'Downloading model from:\n {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        print(f'{args.model_path} downloaded.\n')

    print('=====================================================================')
    img = torch.randn(1, 3, 1280, 720).to(device)
    model = define_model(args)
    model.eval()                    # deactivate dropout & batchnorm
    model = model.to(device)
    macs, params = profile(model, inputs=(img, ))
    print(f'FLOPs: {macs * 2 / 1e9} G, for input size: {img.shape}')
    print(f'')
    print('=====================================================================')

    # setup folder and path
    # folder        = ground_truth directory
    # save_dir      = result directory
    # border        = args.scale
    # window_size   = 8
    folder, save_dir, border, window_size = setup(args) 
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    psnr, ssim = 0, 0



    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):

        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            t1 = time.time()
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            t2 = time.time()
            time_info = t2 - t1


        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

            print('Testing {:4d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f};'
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))


    # summarize psnr/ssim & time info
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}, time: {:.5f} sec'.format(save_dir, ave_psnr, ave_ssim, time_info))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


    

if __name__ == '__main__':
    main()