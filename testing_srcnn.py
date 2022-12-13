import argparse
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from collections import OrderedDict
import os
import time
import PIL.Image as pil_image

from models.srcnn import SRCNN

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))

def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    save_dir = f'results/srcnn_{args.testset}_x{args.scale}'
    folder = args.folder_gt
    border = args.scale

    return folder, save_dir, border


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq, img_gt

def test(img_lq, model):
    return model(img_lq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='pretrained_weight/srcnn/srcnn_x3.pth')
    # parser.add_argument('--image-file', type=str)
    parser.add_argument('--testset', type=str, default='Set5')
    parser.add_argument('--folder_lq', type=str, default='testsets/Set5/LRbicx3', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/Set5/original', help='input ground-truth test image folder')
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()


    folder, save_dir, border = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    psnrs = []

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):

        # image = pil_image.open(args.image_file).convert('RGB')
        (imgname, imgext) = os.path.splitext(os.path.basename(path))
        image = pil_image.open(f'{args.folder_lq}/{imgname}{imgext}').convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        # print(f'img size: {image_width} x {image_height}')
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        # image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr = calc_psnr(y, preds)

        # print('PSNR: {:.2f}'.format(psnr))
        print('{:10s}, PSNR_Y: {:.2f}, img_size: {:3d} x {:3d}'.format(imgname, psnr, image_height, image_width))
        psnrs.append(psnr)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        # output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

    psnrs = torch.tensor(psnrs)
    print(f'Average PSNR_Y: {psnrs.mean()}')
