import numpy as np
import cv2
import os
import glob
import torch
import time



def pad_img(img_path, patch_size=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) # CHW

    h_old, w_old, c_old = img.shape
    # print("old: ", h_old, w_old, c_old)
    h_new = (h_old // patch_size + 1) * patch_size
    w_new = (w_old // patch_size + 1) * patch_size
    # print("new: ", h_new, w_new, c_old)

    img = np.concatenate((img, np.flip(img, 0)), 0)[:h_new, :, :]
    img = np.concatenate((img, np.flip(img, 1)), 1)[:, :w_new, :]
    # print('return img: ', img.shape)
    return img



if __name__ == '__main__':

    # LR x3 directory
    LR_bic_path = "/data/SR/DIV2K/DIV2K_valid_LR_bicubic/X3/"
    LR_unk_path  = "/data/SR/DIV2K/DIV2K_valid_LR_unknown/X3/"

    # HR directory
    HR_path = "/data/SR/DIV2K/DIV2K_valid_HR/"


    # generate LRx3 sub images of size (256 x 256) bicubic
    save_dir = "/data/div2k/valid_LRbicx3/"
    patch_size = 256
    image_list = sorted(glob.glob(os.path.join(LR_bic_path, '*')))
    for image_path in image_list:
        (imgname, imgext) = os.path.splitext(os.path.basename(image_path))
        img = pad_img(image_path, patch_size=patch_size)

        h, w, c = img.shape
        stride = patch_size
        h_idx_list = list(range(0, h-patch_size, stride)) + [h-patch_size]
        w_idx_list = list(range(0, w-patch_size, stride)) + [w-patch_size]
        print('LR: ', imgname, cv2.imread(image_path, cv2.IMREAD_COLOR).shape, ' --> ', img.shape, f'total: {len(h_idx_list) * len(w_idx_list)} patches')

        idx = 0
        for h_start in h_idx_list:
            for w_start in w_idx_list:
                img_patch = img[h_start:h_start+patch_size, w_start:w_start+patch_size, :]
                out_patch_name = f'{imgname}_{idx}{imgext}'
                out_patch_path = os.path.join(save_dir, out_patch_name)
                cv2.imwrite(out_patch_path, img_patch)
                idx += 1



    # generate LRx3 sub images of size (256 x 256) unknown
    save_dir = "/data/div2k/valid_LRunkx3/"
    patch_size = 256
    image_list = sorted(glob.glob(os.path.join(LR_unk_path, '*')))
    for image_path in image_list:
        (imgname, imgext) = os.path.splitext(os.path.basename(image_path))
        img = pad_img(image_path, patch_size=patch_size)

        h, w, c = img.shape
        stride = patch_size
        h_idx_list = list(range(0, h-patch_size, stride)) + [h-patch_size]
        w_idx_list = list(range(0, w-patch_size, stride)) + [w-patch_size]
        print('LR: ', imgname, cv2.imread(image_path, cv2.IMREAD_COLOR).shape, ' --> ', img.shape, f'total: {len(h_idx_list) * len(w_idx_list)} patches')

        idx = 0
        for h_start in h_idx_list:
            for w_start in w_idx_list:
                img_patch = img[h_start:h_start+patch_size, w_start:w_start+patch_size, :]
                out_patch_name = f'{imgname}_{idx}{imgext}'
                out_patch_path = os.path.join(save_dir, out_patch_name)
                cv2.imwrite(out_patch_path, img_patch)
                idx += 1



    # generate HR sub images of size (768, 768)
    save_dir = "/data/div2k/valid_original/"
    patch_size = 768
    image_list = sorted(glob.glob(os.path.join(HR_path, '*')))
    for image_path in image_list:
        (imgname, imgext) = os.path.splitext(os.path.basename(image_path))
        img = pad_img(image_path, patch_size=patch_size)

        h, w, c = img.shape
        stride = patch_size
        h_idx_list = list(range(0, h-patch_size, stride)) + [h-patch_size]
        w_idx_list = list(range(0, w-patch_size, stride)) + [w-patch_size]
        print('HR: ', imgname, cv2.imread(image_path, cv2.IMREAD_COLOR).shape, ' --> ', img.shape, f'total: {len(h_idx_list) * len(w_idx_list)} patches')

        idx = 0
        for h_start in h_idx_list:
            for w_start in w_idx_list:
                img_patch = img[h_start:h_start+patch_size, w_start:w_start+patch_size, :]
                out_patch_name = f'{imgname}_{idx}{imgext}'
                out_patch_path = os.path.join(save_dir, out_patch_name)
                cv2.imwrite(out_patch_path, img_patch)
                idx += 1
