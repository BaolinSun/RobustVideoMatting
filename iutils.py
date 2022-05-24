import os
import cv2
import shutil
import torch

# import numba as nb
import numpy as np
# import scipy.io as sio
# import minpy.numpy as np
import pandas as pd


from skimage import measure
from tqdm import tqdm
from PIL import Image

import time

def result2image(src, pha, img_path=None, output_source='data/output'):

    start_time = time.time()
    
    pha = pha.squeeze(0)
    
    if pha.is_floating_point():
        pha = pha.mul(255).byte()
    
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    pha = pha[:, :, 0]

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    Image.fromarray(binary, mode='L').save(os.path.join(output_source, img_path))

    # print(time.time() - start_time)


    # src = src.squeeze(0)
    # if src.is_floating_point():
    #     src = src.mul(255).byte()
    # src = np.transpose(src.cpu().numpy(), (1, 2, 0))
        


def result2image_old(src, pha, img_path=None, output_source='data/output'):

    color_masks = [
        np.random.randint(0, 256, (1, 3))
        for _ in range(10)
    ]

    start_time = time.time()

    src = src.squeeze(0)
    pha = pha.squeeze(0)

    pha.Resize()

    if src.is_floating_point():
        src = src.mul(255).byte()
    if pha.is_floating_point():
        pha = pha.mul(255).byte()

    src = np.transpose(src.cpu().numpy(), (1, 2, 0))
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    pha = cv2.cvtColor(pha, cv2.COLOR_RGB2BGR)
    pha = cv2.cvtColor(pha, cv2.COLOR_BGR2GRAY)

    # pha = np.zeros((2160, 3840), dtype=np.uint8)

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)

    
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    print(time.time() - start_time)

    # img_show = src.copy()
    img_show = np.zeros_like(label, dtype=np.uint8)
    for i in range(num):
        cur_mask_bool = (label == (i+1)).astype(np.bool)
        if cur_mask_bool.sum() < 10000:
            continue

        img_show[cur_mask_bool] = label[cur_mask_bool] * 255
        
        # img_show[cur_mask_bool] = src[cur_mask_bool] * 0.5 + color_masks[i] * 0.5

    # img_show = cv2.resize(img_show, (src.shape[1], src.shape[0]))
    

    # cv2.imwrite(os.path.join(output_source, img_path), img_show)


def run_eval_miou_simple(pha_file, mask_file):
    pha_list = os.listdir(pha_file)
    mask_list = os.listdir(mask_file)
    pha_list.sort()
    mask_list.sort()

    iou = []
    for fmask in tqdm(mask_list):
        pha = cv2.imread(os.path.join(pha_file, fmask))
        mask = cv2.imread(os.path.join(mask_file, fmask))

        intersection = np.sum(np.logical_and(mask, pha))
        union = np.sum(np.logical_or(mask, pha))

        if (union == 0) or (intersection == 0):
            continue
        iou_score = intersection / union
        iou.append(iou_score)

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    print('....  ...')

def run_eval_miou_pro(pha_file, mask_file):
    pha_list = os.listdir(pha_file)
    mask_list = os.listdir(mask_file)
    pha_list.sort()
    mask_list.sort()

    iou = []
    for fmask in tqdm(mask_list):
        pha = cv2.imread(os.path.join(pha_file, fmask))
        mask = cv2.imread(os.path.join(mask_file, fmask))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # cv2.imwrite('tmp.png', mask)
        # break

        intersection = np.sum(np.logical_and(mask, pha))
        union = np.sum(np.logical_or(mask, pha))

        if (union == 0) or (intersection == 0):
            continue
        iou_score = intersection / union
        iou.append(iou_score)

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    print('....  ...')


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

        # check number of channels
        if pic.shape[-3] > 4:
            raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-3]))

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

        # check number of channels
        if pic.shape[-1] > 4:
            raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-1]))

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)

if __name__ == '__main__':
    run_eval_miou_simple('data/val2022/Hisense/label', 'data/output')

    