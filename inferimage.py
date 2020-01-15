from __future__ import print_function
import paddle.fluid as fluid
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ppdet.utils.coco_eval import get_category_info
import time

def Permute(im, channel_first=True, to_bgr=False):
    if channel_first:
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def DecodeImage(im_path):
    with open(im_path, 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print("DecodeImage shape ", im.shape)
    return im


def ResizeImage(im, target_size=608, max_size=0):
    if len(im.shape) != 3:
        raise ImageError('image is not 3-dimensional.')
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if float(im_size_min) == 0:
        raise ZeroDivisionError('min size of image is 0')
    if max_size != 0:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])

    im = cv2.resize(
             im,
             None,
             None,
             fx=im_scale_x,
             fy=im_scale_y,
             interpolation=2)
    h, w, c = im.shape
    im_info = np.array([h, w, im_scale_x]).astype(np.float32)
    return im, im_info


def NormalizeImage(im,mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True):
    im = im.astype(np.float32, copy=False)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    if is_scale:
        im = im / 255.0
    im -= mean
    im /= std
    return im


def Prepocess(img_path):
    test_img = DecodeImage(img_path)
    img_shape = test_img.shape[:2]
    test_img = NormalizeImage(test_img)
    test_img, im_info = ResizeImage(test_img, target_size=800, max_size=1333)
    test_img = Permute(test_img)
    test_img = test_img[np.newaxis,:]  # reshape, [1, C, H, W]
    return img_shape,test_img, im_info

path='/home/Documents/PaddleDetection/004696.jpg'
shape,test_img, im_info = Prepocess(path)
list=[]
image={'im_shape': 23, 'im_id': 2,'im_info': 3, 'image': 3}
print(type(test_img.shape),type(im_info),type(test_img))
image['im_shape']=shape
image['im_info']=im_info
image['im_id']=1
image['image']=test_img
list.append(image)
print(list)

'''
path='/home/zengyihui/Documents/1.jpg'
test_img, im_info = Prepocess(path)
image={'im_shape': 23, 'im_id': 2,'im_info': 3, 'image': 3}
image['im_shape']=test_img.shape
image['im_info']=im_info
image['im_id']=1
image['image']=test_img
list.append(image)
print(list)
'''

