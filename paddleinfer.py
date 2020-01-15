import os
import glob
import numpy as np
from paddle import fluid
from ppdet.core.workspace import create,load_config
from ppdet.modeling.model_input import create_feed
from ppdet.data.data_feed import create_reader

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))
    return images


def getinferdata(image_path):
    cfg = load_config('configs/faster_rcnn_r50_1x.yml')
    test_feed = create(cfg.test_feed)
    test_images = get_test_images(None, image_path)
    test_feed.dataset.add_images(test_images)
    reader = create_reader(test_feed)
    loader, feed_vars = create_feed(test_feed, iterable=True) #ppdet.modeling.model_input
    place =fluid.CPUPlace()
    loader.set_sample_list_generator(reader,place)
    for iter_id, data in enumerate(loader()):
        print(iter_id, data)
        print(type(data[0]['im_shape']),type(data[0]['im_id']),type(data[0]['im_info']),type(data[0]['image']))
        print('im_shape',str(data[0]['im_shape']))
        print('im_id',str(data[0]['im_id']))
        print('im_info',str(data[0]['im_info']))#
        print('image',np.array(data[0]['image']),np.array(data[0]['image']).shape)
        return data
    
path='/home/zengyihui/Documents/PaddleDetection/004696.jpg'
#path='/home/zengyihui/文档/yolov3L/testdata/ls/004186.jpg'
data=getinferdata(path)
print(data)
'''
path='/home/zengyihui/Documents/PaddleDetection/000003.jpg'
data=getinferdata(path)
print(data)
'''


