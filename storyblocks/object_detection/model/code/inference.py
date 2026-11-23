#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: inference
Created: 2020-08-19

Description:

    inference functions for pytorch serving container

"""
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
import cv2
import numpy as np
import torch
import yaml

from sagemaker_inference import content_types, encoder, errors

# path munging to access the neighboring YAEDP repo
HERE = os.path.dirname(os.path.realpath(__file__))
YAEDP_ROOT = os.path.join(HERE, 'Yet-Another-EfficientDet-Pytorch')
sys.path.insert(0, YAEDP_ROOT)

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess, preprocess_video

# model version is chosen at model startup from the invocation arguments
EFFICIENTDET_COMPOUND_COEF = int(os.environ.get('EFFICIENTDET_COMPOUND_COEF', 0))
DEFAULT_PRED_THRESHOLD = float(os.environ.get('DEFAULT_PRED_THRESHOLD', 0.05))
DEFAULT_IOU_THRESHOLD = float(os.environ.get('DEFAULT_IOU_THRESHOLD', 0.5))
USE_FLOAT16 = os.environ.get('USE_FLOAT16', 'false') == 'true'
# jsonlines requests (batch transform MultiRecord) can carry hundreds of frames. this many go
# through the model at once; raise it until the gpu runs out of memory with the configured number
# of model server workers
GPU_BATCH_SIZE = int(os.environ.get('GPU_BATCH_SIZE', 4))
# frames in a jsonlines request are downloaded from s3 in parallel
S3_DOWNLOAD_THREADS = int(os.environ.get('S3_DOWNLOAD_THREADS', 16))
JSONLINES = 'application/jsonlines'

print(f"EFFICIENTDET_COMPOUND_COEF = {EFFICIENTDET_COMPOUND_COEF}")
print(f"DEFAULT_PRED_THRESHOLD = {DEFAULT_PRED_THRESHOLD}")
print(f"DEFAULT_IOU_THRESHOLD = {DEFAULT_IOU_THRESHOLD}")
print(f"USE_FLOAT16 = {USE_FLOAT16}")
print(f"GPU_BATCH_SIZE = {GPU_BATCH_SIZE}")
print(f"S3_DOWNLOAD_THREADS = {S3_DOWNLOAD_THREADS}")

USE_CUDA = torch.cuda.is_available()

# boto3 clients are thread safe; one per model server worker avoids re-creating it per frame
S3 = boto3.client('s3')

print(f'USE_CUDA = {USE_CUDA}')

# pre and post processing params from YAEDP
with open(os.path.join(YAEDP_ROOT, 'projects', 'coco.yml')) as fp:
    PARAMS = yaml.safe_load(fp)

print(f'PARAMS = {PARAMS}')

# input size is fixed on the compound coef level
MAX_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
MAX_INPUT_SIZE = MAX_INPUT_SIZES[EFFICIENTDET_COMPOUND_COEF]
print(f"MAX_INPUT_SIZE = {MAX_INPUT_SIZE}")


def log_storyblocks_artifact_info():
    """if a face detection repo sha or archive timestamp file exist, print
    their contents"""
    print('looking for archive artifact metadata')
    here = os.path.dirname(os.path.realpath(__file__))

    for fname in ['model_archive_timestamp', 'yaefd_current_sha']:
        fname_full = os.path.join(here, fname)
        if os.path.isfile(fname_full):
            with open(fname_full, 'r') as fp:
                print(f"{fname}: {fp.read().strip()}")


log_storyblocks_artifact_info()


class StoryblocksCustomError(errors.GenericInferenceToolkitError):
    def __init__(self, message):
        super().__init__(400, message)


def get_weights_url(c):
    """this list of urls is hand-created from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/readme.md as of the
    time of writing this file (2020-12-03). as the model is updated, these urls *could* (but
    likely won't) change much. of course we may need to update this over time

    this could be automated with the github releases api, but that's overkill rn
    https://developer.github.com/v3/repos/releases/

    Args
        c: the compound coefficient (read the efficientnet/det papers for details)

    Returns
        url of the weights file to download

    """
    url_map = {
        '0': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d0.pth',
        '1': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d1.pth',
        '2': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d2.pth',
        '3': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d3.pth',
        '4': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d4.pth',
        '5': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d5.pth',
        '6': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d6.pth',
        '7': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
             '/efficientdet-d7.pth',
        '8': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
             '/efficientdet-d8.pth',
        '7x': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
              '/efficientdet-d8.pth', }

    c = str(c)
    try:
        return url_map[c]
    except KeyError:
        raise StoryblocksCustomError(
            "compound coefficient not found, cannot download model weights")


def timer(func):
    def wrapped_func(*args, **kwargs):
        t0 = datetime.datetime.now()
        print(f'entering {func.__name__} at {t0}')
        x = func(*args, **kwargs)
        t1 = datetime.datetime.now()
        print(f'exiting {func.__name__} at {t1}')
        print(f'total time in {func.__name__}: {t1 - t0}')
        return x

    return wrapped_func


@timer
def model_fn(model_dir, context=None):
    # based entirely off of
    # https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/coco_eval.py
    # model_archive_build.sh bundles the weights into model_dir, where load_state_dict_from_url
    # finds them without downloading. without them, every model server worker on every instance
    # downloads from github on startup
    print(f'building and loading efficientdet d{EFFICIENTDET_COMPOUND_COEF}')
    model = EfficientDetBackbone(compound_coef=EFFICIENTDET_COMPOUND_COEF,
                                 num_classes=len(PARAMS['obj_list']),
                                 ratios=eval(PARAMS['anchors_ratios']),
                                 scales=eval(PARAMS['anchors_scales']))
    state_dict = torch.hub.load_state_dict_from_url(
        url=get_weights_url(c=EFFICIENTDET_COMPOUND_COEF),
        model_dir=model_dir,
        map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()

    if USE_CUDA:
        model.cuda(0)

    if USE_FLOAT16:
        model.half()

    return model


def load_s3_image(bucket_and_key):
    """download an image from s3 and decode it the same way cv2.imread would

    Args:
        bucket_and_key: tuple of (bucket, key)

    Returns:
        np.ndarray: the image as a cv2 (BGR) array

    """
    bucket, key = bucket_and_key
    body = S3.get_object(Bucket=bucket, Key=key)['Body'].read()
    img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise StoryblocksCustomError(f"could not decode image s3://{bucket}/{key}")
    return img


def load_s3_images_and_preprocess(bucket_keys):
    """download the files from s3 in parallel, process them using the external library

    returns:
        ori_imgs (list of np.ndarray) the original images as cv2 np arrays
        framed_imgs (list of np.ndarray) a resized and rescaled version of each image in ori_imgs
        framed_metas (list of tuples) a list of tuples of scaling information, where each tuple
            has components new_w, new_h, old_w, old_h, padding_w, padding_h

    """
    with ThreadPoolExecutor(max_workers=S3_DOWNLOAD_THREADS) as pool:
        imgs = list(pool.map(load_s3_image, bucket_keys))

    # preprocess_video is preprocess for in-memory images; mean and std are passed explicitly
    # because its defaults are in a different channel order
    return preprocess_video(*imgs, max_size=MAX_INPUT_SIZE, mean=PARAMS['mean'], std=PARAMS['std'])


def parse_json_input(input_data):
    try:
        j = json.loads(input_data)
        bucket = j['bucket']
        key = j['key']
        pred_threshold = j.get('pred_threshold', DEFAULT_PRED_THRESHOLD)
        assert 0 <= pred_threshold <= 1
        iou_threshold = j.get('iou_threshold', DEFAULT_IOU_THRESHOLD)
        assert 0 <= iou_threshold <= 1
        return bucket, key, pred_threshold, iou_threshold
    except (KeyError, TypeError):
        raise StoryblocksCustomError(
            f"payloads for content type {content_types.JSON} must have json bodies with keys "
            f"\"key\" and \"bucket\"")
    except AssertionError:
        raise StoryblocksCustomError(
            "provided threshold values must be between 0 and 1 (inclusive)")


@timer
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    this is custom in that it doesn't do the pytorch tensor serialization here but instead lets
    the Yet-Another-EfficientDet-Pytorch module handle necessary preprocessing in the predict step.

    additionally, we hijack the 'text/csv' and 'application/json' content_type modes:

        + content_type == 'text/csv': treat `input_data` as an s3 path `{bucket}/{key path}` and
            load the image directly using boto3 and cv2.
        + content_type == 'application/json': parse `input_data` for a `bucket` and a `key`
            attribute, and do the same as we do in CSV. additionally allows users to parameterize
            the prediction threshold and iou threshold values with keys `pred_threshold` and
            `iou_threshold`

    We lose the built-in functionality for CSV, but it is arguably the worst way to transfer the
    data anyway (npy is much better, and json at least allows the multi-dimensional arrays to
    stay multi-dimensional).

    note: application/json will also probably fail and receive a

        HTTP/1.1 413 Request Entity Too Large

    error (a standard size image as json is about 30M)

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns:
        ori_imgs (list of np.ndarray) the original image as a cv2 np array
        framed_imgs (list of np.ndarray) a resized and rescaled version of each image in ori_imgs
        framed_metas (list of tuples) a list of tuples of scaling information, where each tuple
            has components new_w, new_h, old_w, old_h, padding_w, padding_h
        pred_threshold (float): threshold for filtering detection predictions
        iou_threshold (float): threshold for iou filtering in nms postprocessing of detections

    """
    if content_type == content_types.CSV:
        i = input_data.find('/')
        records = [(input_data[:i], input_data[i + 1:], DEFAULT_PRED_THRESHOLD,
                    DEFAULT_IOU_THRESHOLD)]
    elif content_type == content_types.JSON:
        records = [parse_json_input(input_data)]
    elif content_type == JSONLINES:
        records = [parse_json_input(line) for line in _decode(input_data).splitlines()
                   if line.strip()]
    else:
        raise errors.UnsupportedFormatError(content_type)

    # postprocess applies one threshold pair to the whole batch
    thresholds = {(pred_threshold, iou_threshold) for (_, _, pred_threshold, iou_threshold)
                  in records}
    if len(thresholds) != 1:
        raise StoryblocksCustomError(
            "every record in a jsonlines request must use the same threshold values")
    (pred_threshold, iou_threshold), = thresholds

    bucket_keys = [(bucket, key) for (bucket, key, _, _) in records]
    ori_imgs, framed_imgs, framed_metas = load_s3_images_and_preprocess(bucket_keys)
    return ori_imgs, framed_imgs, framed_metas, pred_threshold, iou_threshold


def _decode(input_data):
    return input_data.decode('utf-8') if isinstance(input_data, (bytes, bytearray)) else input_data


@timer
def predict_fn(data, model):
    """mostly copied from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet_test.py

    Args:
        data: tuple of inputs generated by custom input_fn above
        model: PyTorch model loaded in memory by model_fn

    Returns: a list with one detections dict per input image

    """
    ori_imgs, framed_imgs, framed_metas, threshold, iou_threshold = data
    out = []
    for start in range(0, len(framed_imgs), GPU_BATCH_SIZE):
        end = start + GPU_BATCH_SIZE
        out.extend(_predict_batch(model, framed_imgs[start:end], framed_metas[start:end],
                                  threshold, iou_threshold))
    return out


def _predict_batch(model, framed_imgs, framed_metas, threshold, iou_threshold):
    x = torch.stack(
        [torch.from_numpy(fi).cuda() if USE_CUDA else torch.from_numpy(fi)
         for fi in framed_imgs],
        0)

    x = x.to(torch.float32 if not USE_FLOAT16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regress_boxes = BBoxTransform()
        clip_boxes = ClipBoxes()

        out = postprocess(x,
                          anchors=anchors,
                          regression=regression,
                          classification=classification,
                          regressBoxes=regress_boxes,
                          clipBoxes=clip_boxes,
                          threshold=threshold,
                          iou_threshold=iou_threshold)
        out = invert_affine(framed_metas, out)
        return out


def output_fn(prediction, accept):
    """jsonlines responses have one line per input image, so batch transform can join each line
    back to its input record. each line is a one-element list, the same shape a single-image json
    response has, so downstream parsing is identical for both batch strategies

    Args:
        prediction: the list of detection dicts from predict_fn
        accept: the requested response content type

    Returns:
        str: the serialized response

    """
    if accept == JSONLINES:
        return '\n'.join(encoder.encode([detections], content_types.JSON)
                         for detections in prediction)
    return encoder.encode(prediction, accept)
