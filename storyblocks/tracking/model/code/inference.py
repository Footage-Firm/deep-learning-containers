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
import tempfile

import boto3
import torch

from sagemaker_inference import content_types, errors

# path munging to access the neighboring YAEDP repo
HERE = os.path.dirname(os.path.realpath(__file__))
TRACKING_ROOT = os.path.join(HERE, 'tracking')
sys.path.insert(0, TRACKING_ROOT)

# todo: custom imports from the tracking root go here

# todo: add deploy-time environment variables
# if we want to be able to parameterize certain things at the time we deploy the model but NOT
# after that (e.g. default values for arguments that parameterize the MODEL rather than the
# REQUEST), we can do that here. A use case example would be something like wanting the same
# container to possibly deploy different endpoints that have different model params, like different
# backbones or embedding dimension sizes -- things the requester can't ask for. When in doubt, KISS
#
# example:
#   >>> MY_DEPLOY_TIME_VAR = os.environ.get('MY_DEPLOY_TIME_VAR', 'my_default_value')

USE_CUDA = torch.cuda.is_available()

print(f'USE_CUDA = {USE_CUDA}')


def log_storyblocks_artifact_info():
    """if a face detection repo sha or archive timestamp file exist, print
    their contents"""
    print('looking for archive artifact metadata')
    here = os.path.dirname(os.path.realpath(__file__))

    for fname in ['model_archive_timestamp', 'tracking_current_sha']:
        fname_full = os.path.join(here, fname)
        if os.path.isfile(fname_full):
            with open(fname_full, 'r') as fp:
                print(f"{fname}: {fp.read().strip()}")


log_storyblocks_artifact_info()


class StoryblocksCustomError(errors.GenericInferenceToolkitError):
    def __init__(self, message):
        super().__init__(400, message)


# todo: fill in the model_fn
def model_fn(model_dir):
    # you can ignore model_dir completely
    #
    # I *think* you can return whatever you want -- e.g. a tuple of models you want to load, a dict
    # etc. BUT if you can't, I think the worst case scenario here is still doable, e.g.
    #
    #   class WrappedModel(torch.nn.Module):
    #       def __init__(self):
    #           self.detecter = custom_load_detecter_model_code()
    #           self.tracker = custom_load_tracker_model_code()
    #
    # I think here is where we should
    #   + do all the `exp` and `opt` stuff
    #   + build the `Predictor`
    #   + build the `ByteTracker` (or whichever)
    #
    model = None
    return model


# todo: leaving this here in case the code is useful, will need to change to support video,
#  and maybe not preprocess at all
def load_s3_image_and_preprocess(bucket, key):
    """download the file from s3, process it using the external library

    returns:
        ori_imgs (list of np.ndarray) the original image as a cv2 np array
        framed_imgs (list of np.ndarray) a resized and rescaled version of each image in ori_imgs
        framed_metas (list of tuples) a list of tuples of scaling information, where each tuple
            has components new_w, new_h, old_w, old_h, padding_w, padding_h

    """
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[1]) as ntf:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        with open(ntf.name, 'wb') as fp:
            fp.write(obj['Body'].read())

        return preprocess(ntf.name, max_size=MAX_INPUT_SIZE, mean=PARAMS['mean'], std=PARAMS['std'])


# todo: leaving this code here in case it's useful
def parse_json_input(input_data):
    try:
        j = json.loads(input_data)
        bucket = j['bucket']
        key = j['key']

        # you could do custom user-provided arguments in the json payload here, e.g.
        # >>> param = j.get('param_name', default_param_value)
        return bucket, key  # , param1, param2, ...
    except (KeyError, TypeError):
        raise StoryblocksCustomError(
            f"payloads for content type {content_types.JSON} must have json bodies with keys "
            f"\"key\" and \"bucket\"")


# todo: update this function
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns:


    """
    if content_type == content_types.JSON:
        bucket, key = parse_json_input(input_data)
        # bucket, key, param1, param2, ... = parse_json_input(input_data)

    # you can support other content types if you want, but don't have to!
    # elif content_type == content_types.CSV:
    #     bucket, key = parse_csv_input(input_data)
    else:
        raise errors.UnsupportedFormatError(content_type)

    # do any video / image downloading and processing you need, e.g.
    # ori_imgs, framed_imgs, framed_metas = load_s3_image_and_preprocess(bucket=bucket, key=key)

    # return can be anything you want -- there may be restrictions on whether or not it can be a
    # tensor at this point in time or not, I don't remember
    # return ori_imgs, framed_imgs, framed_metas, pred_threshold, iou_threshold


# todo: fill this in
def predict_fn(data, model):
    """

    Args:
        data: tuple of inputs generated by custom input_fn above
        model: PyTorch model loaded in memory by model_fn

    Returns: a prediction

    """
    # note: data is just whatever `input_fn` returns, and `model` is just whatever `model_fn`
    # returns
    # I think here we basically reproduce the code in `track_video`
