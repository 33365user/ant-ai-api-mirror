import warnings

warnings.filterwarnings('ignore', module="tensorflow")
# download from https://www.python.org/downloads/release/python-3913/


import os
import sys
import cv2
import numpy as np
import pandas as pd
import random
import PIL
from queue import Queue
import threading

import time
#from IPython.display import clear_output

import skimage.io
from skimage.transform import rescale
import sklearn
import json

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
ROOT_DIR = os.path.join(ROOT_DIR, "Ant_project_TF2")

# Directory to retrieve logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# print(ROOT_DIR)
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.parallel_model import ParallelModel
import Ants_config_module
import utils_functions_test
import testing_config
import load_models
import tensorflow.compat.v1 as tf
# from loggermod import log_it
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt

#rescale()
def rescale_wrapper(image, *args, channel_axis=None, multichannel=None, **kwargs):
    print(type(image))
    imaget: np.ndarray = image
    print(imaget.shape)

    if channel_axis is None:
        if multichannel is False:
            return rescale(image, *args, channel_axis=None, **kwargs)
        else:
            n = len(imaget.shape)  # shape=(1,2,3,...,n)
            return rescale(image, *args, channel_axis=n-1, **kwargs)
    else:
        return rescale(image, *args, channel_axis=channel_axis, **kwargs)

# override skimage rescale to handle incompatibilities
#skimage.transform.rescale = rescale_wrapper


print(os.getcwd())
import brisque
#from imquality import brisque
# %matplotlib inline

# Defining global variables


# Importing Model and configurations

main_class_names = testing_config.main_class_names
class_names = testing_config.class_names


# WEIGHT_PATH = .h5 models loaded.
#config: Main ants configuration used while training the models
#cfg: Testing configuration for prediction
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import copy


# force load prediction stuffs
#model.keras_model._make_predict_function()
#anp_model.keras_model._make_predict_function()
#ph_model.keras_model._make_predict_function()
#lep_model.keras_model._make_predict_function()
#lin_model.keras_model._make_predict_function()
#sol_model.keras_model._make_predict_function()
#was_model.keras_model._make_predict_function()

modelsQ = Queue()

"""
def make_models_handler(maxim=6):
    global modelsQ
    while True:
        if modelsQ.qsize() < maxim:
            model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model = load_models.load_all_models()
            models = (model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model)
            modelsQ.put(models)


t = threading.Thread(target=make_models_handler)
t.start()
"""


class Data:
    def __init__(self, code, species_list, regionsofinterest_list, score_list):
        self.code = code
        self.species_list = species_list
        self.regionsofinterest = regionsofinterest_list
        self.score_list = score_list

    def encode(self):
        return json.dumps(self.__dict__)

    def save_temp(self, filepath):
        identity = os.path.basename(filepath).split('.')[0]
        logging.info(f"{identity} | Finished Classification! Saving to Temp...")
        fpath = os.path.abspath(os.path.join(os.getcwd(), "..", "temp"))
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        filename = f'temp_{identity}.json'
        fullfpath = os.path.join(fpath, filename)
        with open(fullfpath, 'w') as f:
            f.write(self.encode())
        print(f"SUBPROC_OUT_V1_CONF_CODE_001={fullfpath}")
        return self.encode()


def detect_species(filename):
    global modelsQ
    # print("Running Detection on input file. Please be patient!")
    fname=os.path.basename(filename)
    if utils_functions_test.valid_file_input(filename):
        image = skimage.io.imread(filename)
        print(f"{fname} | Image Read")
        qlt_score = brisque.score(image)
        print(f"{fname} | Brisque Score Complete")
        if qlt_score < 60:
            config = Ants_config_module.AntsConfig()
            cfg = testing_config.PredictionConfig()
            print("Loading Models...")
            # nb: model is the slowest - seems to be region detection engine.
            model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model = load_models.load_all_models()
            # model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model = modelsQ.get()
            print(f"{fname} | Valid Source Image - Now Detecting")
            # conf = model.keras_model.get_config()
            # print(model.keras_model.inner_model)
            results = model.detect([image], verbose=0)
            print(f"{fname} | Regions of Interest: {results[0]['rois']}")
            # Visualize results
            r = results[0]
            idr = r['class_ids']
            cln = []
            if len(idr) == 0:
                # print('The image quality is not acceptable. Please try again')
                del anp_model, ph_model, lep_model, lin_model, sol_model, was_model, model
                return Data(-2, ["INVALID"], [None], [0]).save_temp(filename)
            else:
                print(f"{fname} | Quality Valid")
                idr = int(max(set(idr)))
                cln = main_class_names[idr]
                if utils_functions_test.target_species(cln):
                    print(f"{fname} | Target Species Detected in Image")
                    sp_name, rois, score = utils_functions_test.call_submodel_camera_image(cln, image, anp_model, ph_model, 
                                                                                 lep_model, lin_model, sol_model, was_model, verbose=0)
                    # print('Detected Species: ', sp_name)
                    # print('Region of interest: ', rois)
                    # print('Confidence score: ', score)
                    rois: np.ndarray = rois
                    score: np.ndarray = score
                    rois = rois.tolist()
                    score = score.tolist()
                    del anp_model, ph_model, lep_model, lin_model, sol_model, was_model, model
                    return Data(1, sp_name, rois, score).save_temp(filename)
                # else:
                    # print("Non-Target Ants detected. No further evaluation at this point")
                    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                # main_class_names, r['scores'], title="Predictions")
                else:
                    del anp_model, ph_model, lep_model, lin_model, sol_model, was_model, model
                    return Data(0, idr, r['rois'].tolist(), r['scores'].tolist()).save_temp(filename)
        else:
            # print('The image quality is not acceptable. Please try again')
            return Data(2, ["INVALID"], [None], [0]).save_temp(filename)
    else:
        # print("Please input a valid image file")
        return Data(-1, ["INVALID"], [None], [0]).save_temp(filename)


if True:
    detect_species(sys.argv[1])