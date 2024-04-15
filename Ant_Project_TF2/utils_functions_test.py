'''
This file contains all the external functions to be called by the training and testing modules

'''
'''defining the libraries'''
import os
from os import listdir
import numpy as np
import pandas as pd
import shutil
import sys
import skimage.io
import random
from PIL import Image, ImageDraw, ImageFont
from xml.etree import ElementTree as ET
import imquality.brisque as brisque
import testing_config
import tensorflow as tf
# Defining global variables

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# ROOT_DIR = r'C:\Users\20200157'
ROOT_DIR = os.path.join(ROOT_DIR, "Ant_project_TF2")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
test_dir =  os.path.join(ROOT_DIR,"./Test_Sample_old")

# open it for the prev good results trained on 100 epochs #ROOT_DIR_path = "F:\Fatima\Ant_dataset"
ROOT_DIR_path = "F:\Fatima\Ant_dataset_new_trial"
#ROOT_DIR_path = "F:\Fatima\Ant_dataset_new_updated"
#Target_path = "F:\Anotated_Ants"

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#print(ROOT_DIR)


sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.utils import compute_ap, compute_recall
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.utils import Dataset
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


import Anoplolepis_config
import Pheidole_config
import Lepisiota_config
import L_Humile_config
import Solenopsis_config
import Wasmannia_config
#import load_models

################################################################
######## Checking for valid image files
#################################################################
def valid_file_input(filename):
    if os.path.exists(filename) and filename.endswith(('.jpg', '.JPG', '.JPEG','.jpeg', '.tiff', 'tif')):
            return True
    else:
        return False
    return
################################################################
######## Checking for Target Species
#################################################################
target_sp =['Anoplolepis_gracilipes','Pheidole_megacephala','Lepisiota_frauenfeldi',
            'Linepithema_humile','Solenopsis_Target','Wasmannia_auropunctata']
def target_species(sp_name):
    if sp_name in target_sp:
        return True
    else:
        return False
################################################################
######## Calling to detect species based on single input file name
#################################################################
    
def detect_species(filename):
    print(filename)
    image = skimage.io.imread(filename)
    if valid_file_input(filename):
        qlt_score = brisque.score(image)
        print(qlt_score)
        if qlt_score <50:
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
            #                                class_names, r['scores'])
            #call sub models for varification
            idr = r['class_ids']
            cln = []
            for a in range(len(idr)):
                id_int = int(idr[a])
                cln.append(main_class_names[id_int])
            if len(cln) == 0:
                print('No ant detected')
            else:
                sp_name, rois, score = utils_functions_test.call_submodel_camera_image(cln, image, anp_model, ph_model, 
                                                                                 lep_model, lin_model, sol_model, was_model)
                print('Detected Species: ', sp_name)
                print('Region of interest: ', rois)
                print('Confidence score: ', score)
        else:
            print('The image quality is not acceptable. Please try again')
    else:
        print("Please input a valid image file")
    return    
######################################################################################
########## RUNNING SUB MODELS FOR ACTUAL CLASS IDENTIFICATION - single Image #########
######################################################################################

def call_submodel_camera_image(cln, image, anp_model, ph_model, lep_model, lin_model, sol_model, was_model, verbose=1):
    #while True:
    if "Anoplolepis_gracilipes" in cln:
        print("Anoplolepis_gracilipes is present in the detected classes")
        class_names = ['BG','Anoplolepis_gracilipes', 'Target_Ants','Non_Target_Ants']
        new_results = anp_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']
    if "Pheidole_megacephala" in cln:
        print("pheidole is present in the detected classes")
        class_names = ['BG','Pheidole_megacephala','Target_Ants', 'Non_Target_Ants']
        new_results = ph_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']

    if "Lepisiota_frauenfeldi" in cln:
        print("Lepisiota_frauenfeldi is present in the detected classes")
        class_names = ['BG','Lepisiota_frauenfeldi','Target_Ants','Non_Target_Ants']
        new_results = lep_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']

    if "Linepithema_humile" in cln:
        print("Linepithema_humile is present in detected classes")
        class_names = ['BG','Linepithema_humile', 'Target_Ants','Non_Target_Ants']
        new_results = lin_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']

    if "Solenopsis_Target" in cln:
        print("Solenopsis_Target is present in the detected classes")
        class_names = ['BG', 'Solenopsis_Target','Target_Ants', 'Non_Target_Ants']
        new_results = sol_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']

    if "Wasmannia_auropunctata" in cln:
        print("Wasmannia_auropunctata is present in the detected classes")
        class_names = ['BG', 'Wasmannia_auropunctata','Target_Ants', 'Non_Target_Ants']
        new_results = was_model.detect([image], verbose=verbose)
        new_r = new_results[0]
        idr = new_r['class_ids']
    spname = []
    for i in range(len(idr)):
        spname.append((class_names[idr[i]]))
    #visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'],
    #                        class_names, new_r['scores'], title="Predictions")
    
    return(spname, new_r['rois'], new_r['scores'])
