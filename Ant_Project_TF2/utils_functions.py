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
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
from xml.etree import ElementTree as ET
import testing_config
import tensorflow as tf
# Defining global variables

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# ROOT_DIR = r'C:\Users\20200157'
ROOT_DIR = os.path.join(ROOT_DIR, "Ant_project_TF2")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# open it for the prev good results trained on 100 epochs #ROOT_DIR_path = "F:\Fatima\Ant_dataset"
ROOT_DIR_path = "F:\Fatima\Ant_dataset_new_download"
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

import Ants_config_module
import Anoplolepis_config
import Pheidole_config
import Lepisiota_config
import L_Humile_config
import Solenopsis_config
import Wasmannia_config

#########################################################
###### Collection of data for Dataset creation
#########################################################
def folder_path_def():
    '''
    This function creates folders for the dataset and returns path of destination directories for annotations and images
    
    dest_dir: Path to a folder named 'dataset' on root directory to store images and annotations
    img_dir: Path to a folder inside the dataset directory to hold all the images (.jpg, .png, .tif)
    annot_dir: Path to a folder inside the dataset directory to hold all the annotations (.xml)
    '''
    #Changing the Drive to F:\
    ROOT_DIR_Dataset = os.chdir(ROOT_DIR_path)
    ROOT_DIR_Dataset = os.getcwd()
    #creating a new parents dataset dir on root
    dest_dir = os.path.join(ROOT_DIR_Dataset, 'dataset')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    #creating a new images folder with in the dataset dir
    img_dir = os.path.join(dest_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    #creating a new annotation folder with in the dataset dir
    annot_dir = os.path.join(dest_dir, 'annots')
    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)
    working_path = os.chdir(ROOT_DIR)
    return dest_dir, img_dir, annot_dir


def collecting_data_from_target(Target_path):
    '''
    This function collects all the data from the destination directory and 
    stores it in separate folders in the dataset directory on the root.
    Returns path to detaset folder, images folder and annotations folder in the working directory. 
    
    dest_path: path to dataset
    img_path: path to images folder
    annot_path: path to annotations folder
    count_i: count of all the image files in the target folder
    count_a: count of all the xml files in the target folder
    tagert_folder_path: path to target folder from HBI
    SourceFolder_i: path to an image file in the target folder
    SourceFolder_a: path to a matching xml file in the target folder
    '''
    dest_path, img_path, annot_path = folder_path_def() #creating required directories and paths

    tagert_folder_path = os.chdir(Target_path)
    TARGET_FOLDER = os.getcwd()
    count_i = 0
    count_a = 0
    for root, dirs, files in os.walk((os.path.normpath(TARGET_FOLDER)), topdown=False):
            for name in files:
                if (name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.PNG') or name.endswith('.png') or name.endswith('.tif') or name.endswith('.TIF') or name.endswith('.tiff') or name.endswith('.TIFF')):
                    count_i += 1
                    SourceFolder_i = os.path.join(root,name)
                    shutil.copy2(SourceFolder_i, img_path) #copies jpg and png to new folder
                    #image_con = name[:-4]                    
                if (name.endswith('.xml') or name.endswith('.XML')):
                    count_a +=1
                    SourceFolder_a = os.path.join(root,name)
                    shutil.copy2(SourceFolder_a, annot_path) #copies xml to new folder
    
    #changing the Root directory back to working directory
    working_path = os.chdir(ROOT_DIR)
    return dest_path, img_path, annot_path, count_i, count_a

'''
Supporting functions for cleaning the dataset and stat counting.
Cleaning function checks and remove any image file which does not have a corresponding xml file or vice versa.
Stat counting functions return total number of files with classes for train and test datasets.

'''
def removing_test_data_from_training(imgs_path, annots_path):
    #training_path = os.path.join(ROOT_DIR_path, 'dataset\images')
    for filename in os.listdir(os.path.join(ROOT_DIR,'Test_Sample/images')):
        if os.path.isfile(os.path.join(imgs_path, filename)):
            #a = a+1
            os.remove(os.path.join(imgs_path, filename))
            f_xml = filename.split(".")[0]+ ".xml"
            os.remove(os.path.join(annots_path, f_xml))
    return
def clean_dataset(d_path, i_path, a_path):
    #cleaning the dataset for unavailable XML or JPG files
    print(d_path, i_path, a_path)
    
    new_files = os.listdir(d_path)
    for i in new_files:
        for j in os.listdir(d_path+ '/'+ i):
            if (j.split(".")[0] + ".xml" not in os.listdir(a_path)) or ((j.split(".")[0] + ".jpg" not in os.listdir(i_path)) and (j.split(".")[0] + ".png" not in os.listdir(i_path)) and (j.split(".")[0] + ".tif" not in os.listdir(i_path)) and (j.split(".")[0] + ".JPG" not in os.listdir(i_path)) and (j.split(".")[0] + ".PNG" not in os.listdir(i_path)) and (j.split(".")[0] + ".TIF" not in os.listdir(i_path)) and (j.split(".")[0] + ".TIFF" not in os.listdir(i_path)) and (j.split(".")[0] + ".tiff" not in os.listdir(i_path))):
                #print(d_path+ '/'+i+"/"+j)
                os.remove(d_path+ '/'+i+"/"+j)
    working_path = os.chdir(ROOT_DIR)
    return

def dataset_stats_dt(dt):
    print("STATS FOR TRAINING SET")
    print("Image Count: {}".format(len(dt.image_ids)))
    print("Class Count: {}".format(dt.num_classes))
    for i, info in enumerate(dt.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    working_path = os.chdir(ROOT_DIR)
    return

def dataset_stats_dv(dv):
    print("STATS FOR VALIDATION SET")
    print("Image Count: {}".format(len(dv.image_ids)))
    print("Class Count: {}".format(dv.num_classes))
    for i, info in enumerate(dv.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    working_path = os.chdir(ROOT_DIR)
    return

##########################################################
################Checking any extra file in annot / image folders
##########################################################

def extra_files(im_path, an_path):
    img_files = os.listdir(im_path)
    ann_files = os.listdir(an_path)
    #print(len(img_files))
    #print(len(ann_files))
    im = []
    an = []
    for i in img_files:
        m = i.split(".")[0]
        im.append(m)
        #print(n)
    for j in ann_files:
        n = j.split(".")[0]
        an.append(n)
    print(len(im))
    print(len(an))

    for l in range(len(an)):
        if im[l] != an[l]:
            print('files that do not match are ', im[l])
            print('files that do not match are ', an[l])
    return

##########################################################

##########################################################
######### Counting images for target species##############
##########################################################

def counting_target_species(Target_path):
    '''
    This function counts and display the number of training images available 
    in the Target folders from HBI for each class.
    '''
    tagert_folder_path = os.chdir(Target_path)
    TARGET_FOLDER = os.getcwd()
    print('This is the target folder',TARGET_FOLDER)    
    count = 0

    sp1 = 0 
    sp2 = 0 
    sp3 = 0 
    sp4 = 0 
    sp5 = 0
    sp6 = 0
    sp7 = 0
    target_sp = []
    target_count = []

        # Iterate directory
    for path in os.listdir(TARGET_FOLDER):
        if (path == 'Anoplolepis_gracilipes' or path == 'Lepisiota_frauenfeldi' or path == 'Linepithema_humile' or path == 'Pheidole_megacephala' or path == 'Solenopsis_geminata') or path == 'Solenopsis_invicta' or path == 'Wasmannia_auropunctata':
                target_sp.append(path)

        count = 0
        new_path = os.path.join(TARGET_FOLDER, path)
        for root, dirs, files in os.walk((new_path), topdown=False):
            if 'Photos' in root and path == 'Anoplolepis_gracilipes':
                l = len(files)
                sp1 = sp1+l
            elif 'Photos' in root and path == 'Lepisiota_frauenfeldi':
                l = len(files)
                sp2 = sp2+l
            elif 'Photos' in root and path == 'Linepithema_humile':
                l = len(files)
                sp3 = sp3 +l
            elif 'Photos' in root and path == 'Pheidole_megacephala':
                l = len(files)
                sp4 = sp4+l
            elif 'Photos' in root and path == 'Solenopsis_geminata':
                l = len(files)
                sp5 =sp5+l
            elif 'Photos' in root and path == 'Solenopsis_invicta':
                l = len(files) 
                sp6 =sp6+l
            elif 'Photos' in root and path == 'Wasmannia_auropunctata':
                l = len(files)
                sp7 = sp7+l


    target_count.append(sp1)
    target_count.append(sp2)
    target_count.append(sp3)
    target_count.append(sp4)
    target_count.append(sp5)
    target_count.append(sp6)
    target_count.append(sp7)

        #print(count)
    print(target_sp)
    print(target_count)

        #changing the Root directory back to working directory
    working_path = os.chdir(ROOT_DIR)
    return
##############################################################
########### WORKING WITH XML FILES FOR CLEANING PURPOSES######
##############################################################
def img_xml_dim_matching(img_path, annot_path):   
    count_correct = 0
    count_wrong = 0

    for fname in os.listdir(img_path):
        image_id = fname.split(".")[0]
        image_path = os.path.join(img_path, fname)
        XML_path = os.path.join(annot_path, image_id + '.xml')
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        w, h = image.size
        tree=ET.parse(XML_path)
        root=tree.getroot()
        x = int(tree.find('.//width').text)
        y = int(tree.find('.//height').text)
        if (x == w and y == h):
            count_correct +=1
            continue
        else:
            count_wrong +=1
            print(fname)
            #tree.find('.//width').text=str(w)
            #tree.find('.//height').text=str(h)
            #tree.write(XML_path)

    print('already corrected images', count_correct)
    print('corrected with code', count_wrong)
    working_path = os.chdir(ROOT_DIR)
    return


def xml_name_changing(target_path, folder_path_name, annot_path_name, correct_name):
    
    folder_path = os.path.join(target_path, folder_path_name) # E.g of folder_path_name:'Anoplolepis_gracilipes\XML Files'
    annot_folder_path = os.path.join(folder_path, annot_path_name) # E.g of annot_path_name:'22_04_29_Platform'
    #print(annot_folder_path)
    for xmlname in os.listdir(annot_folder_path):
        xmlfile = os.path.join(annot_folder_path, xmlname)
        print(xmlfile)
        tree=ET.parse(xmlfile)
        root=tree.getroot()
        name_tag_count = 0
        for elems in tree.findall('.//name'):
            name_tag_count +=1
            elems.text = correct_name # E.g of correct_name: "Anoplolepis_gracilipes"
            tree.write(xmlfile)
    print('total name tags updated = ', name_tag_count)
    working_path = os.chdir(ROOT_DIR)
    return

##############################################################
###### Finding the latest model in model logs directory ######
##############################################################
def newest(path):
    '''
    This function parse through the log directory to find the latest trained model and 
    returns path to the final .h5 file to load in the training weights.

    '''
    dir_name = next(os.walk(MODEL_DIR))[1]
    for j in dir_name:
        x = os.path.join(MODEL_DIR, j)
    l = os.path.join(MODEL_DIR,x)
    files = os.listdir(l)
    paths = [os.path.join(l, basename) for basename in files]
    return max(paths, key=os.path.getctime)

################################################################
######## Checking for valid image files
#################################################################
def valid_file_input(filename):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', 'tif')):
        return True
    else:
        return False

################################################################
######## loading all models for testing module
#################################################################

def load_all_models():
    ####### Loading weights for main model
    cfg= testing_config.PredictionConfig
    #weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_16_classes_0140.h5')
    #weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_Sampled_0150.h5')
    #weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_balanced_0140.h5')
    #weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_new_model_0150.h5')
    #weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_0170.h5')
    weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_NEW_0150.h5')
    print("Loading new weights, Please Wait..", weights_path)
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=cfg)
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
    #########

    ####### Loading weights for all sub model
    
    #(1: Anoplolepis) -- Waiting
    anp_cfg= testing_config.PredictionConfig_Anoplolepis
    #anp_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Anoplolepis_0150.h5')
    #anp_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Anoplolepis_gracilipes_0150.h5')
    anp_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Anoplolepis_gracilipes_0150.h5')
    print("Detecting with new weights, Please Wait..", anp_model_path)
    anp_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=anp_cfg)
    tf.keras.Model.load_weights(anp_model.keras_model, anp_model_path, by_name=True)
    
    
    #(2: Pheidole) -- updated Done
    ph_cfg= testing_config.PredictionConfig_pheidole
    #ph_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Pheidole_0150.h5')
    #ph_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Pheidole_megacephala_0140.h5')
    ph_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Pheidole_megacephala_0150.h5')
    print("Detecting with new weights, Please Wait..", ph_model_path)
    ph_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=ph_cfg)
    tf.keras.Model.load_weights(ph_model.keras_model, ph_model_path, by_name=True)

    #(3: Lepisiota) -- updated DONE
    Lep_cfg= testing_config.PredictionConfig_lepisiota
    #lep_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Lepisiota_0142.h5')
    #lep_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Lepisiota_frauenfeldi_0150.h5')
    lep_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Lepisiota_frauenfeldi_0150.h5')
    print("Detecting with new weights, Please Wait..", lep_model_path)
    lep_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=Lep_cfg)
    tf.keras.Model.load_weights(lep_model.keras_model, lep_model_path, by_name=True)

    #(4: Linepithema) -- updated DONE
    Lin_cfg= testing_config.PredictionConfig_linepithema
    #lin_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Linepithema_humile_0150.h5')
    lin_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Linepithema_humile_0150.h5')
    print("Detecting with new weights, Please Wait..", lin_model_path)
    lin_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=Lin_cfg)
    tf.keras.Model.load_weights(lin_model.keras_model, lin_model_path, by_name=True)

    #(5: Solenopsis) -- updated DONE
    sol_cfg= testing_config.PredictionConfig_Solenopsis
    #sol_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Solenopsis_0150.h5')
    #sol_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Solenopsis_Target_0150.h5')
    sol_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Solenopsis_Target_0150.h5')
    print("Detecting with new weights, Please Wait..", sol_model_path)
    sol_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=sol_cfg)
    tf.keras.Model.load_weights(sol_model.keras_model, sol_model_path, by_name=True)

    #(6: Wasmania) -- updated DONE
    was_cfg= testing_config.PredictionConfig_Wasmannia
    #was_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Wasmannia_0150.h5')
    #was_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_Wasmannia_auropunctata_0149.h5')
    was_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Wasmannia_auropunctata_0150.h5')
    print("Detecting with new weights, Please Wait..", was_model_path)
    was_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=was_cfg)
    tf.keras.Model.load_weights(was_model.keras_model, was_model_path, by_name=True)

    return(model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model)
######################################################################################
########## RUNNING SUB MODELS FOR ACTUAL CLASS IDENTIFICATION - single Image #########
######################################################################################


def test_single_image(dataset_test, cfg, image_id, model, main_class_names, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, cfg, 
                           image_id, use_mini_mask = True)
    print("the gt_class_id is :", gt_class_id)
    visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
                          dataset_test.class_names, figsize=(8, 8), title="First_level_Actual")
    #print("image_id ", image_id, dataset_test.image_reference(image_id))

    #print(gt_class_id)
    #run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    print(r['class_ids'])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_test.class_names, r['scores'], figsize=(8, 8), title="First_level_Predictions")
    idr = r['class_ids']
    print(idr)
    cln = []
    if len(idr) == 0:
        new_idr  = idr
        Actual_GT = gt_class_id
        print("Actual GT here is :", Actual_GT)
        cname = ''
        GT_name = main_class_names[int(Actual_GT[0])]
    else:
        for a in range(len(idr)):
            id_int = int(idr[a])
            cln.append(main_class_names[id_int])
        print('Now calling next level model for prediction')
        #Call sub_models for Actual Prediction in case of INVASIVE SPECIES
        new_idr, Actual_GT, cname, GT_name = call_sub_models_single_image(cln,image_id, anp_model, ph_model, lep_model, lin_model, sol_model, was_model)
        
        
    return(new_idr, Actual_GT, cname, GT_name)
#######################################
#######################################
def call_sub_models_single_image(cln_list, nimage_id, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    while True:
        if "Anoplolepis_gracilipes" in cln_list:
            print("Anoplolepis is present in the detected classes")

            anp_cfg= testing_config.PredictionConfig_Anoplolepis
            DT = Anoplolepis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Anoplolepis_gracilipes','Target_Ants','Non_Target_Ants']

            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, anp_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            #print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                          DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            new_results = anp_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual next Level Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            print('The AGT at the call sub model single time in Anop is:',AGT)
            #print('The NAME at the call sub model single time in Anop is:',gt_name)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
            
        if "Pheidole_megacephala" in cln_list:
            print("pheidole is present in the detected classes")
            
            ph_cfg= testing_config.PredictionConfig_pheidole
            DT = Pheidole_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Pheidole_megacephala','Target_Ants','Non_Target_Ants']

            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, ph_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                          DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            print(Agt_class_id)
            new_results = ph_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            print('The NAME at the call sub model single time in Pheidole is:',gt_name)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            print("in the else part of pheidole")
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
            print("the gt_name in else part is: ", gt_name)
            
        if "Lepisiota_frauenfeldi" in cln_list:
            #print("Lepisiota_frauenfeldi is present in detected classes")
            Lep_cfg= testing_config.PredictionConfig_lepisiota
            
            DT = Lepisiota_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Lepisiota_frauenfeldi','Target_Ants','Non_Target_Ants']

            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lep_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            #print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                         DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            new_results = lep_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            #print('The NAME at the call sub model single time in Lepisiota is:',gt_name)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
            
        if "Linepithema_humile" in cln_list:
            print("Linepithema_humile is present in detected classes")
            Lin_cfg= testing_config.PredictionConfig_linepithema
            
            DT = L_Humile_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG','Linepithema_humile','Target_Ants','Non_Target_Ants']
            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lin_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            #print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                          DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            new_results = lin_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            #print('The NAME at the call sub model single time in Linepithema is:',gt_name)
           # print('The AGT at the call sub model single time in Lin is:',AGT)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
            
        if "Solenopsis_Target" in cln_list:
            print("Solenopsis_Target is present in detected classes")
            sol_cfg= testing_config.PredictionConfig_Solenopsis
            
            DT = Solenopsis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Solenopsis_Target', 'Target_Ants','Non_Target_Ants']
          
            #run detection
            
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, sol_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            #print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                          DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            new_results = sol_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
           # print('The AGT at the call sub model single time in Sol is:',AGT)
            #print('The NAME at the call sub model single time in Solenpsis is:',gt_name)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
            
        if "Wasmannia_auropunctata" in cln_list:
            print("Wasmannia_auropunctata is present in detected classes")
            
            was_cfg= testing_config.PredictionConfig_Wasmannia
            DT = Wasmannia_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Wasmannia_auropunctata', 'Target_Ants','Non_Target_Ants']
            
            #run detection
            #new_results = new_model.detect([image], verbose=1)
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, was_cfg, nimage_id, use_mini_mask=True)
            #print("image_id ", nimage_id, DT.image_reference(nimage_id))
            #print('The actual class in loadImageGT', Agt_class_id)
            visualize.display_instances(image, gt_bbox, gt_mask, Agt_class_id, 
                          DT.class_names, figsize=(8, 8), title="Actual Ground Truth")
            
            new_results = was_model.detect([image], verbose=1)
            
            # Visualize results
            new_r = new_results[0]
            visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], class_names, new_r['scores'], 
                                        figsize=(8, 8), title="Actual Predictions")
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            #print('The NAME at the call sub model single time in Wasmania is:',gt_name)
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants' 
        #print("No Further evaluation needed")
        break
    
    print('The sp NAME at the call sub model single time is:',gt_name)
    return(idr, AGT, c_name, gt_name)

#######################################################################
############### RUNNING THE SUM MODELS ON WHOLE DATASET
#######################################################################

def test_data_image(dataset_test, cfg, image_id, model, main_class_names, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, cfg, 
                           image_id, use_mini_mask = True)
    #run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    idr = r['class_ids']
    cln = []
    if len(idr) == 0:
        new_idr  = idr
        Actual_GT = gt_class_id
        cname = ''
        GT_name = main_class_names[int(Actual_GT[0])]
    else:
        for a in range(len(idr)):
            id_int = int(idr[a])
            cln.append(main_class_names[id_int])
        #Call sub_models for Actual Prediction in case of INVASIVE SPECIES
        new_idr, Actual_GT, cname, GT_name = call_sub_models_data_image(cln,image_id, anp_model, ph_model, lep_model, lin_model, sol_model, was_model)
    
    return(new_idr, Actual_GT, cname, GT_name)


#######################################
#######################################
def call_sub_models_data_image(cln_list, nimage_id, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    
    while True:
        if "Anoplolepis_gracilipes" in cln_list:
            anp_cfg= testing_config.PredictionConfig_Anoplolepis
            DT = Anoplolepis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Anoplolepis_gracilipes','Target_Ants','Non_Target_Ants']
      
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, anp_cfg, nimage_id, use_mini_mask=True)
            new_results = anp_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]        
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants'
        
        if "Pheidole_megacephala" in cln_list:
            ph_cfg= testing_config.PredictionConfig_pheidole
            DT = Pheidole_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Pheidole_megacephala','Target_Ants','Non_Target_Ants']

            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, ph_cfg, nimage_id, use_mini_mask=True)
            new_results = ph_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants'
            
        if "Lepisiota_frauenfeldi" in cln_list:
            Lep_cfg= testing_config.PredictionConfig_lepisiota
            DT = Lepisiota_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Lepisiota_frauenfeldi','Target_Ants','Non_Target_Ants']

            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lep_cfg, nimage_id, use_mini_mask=True)
            new_results = lep_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants'
            
        if "Linepithema_humile" in cln_list:
            Lin_cfg= testing_config.PredictionConfig_linepithema
            DT = L_Humile_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG','Linepithema_humile', 'Target_Ants','Non_Target_Ants']
            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lin_cfg, nimage_id, use_mini_mask=True)
            new_results = lin_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants'
            
        if "Solenopsis_Target" in cln_list:   
            sol_cfg= testing_config.PredictionConfig_Solenopsis
            DT = Solenopsis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Solenopsis_Target','Target_Ants', 'Non_Target_Ants']
          
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, sol_cfg, nimage_id, use_mini_mask=True)
            new_results = sol_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_target_Ants'
            
        if "Wasmannia_auropunctata" in cln_list:
            
            was_cfg= testing_config.PredictionConfig_Wasmannia
            DT = Wasmannia_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Wasmannia_auropunctata','Target_Ants', 'Non_Target_Ants']
            
            #run detection         
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, was_cfg, nimage_id, use_mini_mask=True)
            new_results = was_model.detect([image], verbose=1)
            # Visualize results
            new_r = new_results[0]
            idr = new_r['class_ids']
            AGT = Agt_class_id
            gt_name = class_names[int(AGT[0])]
            if len(idr) == 0:
                cname = ''
            else:
                for a in range(len(idr)):
                    id_int = int(idr[a])
                    c_name = class_names[id_int]
                break
        else:
            idr = [15]
            AGT = [15]
            c_name = 'Non_Target_Ants'
            gt_name = 'Non_Target_Ants'
        print("No Further evaluation needed")
        break
    
    return(idr, AGT, c_name, gt_name)



#######################################################################
########## RUNNING SUB MODELS FOR ACTUAL CLASS IDENTIFICATION #########
#######################################################################

def call_sub_models(cln_list, nimage_id, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    while True:
        if "Anoplolepis_gracilipes" in cln_list:
            print("Anoplolepis is present in the detected classes")
            anp_cfg= testing_config.PredictionConfig_Anoplolepis
            DT = Anoplolepis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Anoplolepis_gracilipes','Target_Ants','Non_Target_Ants']

            #run detection

            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, anp_cfg, nimage_id, use_mini_mask=True)
            new_results = anp_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            print('The AGT at the call sub model Ann time is:',AGT)
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        #else:
         #   break
        if "Pheidole_megacephala" in cln_list:
            print("pheidole is present in the detected classes")
            
            ph_cfg= testing_config.PredictionConfig_pheidole
            DT = Pheidole_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Pheidole_megacephala','Target_Ants','Non_Target_Ants']

            #run detection

            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, ph_cfg, nimage_id, use_mini_mask=True)
            new_results = ph_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        #else:
            '''
            idr = [15]
            AGT = [15]
            '''
        if "Lepisiota_frauenfeldi" in cln_list:
            print("Lepisiota_frauenfeldi is present in detected classes")
            Lep_cfg= testing_config.PredictionConfig_lepisiota
            DT = Lepisiota_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()

            class_names = ['BG','Lepisiota_frauenfeldi','Target_Ants','Non_Target_Ants']

            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lep_cfg, nimage_id, use_mini_mask=True)
            new_results = lep_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        #else:
        '''
            idr = [15]
            AGT = [15]
        '''
        if "Linepithema_humile" in cln_list:
            print("Linepithema_humile is present in detected classes")
            Lin_cfg= testing_config.PredictionConfig_linepithema
            
            DT = L_Humile_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG','Linepithema_humile','Target_Ants', 'Non_Target_Ants']
            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, Lin_cfg, nimage_id, use_mini_mask=True)
            new_results = lin_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            print('The AGT at the call sub model Lin time is:',AGT)
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        '''
        else:
            idr = [15]
            AGT = [15]
        '''
        if "Solenopsis_Target" in cln_list:
            print("Solenopsis_Target is present in detected classes")
            sol_cfg= testing_config.PredictionConfig_Solenopsis
            
            DT = Solenopsis_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Solenopsis_Target','Target_Ants', 'Non_Target_Ants']
            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, sol_cfg, nimage_id, use_mini_mask=True)
            new_results = sol_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        '''
        else:
            idr = [15]
            AGT = [15]
        '''
        if "Wasmannia_auropunctata" in cln_list:
            print("Wasmannia_auropunctata is present in detected classes")
            
            was_cfg= testing_config.PredictionConfig_Wasmannia
            DT = Wasmannia_config.AntDetector()
            DT.load_dataset('./Test_Sample_old', is_train=False)
            DT.prepare()
            
            class_names = ['BG', 'Wasmannia_auropunctata', 'Target_Ants','Non_Target_Ants']
            
            #run detection
            image, image_meta, Agt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, was_cfg, nimage_id, use_mini_mask=True)
            new_results = was_model.detect([image], verbose=1)
            new_r = new_results[0]
            AGT_bbox = gt_bbox
            AGT_mask = gt_mask
            idr = new_r['class_ids']
            p_bbox = new_r['rois']
            p_score = new_r['scores']
            p_mask = new_r['masks']
            AGT = Agt_class_id
            #c_name = class_names[AGT[0]]
            c_name = class_names[AGT[0]] if (AGT[0] > 0) else 'BG'
            break
        '''
        else:
            idr = [15]
            AGT = [15]
        '''
        break
    return(idr, AGT, AGT_bbox, AGT_mask, p_bbox, p_score, p_mask, c_name)


def call_submodel_camera_image(cln, image, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    if "Anoplolepis_gracilipes" in cln:
        print("Anoplolepis_gracilipes is present in the detected classes")
        class_names = ['BG','Anoplolepis_gracilipes', 'Target_Ants','Non_Target_Ants']
        new_results = anp_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])

    if "Pheidole_megacephala" in cln:
        print("pheidole is present in the detected classes")
        class_names = ['BG','Pheidole_megacephala','Target_Ants', 'Non_Target_Ants']
        new_results = ph_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])
    if "Lepisiota_frauenfeldi" in cln:
        print("Lepisiota_frauenfeldi is present in the detected classes")
        class_names = ['BG','Lepisiota_frauenfeldi','Target_Ants','Non_Target_Ants']
        new_results = lep_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])
    if "Linepithema_humile" in cln:
        print("Linepithema_humile is present in detected classes")
        class_names = ['BG','Linepithema_humile', 'Target_Ants','Non_Target_Ants']
        new_results = lin_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])
    if "Solenopsis_Target" in cln:
        print("Solenopsis_Target is present in the detected classes")
        class_names = ['BG', 'Solenopsis_Target','Target_Ants', 'Non_Target_Ants']
        new_results = sol_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])
    if "Wasmannia_auropunctata" in cln:
        print("Wasmannia_auropunctata is present in the detected classes")
        class_names = ['BG', 'Wasmannia_auropunctata','Target_Ants', 'Non_Target_Ants']
        new_results = was_model.detect([image], verbose=1)
        new_r = new_results[0]
        idr = new_r['class_ids']
        visualize.display_instances(image, new_r['rois'], new_r['masks'], new_r['class_ids'], 
                        class_names, new_r['scores'])
        print(idr)
    
    return


################################################################
######## computing results for test datasets
#################################################################

def compute_predictions_for_confusion_matrix(DT, MCN, MD, cfg, anp_model, ph_model, lep_model, lin_model, sol_model, was_model):
    '''
    This function computes the confucin matrix called in the testing module.
    returns an array of actual and predicted vectors, a dataframe of results 
    with actual and predicted classes of the whole test dataset
    
    actual_class: list of actual classes of size n1
    predicted_class: list of predicted classes of size n2
    df_results: A dataframe of defined columns for results
    
    DT = Dataset Test
    MCN = Main Class Names 
    MD = Main model
    cfg = main test configuration
    anp_model = Ano[losis model
    ph_model = Pheidole model
    lep_model = Lepisiota model
    lin_model = Linepethima model
    sol_model = solenopsis model
    was_model = Wasmannia model
    
    '''
    actual_class = []
    predicted_class = []
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAP_ = []

    dfcols = ['img_id', 'actual_class_MM', 'predicted_class', 'Actual_Class','actual_GT', 'actualpredicted_class']
    df_results = pd.DataFrame(columns=dfcols)

    for image_id in DT.image_ids:    
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, cfg, image_id, use_mini_mask=True)
        info = DT.image_info[image_id]
        print("image_id ", image_id, DT.image_reference(image_id))
        # Run the model
        results = MD.detect([image], verbose=1)
        r = results[0]

        #######################################################
        #computing new prediction with sub models
        #######################################################
        #print(idr)
        #
        
        idr = r['class_ids']
        print(idr)
        cln = []
        for a in range(len(idr)):
            id_int = int(idr[a])
            cln.append(MCN[id_int])
            #print(cln)
            #Call sub_models for Actual Prediction and Ground Truth in case of INVASIVE SPECIES
        if(1 in idr or 4 in idr or 5 in idr or 8 in idr or 11 in idr or 14 in idr):
            APClass, GT, GT_bbox, GT_mask, pr_bbox, pr_score, pr_mask, cname = call_sub_models(cln, image_id, anp_model, ph_model,
                                                                                               lep_model, lin_model, sol_model, was_model)
            AClass = gt_class_id
            PClass = r['class_ids']
            #GT = GT
            GT = gt_class_id
            APClass = APClass
            cName = cname
            
        else:
            AClass = gt_class_id
            print("The AClass value is:", AClass[0])
            cName = MCN[int(AClass[0])] if (AClass[0] > 0) else 'BG'
            PClass = r['class_ids']
            GT = gt_class_id
            APClass = r['class_ids']
            GT_bbox = gt_bbox
            GT_mask = gt_mask
            pr_bbox = r['rois']
            pr_score = r['scores']
            pr_mask = r['masks']
            
        #######################################################
        
        #computing data for dataframe
        '''
        AClass = gt_class_id
        PClass = r['class_ids']
        GT = GT
        APClass = APClass
        '''
        df_results = df_results.append(pd.Series([image_id, AClass, PClass, cName, GT, APClass], index=dfcols), ignore_index=True)
        df_results = pd.DataFrame(df_results)
    
        #######################################################
    
        #compute gt_tot and pred_tot
        #gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        gt, pred = utils.gt_pred_lists(GT, GT_bbox, APClass, pr_bbox)
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)
        #precision_, recall_, AP_ 
        #AP_, precision_, recall_, overlap_ = utils.compute_ap(GT_bbox, GT, GT_mask,
       #                                   pr_bbox, APClass, pr_score, pr_mask)
        #mAP_.append(AP_)
        #check if the vectors len are equal
        #print("the actual len of the gt vect is : ", len(gt_tot))
        #print("the actual len of the pred vect is : ", len(pred_tot))
    

    gt_tot=gt_tot.astype(int)
    pred_tot=pred_tot.astype(int)
    #save the vectors of gt and pred
    save_dir = "output"
    gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
    df = pd.DataFrame(gt_pred_tot_json)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_json(os.path.join(save_dir,"gt_pred_test.json"))
    
    return(gt_tot, pred_tot, df_results, df)


#######################################################################
###########SAVING THE RESULTING IMAGES#################################
#######################################################################
def save_image(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
               scores_thresh=0.1, save_dir=None, mode=0):
    """
        image: image array
        image_name: image name
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        save_dir: (optional) the path to store image
        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
    mode_list = [0, 1, 2, 3]
    assert mode in mode_list, "mode's value should in mode_list %s" % str(mode_list)

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    useful_mask_indices = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        useful_mask_indices.append(i)

    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return

    colors = visualize.random_colors(len(useful_mask_indices))

    if mode != 3:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)

    if mode != 1:
        for index, value in enumerate(useful_mask_indices):
            masked_image = visualize.apply_mask(masked_image, masks[:, :, value], colors[index])

    masked_image = Image.fromarray(masked_image)

    if mode == 3:
        masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
        return

    draw = ImageDraw.Draw(masked_image)
    colors = np.array(colors).astype(int) * 255

    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        score = scores[value]
        label = class_names[class_id]

        y1, x1, y2, x2 = boxes[value]
        if mode != 2:
            color = tuple(colors[index])
            draw.rectangle((x1, y1, x2, y2), outline=color)

        # Label
        font = ImageFont.truetype('/Library/Fonts/arial.ttf', 15)
        draw.text((x1, y1), "%s %f" % (label, score), (255, 255, 255), font)

    masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
    return
#######################################################################