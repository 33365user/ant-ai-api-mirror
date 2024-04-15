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
from PIL import Image, ImageDraw, ImageFont
from xml.etree import ElementTree as ET

# Defining global variables

# Root directory of the project

ROOT_DIR = r'C:\Users\20200157'
ROOT_DIR = os.path.join(ROOT_DIR, "Ant_project_TF2")

ROOT_DIR_path = "F:\Fatima\Pheidole_megacephala_dataset"
#Target_path = "F:\Anotated_Ants"

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#print(ROOT_DIR)


#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.utils import compute_ap, compute_recall
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.utils import Dataset
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


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
    print('This is the root dir:', ROOT_DIR)
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
    print(TARGET_FOLDER)
    count_i = 0
    count_a = 0
    for root, dirs, files in os.walk((os.path.normpath(TARGET_FOLDER)), topdown=False):
            for name in files:
                if name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.png') or name.endswith('.PNG') or name.endswith('.tif') or name.endswith('.tiff') or name.endswith('.TIF') or name.endswith('.TIFF'):
                    count_i += 1
                    SourceFolder_i = os.path.join(root,name)
                    shutil.copy2(SourceFolder_i, img_path) #copies jpg and png to new folder
                    #image_con = name[:-4]                    
                if name.endswith('.xml'):
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
def removing_test_data_from_training():
    training_path = os.path.join(ROOT_DIR_path, 'dataset\images')
    for filename in os.listdir('./Test/images'):
        if os.path.isfile(os.path.join(training_path, filename)):
            os.remove(os.path.join(training_path, filename))
    return
def clean_dataset(d_path, i_path, a_path):
    #cleaning the dataset for unavailable XML or JPG files
    print(d_path, i_path, a_path)
    
    new_files = os.listdir(d_path)
    for i in new_files:
        for j in os.listdir(d_path+ '/'+ i):
            if (j.split(".")[0] + ".xml" not in os.listdir(a_path)) or (j.split(".")[0] + ".jpg" not in os.listdir(i_path) and j.split(".")[0] + ".png" not in os.listdir(i_path) and j.split(".")[0] + ".tif" not in os.listdir(i_path) and j.split(".")[0] + ".JPG" not in os.listdir(i_path) and j.split(".")[0] + ".PNG" not in os.listdir(i_path) and j.split(".")[0] + ".TIF" not in os.listdir(i_path) and j.split(".")[0] + ".TIFF" not in os.listdir(i_path) and j.split(".")[0] + ".tiff" not in os.listdir(i_path) and j.split(".")[0] + ".tif" not in os.listdir(i_path)):
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
            tree.find('.//width').text=str(w)
            tree.find('.//height').text=str(h)
            tree.write(XML_path)

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
######## computing results for test datasets
#################################################################

def compute_predictions_for_confusion_matrix(DT, MD, cfg):
    '''
    This function computes the confucin matrix called in the testing module.
    returns an array of actual and predicted vectors, a dataframe of results 
    with actual and predicted classes of the whole test dataset
    
    actual_class: list of actual classes of size n1
    predicted_class: list of predicted classes of size n2
    df_results: A dataframe of defined columns for results
    
    '''
    actual_class = []
    predicted_class = []
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAP_ = []

    dfcols = ['img_id', 'actual_class', 'predicted_class']
    df_results = pd.DataFrame(columns=dfcols)

    for image_id in DT.image_ids:    
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(DT, cfg, image_id, use_mini_mask=True)
        info = DT.image_info[image_id]
        print(image_id)
        # Run the model
        results = MD.detect([image], verbose=1)
        r = results[0]

        #computing data for dataframe
        AClass = gt_class_id
        PClass = r['class_ids']
        df_results = df_results.append(pd.Series([image_id, AClass, PClass], index=dfcols), ignore_index=True)
        df_results = pd.DataFrame(df_results)
    
    
    
        #compute gt_tot and pred_tot
        gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)
        #precision_, recall_, AP_ 
        AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
        mAP_.append(AP_)
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
    
    return(gt_tot, pred_tot, df_results)

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