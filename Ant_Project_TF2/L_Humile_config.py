import os
from os import listdir
import shutil
import sys
import random
import math
import re
import time
import cv2
import itertools
import logging
import re
import time
import concurrent.futures

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines


from xml.etree import ElementTree
import skimage.draw
from skimage.util import img_as_float32
from distutils.version import LooseVersion
import cv2
import imgaug
from imgaug import augmenters as iaa

import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

import PIL
from PIL import Image


    
###############################################################
##################### Import Mask RCNN ########################
###############################################################
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


###############################################################
################Class configuration for Training ##############
###############################################################

class AntsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Ants"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 2  # background + 1 invasive-ant-types + 1 Non-Target-Ants
    NUM_CLASSES = 1 + 3  # background + 2 invasive-ant-types (incl. resembling ants) + Any other target ant + 1 Non-Target-Ants
      
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    
    IMAGE_RESIZE_MODE = 'square'
    #IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    #IMAGE_MIN_SCALE = 2.0
    
    

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512,1024) # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    #RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    LOSS_WEIGHTS={'mrcnn_mask_loss': 1.0,
                  'rpn_class_loss': 1.0, 'mrcnn_class_loss': 1.0,
                  'mrcnn_bbox_loss': 1.0, 'rpn_bbox_loss': 1.0}
    
    # ROIs kept after non-maximum supression (training and inference)
    ##POST_NMS_ROIS_TRAINING = 1000
   # POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    #RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    
    # Use epoch as per requirement
    STEPS_PER_EPOCH = 120
    LEARNING_RATE = 0.005
    WEIGHT_DECAY  = 0.0005
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    
config = AntsConfig()
#config.display()

###############################################################
################  DATASET PREPARATION MODULES #################
###############################################################

#creating our class for model training using bounding box annotation for multiple classes
class AntDetector(utils.Dataset):
        '''
        Class AntDetector to create a training and test dataset and loads it for training module.
        Dataset is created with the 16 classes defined as 1 background, 7 target and 8 non-target.
        All species that do not fall in the defined 7 classes will be assumed as non-target.
            annotations are .xml files
            images are .jpg or .png .tif files
           Extract boxes function will extract the annotated box from corresponding xml file and load mask function will 
       generate masks for the annotated ants in the images files and allocate corresponding class to it.
       
        '''
        #load_dataset function is used to load the train and test dataset
        def load_dataset(self, dataset, is_train=True):
            #we use add_class for each class in our dataset and assign numbers to them. 0 is background
            # self.add_class('source', 'class id', 'class name')
            self.add_class("dataset", 1, "Linepithema_humile")
            self.add_class("dataset", 2, "Target_Ants")
            self.add_class("dataset", 3, "Non_Target_Ants")
        
            
            # we concatenate the dataset with /images and /annots
            images_dir = dataset + '/images/'
            annotations_dir = dataset + '/annots/'
            
            #print(images_dir)
            #print(annotations_dir)
            #x=0
            # is_train will be true if we are training our model and false when we are testing the model
            for filename in listdir(images_dir):
                # extract image id
                image_id = filename.split(".")[0] # used to skip extension chars which is '.jpg or .tiff etc' (class_id.jpg)
                
                # if is_train is True to create training dataset
                # roughly 80% of dataset for training later using split function of keras
                
                               
                if is_train: 
                    #print("training continue")
                    #print("declaring image path and annotations path for training images")
                    img_path = images_dir + filename
                    ann_path = annotations_dir + image_id + '.xml'
                
                                    
                    # using add_image function we pass image_id, image_path and ann_path so that the current
                    # image is added to the dataset for training or testing
                
                    #print("adding images now")
                
                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3])

                    #continue
                # if is_train is not True to create testing testing
                if not is_train:
                    #print("testing continue")
                    #print("declaring image path and annotations path for test images")
                    img_path = images_dir + filename
                    ann_path = annotations_dir + image_id + '.xml'
                
                    #print(img_path)
                    #print(ann_path)
                
                    # using add_image function we pass image_id, image_path and ann_path so that the current
                    # image is added to the dataset for training or testing
                
                    #print("adding images now")
                
                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3])
                    
                    #continue
                    
        
        # this functions takes the image_id and returns the path of the image
        def image_reference(self, image_id):
            info = self.image_info[image_id]
            return info['path']
        
        
        # function used to extract bouding boxes from annotated files
        def extract_boxes(self, filename):

            # you can see how the images are annotated we extracrt the width, height and bndbox values

            # <annotation>
            # <size>
            #       <width>640</width>
            #       <height>360</height>
            #       <depth>3</depth>
            # </size>
            # <object>
            #          <name>damage</name>
            #          <pose>Unspecified</pose>
            #          <truncated>0</truncated>
            #          <difficult>0</difficult>
            #          <bndbox>
            #                 <xmin>315</xmin>
            #                 <ymin>160</ymin>
            #                 <xmax>381</xmax>
            #                 <ymax>199</ymax>
            #          </bndbox>
            # </object>
            # </annotation>

            # used to parse the .xml files
            tree = ElementTree.parse(filename)
        
            # to get the root of the xml file
            root = tree.getroot()
        
            # we will append all x, y coordinated in boxes
            # for all instances of an object
            boxes = list()
        
            # we find all attributes with name bndbox
            # bndbox will exist for each ground truth in an image
            for box in root.findall('.//object'):
                name = box.find('name').text
                xmin = int(box.find('./bndbox/xmin').text)
                ymin = int(box.find('./bndbox/ymin').text)
                xmax = int(box.find('./bndbox/xmax').text)
                ymax = int(box.find('./bndbox/ymax').text)
                coors = [xmin, ymin, xmax, ymax, name]
                boxes.append(coors)
            
                # I have included this line to skip any un-annotated images
                if name=='Level-1' or name=='Level-2' or name == 'Level-3':
                    boxes.append(coors)

                # extract width and height of the image
            width = int(root.find('.//size/width').text)
            height = int(root.find('.//size/height').text)
        
            # return boxes-> list, width-> int and height-> int 
            return boxes, width, height
    
        # this function calls on the extract_boxes method and is used to load a mask for each instance in an image
        # returns a boolean mask with following dimensions width * height * instances        
        def load_mask(self, image_id):

            # info points to the current image_id
            info = self.image_info[image_id]
        
            # we get the annotation path of image_id which is dataset_dir/annots/image_id.xml
            path = info['annotation']
        
            # we call the extract_boxes method(above) to get bndbox from .xml file
            boxes, w, h = self.extract_boxes(path)
        
            # we create len(boxes) number of masks of height 'h' and width 'w'
            masks = zeros([h, w, len(boxes)], dtype='uint8')

            class_ids = list()
        
            # we loop over all boxes and generate masks (bndbox mask) and class id for each instance
            # masks will have rectange shape as we have used bndboxes for annotations
            # for example: if 2.jpg have four objects we will have following masks and class_ids
            # 000000000 000000000 000003330 111100000
            # 000011100 022200000 000003330 111100000
            # 000011100 022200000 000003330 111100000
            # 000000000 022200000 000000000 000000000
            #    1         2          3         1<- class_ids
            for i in range(len(boxes)):
                box = boxes[i]
                row_s, row_e = box[1], box[3]
                col_s, col_e = box[0], box[2]
            
                # box[4] will have the name of the class for a particular ant-type
                
                if(box[4] == 'Linepithema_humile'):
                    masks[row_s:row_e, col_s:col_e, i] = 1
                    class_ids.append(self.class_names.index('Linepithema_humile'))
                elif(box[4] == 'Anoplolepis_gracilipes' or box[4] == 'Lepisiota_frauenfeldi' or box[4] == 'Solenopsis_geminata' or box[4] == 'Pheidole_megacephala' or box[4] =='Solenopsis_invicta' or box[4] =='Wasmannia_auropunctata'):
                    masks[row_s:row_e, col_s:col_e, i] = 2
                    class_ids.append(self.class_names.index('Target_Ants'))
                else:
                    masks[row_s:row_e, col_s:col_e, i] = 3
                    class_ids.append(self.class_names.index('Non_Target_Ants'))                    
            # return masks and class_ids as array
            return masks, asarray(class_ids, dtype='int32')
    
datasetconfig = AntDetector()