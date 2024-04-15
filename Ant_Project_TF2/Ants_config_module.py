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


'''###############################################################
##################### Claculating Class_Weights################
###############################################################
import numpy as np

CLASS_WEIGHT = {0:10000, 1:4255, 2:1644, 3:2844, 4:5952, 5:41725, 6:5121, 7:352, 8:24563, 9:468, 10:3421, 11:32718, 12:1482, 13:88, 14:9514, 15:7220}
def compute_class_weights(CLASS_WEIGHT=CLASS_WEIGHT):
    mean = np.array(list(CLASS_WEIGHT.values())).mean() # sum_class_occurence / nb_classes
    max_weight = np.array(list(CLASS_WEIGHT.values())).max()
    CLASS_WEIGHT.update((x, float(max_weight/(y))) for x, y in CLASS_WEIGHT.items())
    CLASS_WEIGHT=dict(sorted(CLASS_WEIGHT.items()))
    return CLASS_WEIGHT
compute_class_weights(CLASS_WEIGHT)
'''
    
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
    NUM_CLASSES = 1 + 15  # background + 15 invasive-ant-types (incl. resembling ants) + 1 Non-Target-Ants
    
      
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
    #IMAGE_META_SIZE = 28
    

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024) # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    #RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 60
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
    #CLASS_WEIGHT = CLASS_WEIGHT
    # Use epoch as per requirement
    STEPS_PER_EPOCH = 100
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
            self.add_class("dataset", 1, "Anoplolepis_gracilipes")
            self.add_class("dataset", 2, "Bicoloured_ants_NI")
            self.add_class("dataset", 3, "Large_black_ants_NI")
            self.add_class("dataset", 4, "Lepisiota_frauenfeldi")
            self.add_class("dataset", 5, "Linepithema_humile")
            self.add_class("dataset", 6, "Meat_ants_NI")
            self.add_class("dataset", 7, "Orange_ants_NI")
            self.add_class("dataset", 8, "Pheidole_megacephala")
            self.add_class("dataset", 9, "Pony_ants_NI")
            self.add_class("dataset", 10, "Small_black_ants_NI")
            self.add_class("dataset", 11, "Solenopsis_Target")
            self.add_class("dataset", 12, "Spiny_ants_NI")
            self.add_class("dataset", 13, "Trap_jaw_ants_NI")
            self.add_class("dataset", 14, "Wasmannia_auropunctata")
            self.add_class("dataset", 15, "Non_Target_Ants")
        
            
            # we concatenate the dataset with /images and /annots
            images_dir = dataset + '/images/'
            annotations_dir = dataset + '/annots/'
            
            #print(images_dir)
            #print(annotations_dir)
            #x=0
            # is_train will be true if we are training our model and false when we are testing the model
            for filename in listdir(images_dir):
                # extract image id
                image_id = filename.split(".")[0] # used to skip extension chars which is '.jpg' or ',tiff' etc (class_id.jpg)
                
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
                
                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

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
                
                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
                    
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
                if name=='Level-1' or name=='Level-2' or name=='Level-3' or name=='Level-4' or name=='Level-5' or name=='Level-6' or name=='Level-7' or name=='level-8' or name=='level-9' or name=='level-10' or name=='level-11' or name=='level-12' or name=='level-13' or name=='level-14' or name=='level-15':
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
                if (box[4] == 'Anoplolepis_gracilipes' or box[4] == 'Plagiolepis_alluaudi' or box[4] =='Oecophylla_smaragdina'):
                    masks[row_s:row_e, col_s:col_e, i] = 1
                    class_ids.append(self.class_names.index('Anoplolepis_gracilipes'))
                elif(box[4] == 'Monomorium_floricola' or box[4] == 'Opisthopsis_picta' or box[4] == 'Opisthopsis_haddoni' or box[4] == 'Camponotus_sp_nov_gp' or box[4] =='Leptomyrmex_rufipes' or box[4] == 'Podomyrma_adelaidae' or box[4] == 'Chelaner_kiliani'):
                    masks[row_s:row_e, col_s:col_e, i] = 2
                    class_ids.append(self.class_names.index('Bicoloured_ants_NI'))
                elif(box[4] == 'Camponotus_aeneopilosus' or box[4] == 'Camponotus_sp_9' or box[4] == 'Camponotus_sp_A' or box[4] == 'Diacamma_leve' or box[4] == 'Bothroponera_denticulata' or box[4] == 'Iridomyrmex mayri'):
                    masks[row_s:row_e, col_s:col_e, i] = 3
                    class_ids.append(self.class_names.index('Large_black_ants_NI'))
                elif(box[4] == 'Lepisiota_frauenfeldi' or box[4] == 'Paratrechina_longicornis'):
                    masks[row_s:row_e, col_s:col_e, i] = 4
                    class_ids.append(self.class_names.index('Lepisiota_frauenfeldi'))                
                elif(box[4] == 'Linepithema_humile' or box[4] == 'Iridomyrmex_suchieri' or box[4] == 'Iridomyrmex_pallidus'):
                    masks[row_s:row_e, col_s:col_e, i] = 5
                    class_ids.append(self.class_names.index('Linepithema_humile'))
                elif(box[4] == 'Iridomyrmex_sanguineus' or box[4] == 'Iridomyrmex_reburrus' or box[4] == 'Iridomyrmex_purpureus' or box[4] == 'Iridomyrmex_purpureus_gp'):
                    masks[row_s:row_e, col_s:col_e, i] = 6
                    class_ids.append(self.class_names.index('Meat_ants_NI'))
                elif(box[4] == 'Aphaenogaster_longiceps' or box[4] == 'Papyrius_nitidus' or box[4] == 'Ooceraea_australis' or box[4] == 'Acropyga_acutiventris' or box[4] == 'Cardiocondyla_nuda' or box[4] == 'Cardiocondyla_nuda_atalanta'):
                    masks[row_s:row_e, col_s:col_e, i] = 7
                    class_ids.append(self.class_names.index('Orange_ants_NI'))              
                elif(box[4] == 'Pheidole_megacephala' or box[4] == 'Pheidole_spp' or box[4] == 'Pheidole_sp' or box[4] == 'Pheidole_QM1' or box[4] == 'Pheidole_QM2' or box[4] == 'Pheidole_QM3' or box[4] == 'Pheidole_QM8' or box[4] == 'Pheidole_QM10' or box[4] == 'Carebara_affinis' or box[4] == 'Pheidole_murdoch1' or box[4] == 'Pheidole_murdoch2'):
                    masks[row_s:row_e, col_s:col_e, i] = 8
                    class_ids.append(self.class_names.index('Pheidole_megacephala'))
                elif(box[4] == 'Rhytidoponera_victoriae' or box[4] == 'Rhytidoponera_metallica'):
                    masks[row_s:row_e, col_s:col_e, i] = 9
                    class_ids.append(self.class_names.index('Pony_ants_NI'))
                elif(box[4] == 'Anonychomyrma_sp_A' or box[4] == 'Anonychomyrma_sp' or box[4] == 'Ochetellus_glaber' or box[4] == 'Technomyrmex_jocosus' or box[4] == 'Nylanderia_QM1' or box[4] == 'Nylanderia_QM3' or box[4] == 'Iridomyrmex_sp2_mattiroloi_gp' or box[4] == 'Iridomyrmex_chasei' or box[4] == 'Iridomyrmex_anceps' or box[4] == 'Iridomyrmex_cyaneus' or box[4] == 'Iridomyrmex_spp'):
                    masks[row_s:row_e, col_s:col_e, i] = 10
                    class_ids.append(self.class_names.index('Small_black_ants_NI'))
                elif(box[4] == 'Solenopsis_geminata' or box[4] == 'Solenopsis_invicta' or box[4] == 'Trichomyrmex_mayri' or box[4] == 'Tetramorium_bicarinatum' or box[4] == 'Trichomyrmex_destructor' or box[4] == 'Monomorium_rothsteini'):
                    masks[row_s:row_e, col_s:col_e, i] = 11
                    class_ids.append(self.class_names.index('Solenopsis_Target'))
                elif(box[4] == 'Crematogaster_QM1' or box[4] == 'Crematogaster_QM3' or box[4] =='Dolichoderus_scrobiculatus' or box[4] == 'Dolichoderus_scabridus' or box[4] == 'Dolichoderus_extensispinus' or box[4] == 'Polyrhachis_ammon' or box[4] == 'Polyrhachis_brisbanensis' or box[4] == 'Polyrhachis_rufifemur' or box[4] == 'Polyrhachis_spp' or box[4] == 'Crematogaster_spp'):
                    masks[row_s:row_e, col_s:col_e, i] = 12
                    class_ids.append(self.class_names.index('Spiny_ants_NI'))
                elif(box[4] == 'Amblyopone_australis' or box[4] == 'Odontomachus_sp_nr_turneri' or box[4] == 'Odontomachus_turneri'):
                    masks[row_s:row_e, col_s:col_e, i] = 13
                    class_ids.append(self.class_names.index('Trap_jaw_ants_NI'))
                elif(box[4] == 'Wasmannia_auropunctata' or box[4] == 'Tetramorium_simillimum' or box[4] == 'Tapinoma_melanocephalum' or box[4] == 'Tapinoma_QMAI1' or box[4] == 'Tapinoma_sp_1'):
                    masks[row_s:row_e, col_s:col_e, i] = 14
                    class_ids.append(self.class_names.index('Wasmannia_auropunctata'))              
                else:
                    masks[row_s:row_e, col_s:col_e, i] = 15
                    class_ids.append(self.class_names.index('Non_Target_Ants'))                
            # return masks and class_ids as array
            return masks, asarray(class_ids, dtype='int32')
    
datasetconfig = AntDetector()