import Ants_config_module
import utils_functions

import Anoplolepis_utils_functions
import Anoplolepis_config

import pheidole_utils_functions
import Pheidole_config

import Lepisiota_config
import Lepisiota_utils_functions

import L_Humile_utils_functions
import L_Humile_config

import Wasmannia_utils_functions
import Wasmannia_config

import Solenopsis_utils_functions
import Solenopsis_config

##############################################################
########## CLASS NAMES DEFINITION ############################
##############################################################

main_class_names = ['BG',
                    'Anoplolepis_gracilipes',
                    'Bicoloured_ants_NI', 
                    'Large_black_ants_NI', 
                    'Lepisiota_frauenfeldi', 
                    'Linepithema_humile',
                    'Meat_ants_NI',
                    'Orange_ants_NI',
                    'Pheidole_megacephala', 
                    'Pony_ants_NI',
                    'Small_black_ants_NI',
                    'Solenopsis_Target',
                    'Spiny_ants_NI',
                    'Trap_jaw_ants_NI',
                    'Wasmannia_auropunctata', 
                    'Non_Target_Ants']
class_names = ['Anoplolepis_gracilipes',
                    'Bicoloured_ants_NI', 
                    'Large_black_ants_NI', 
                    'Lepisiota_frauenfeldi', 
                    'Linepithema_humile',
                    'Meat_ants_NI',
                    'Orange_ants_NI',
                    'Pheidole_megacephala', 
                    'Pony_ants_NI',
                    'Small_black_ants_NI',
                    'Solenopsis_Target',
                    'Spiny_ants_NI',
                    'Trap_jaw_ants_NI',
                    'Wasmannia_auropunctata', 
                    'Non_Target_Ants']

###############################################################
######## Class configuration for Testing 15 Classes ###########
###############################################################

class PredictionConfig(Ants_config_module.AntsConfig):
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 15
    DETECTION_MIN_CONFIDENCE = 0.40 #0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 28
    BATCH_SIZE = 1
    USE_MINI_MASK = False
    


###############################################################
#############Class configuration for Testing Anoplolepis#######
###############################################################

class PredictionConfig_Anoplolepis(Anoplolepis_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False

###############################################################
#############Class configuration for Testing Pheidole #########
###############################################################

class PredictionConfig_pheidole(Pheidole_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False
    


###############################################################
######Class configuration for Testing Lepisiota_frauenfeldi ###
###############################################################

class PredictionConfig_lepisiota(Lepisiota_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False
    

class PredictionConfig_linepithema(L_Humile_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False
    
    
class PredictionConfig_Solenopsis(Solenopsis_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False

class PredictionConfig_Wasmannia(Wasmannia_config.AntsConfig):    
    NAME = "Ant_Detection"
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [1024,1024,3]
    IMAGE_META_SIZE = 16
    BATCH_SIZE = 1
    USE_MINI_MASK = False
    
cfg= PredictionConfig