import os
import sys
import testing_config
import tensorflow as tf
from tqdm.notebook import tqdm_notebook
import mrcnn.model as modellib
# ROOT_DIR = r'C:\Users\20200157'
ROOT_DIR = os.path.abspath("../")
ROOT_DIR = os.path.join(ROOT_DIR, "Ant_project_TF2")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
sys.path.append(ROOT_DIR)

################################################################
######## loading all models for testing module
#################################################################

def load_all_models():
    with tqdm_notebook(total = 100,
    desc = 'Loading configurations and models. Please wait .. ') as pbar:

        ####### Loading weights for main model
        #print("The system is now loading all models. Please Wait...")
        cfg= testing_config.PredictionConfig
        weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_ants_NEW_0150.h5')
        #print("Loading new weights, Please Wait..", weights_path)
        model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=cfg)
        tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
        pbar.update(15)
        #########

        ####### Loading weights for all sub model

        #(1: Anoplolepis) -- Waiting
        anp_cfg= testing_config.PredictionConfig_Anoplolepis
        anp_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Anoplolepis_gracilipes_0150.h5')
        #print("Detecting with new weights, Please Wait..", anp_model_path)
        anp_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=anp_cfg)
        tf.keras.Model.load_weights(anp_model.keras_model, anp_model_path, by_name=True)
        pbar.update(14)

        #(2: Pheidole) -- updated Done
        ph_cfg= testing_config.PredictionConfig_pheidole
        ph_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Pheidole_megacephala_0150.h5')
        #print("Detecting with new weights, Please Wait..", ph_model_path)
        ph_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=ph_cfg)
        tf.keras.Model.load_weights(ph_model.keras_model, ph_model_path, by_name=True)
        pbar.update(14)
        
        #(3: Lepisiota) -- updated DONE
        Lep_cfg= testing_config.PredictionConfig_lepisiota
        lep_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Lepisiota_frauenfeldi_0150.h5')
        #print("Detecting with new weights, Please Wait..", lep_model_path)
        lep_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=Lep_cfg)
        tf.keras.Model.load_weights(lep_model.keras_model, lep_model_path, by_name=True)
        pbar.update(14)
        
        #(4: Linepithema) -- updated DONE
        Lin_cfg= testing_config.PredictionConfig_linepithema
        lin_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Linepithema_humile_0150.h5')
        #print("Detecting with new weights, Please Wait..", lin_model_path)
        lin_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=Lin_cfg)
        tf.keras.Model.load_weights(lin_model.keras_model, lin_model_path, by_name=True)
        pbar.update(14)
        
        #(5: Solenopsis) -- updated DONE
        sol_cfg= testing_config.PredictionConfig_Solenopsis
        sol_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Solenopsis_Target_0150.h5')
        #print("Detecting with new weights, Please Wait..", sol_model_path)
        sol_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=sol_cfg)
        tf.keras.Model.load_weights(sol_model.keras_model, sol_model_path, by_name=True)
        pbar.update(14)
        
        #(6: Wasmania) -- updated DONE
        was_cfg= testing_config.PredictionConfig_Wasmannia
        was_model_path = os.path.join(MODEL_DIR, 'Models\mask_rcnn_ants_NEW_Wasmannia_auropunctata_0150.h5')
        #print("Detecting with new weights, Please Wait..", was_model_path)
        was_model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=was_cfg)
        tf.keras.Model.load_weights(was_model.keras_model, was_model_path, by_name=True)
        pbar.update(15)
        
    return(model, anp_model, ph_model, lep_model, lin_model, sol_model, was_model)
