#import time
import json
import os
import sys
import warnings
import multiprocessing as mp
import subprocess as sp
warnings.filterwarnings("ignore")
import re
cwd = os.getcwd()
os.chdir(os.path.join(cwd, "Ant_Project_TF2"))
#from Ant_Project_TF2.aimodule import detect_species
os.chdir(cwd)
LINKER_VER = "1"
CLASSIFICATION_VER = "1"
#LIST_OF_SPECIES = ["NO AI"]
LIST_OF_SPECIES = ['BG',
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
                    'Non_Target_Ants',
                    'INVALID'
                  ]


def do_classification(identity, filepath, verbose=True):

    #establishing the variables here
    species_name=[""]
    regions_of_interest=[None]
    score=[0]
    code=-1
    identity=identity
    
    try:
        # always expect 4 params regardless
        data = sp.run([f"{cwd}\\interpreter39\\Scripts\\python.exe",
                       f"{cwd}\\Ant_Project_TF2\\aimodule.py", filepath],
                      capture_output=True, shell=verbose, text=True, universal_newlines=True,
                      cwd=os.path.join(cwd, "Ant_Project_TF2"))
        print(f"{identity} | Classification Complete!")
        patterns = re.findall("SUBPROC_OUT_V1_CONF_CODE_001=(.*)", str(data.stdout))

        if len(patterns) > 0:
            pass
        else:
            print("ERROR: Classification Failed due to technical error!")
            return dict(code=3, reason="Classification Failed due to technical server-side error", species_name=["INVALID"],
                        regions_of_interest=[None], score=[0], identity=identity)
        with open(patterns[0], 'r') as f:
            data = json.load(f)
            #code, species_name, regions_of_interest, score = detect_species(filepath)
            code = data['code']
            species_name = data['species_list']
            regions_of_interest = data['regionsofinterest']
            score = data['score_list']

        os.remove(patterns[0])
        if code == 0:
            reason = "Other Ant Species Detected"
        elif code == -1:
            reason = "Input File Type is Invalid"
        elif code == -2:
            reason = "Input File of Insufficient Quality for Detection (No Ants Detected)"
        else:
            reason = "Good"
            # nb: species_name = [], regions_of_interest = [ [x1, x2, y1, y2], [...], ...], score = []
        return dict(species_name=species_name, regions_of_interest=regions_of_interest, score=score, code=code, identity=identity, reason=reason)
        
    except BaseException as BE:
        ty = sys.exc_info()

        print(f"An exception occurred ({BE.__class__}): {BE}\n"
              f"\tType={ty[0]}, {ty[1]}")
        return dict(code=-1, reason=str(ty[0]), species_name=["INVALID"], regions_of_interest=[None], score=[0], identity=identity)

        #return false?

'''
# simulate classification
#time.sleep(15)
# as this isn't hooked up, return base stats.
return dict(species_name="NO AI", regions_of_interest=[(0,0, 100, 100), (100, 100, 150, 150)], score=0.0,
            code=-1, identity=identity)
'''


def add_parameters(identity, filepath):
    # perhaps some metainformation about the AI/linker?
    return dict(linker_version=LINKER_VER, ai_version=CLASSIFICATION_VER, classification_species=LIST_OF_SPECIES,
                message="Thanks for using the API!")


def test_method(*args, **kwargs):
    pass
