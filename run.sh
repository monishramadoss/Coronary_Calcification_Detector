#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis

python /home/mramados/Coronary_Calcification_Detector/struct_seg.py > /home/mramados/Coronary_Calcification_Detector/struct_stdout 2>&1
python /home/mramados/Coronary_Calcification_Detector/heart_mask_gen.py > /home/mramados/Coronary_Calcification_Detector/mask_stdout 2>&1
python /home/mramados/Coronary_Calcification_Detector/plaque_seg.py > /home/mramados/Coronary_Calcification_Detector/plaque_stdout 2>&1