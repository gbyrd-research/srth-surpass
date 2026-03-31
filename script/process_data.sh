#!/bin/bash
DATASETS=("tissue_2")  #tissue_1
PATH_TO_DATASET="/cis/home/sschmi46/chole_ws/data/Jesse"

# Encoding instructions
python encode_instruction.py --dataset_dir $PATH_TO_DATASET --encoder distilbert --from_count

