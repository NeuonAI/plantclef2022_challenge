# -*- coding: utf-8 -*-
"""
NEUON AI PLANTCLEF 2022
"""

import os
import datetime
from a_construct_inception_dictionary import construct_dictionary_module as construct_dictionary
from a_validate_inception_testset import validate_testset_module as validate_testset


checkpoint_model = "PATH_TO_TRAINED_CHECKPOINT.ckpt"
latest_ckpt = checkpoint_model.split("/")[-1]
latest_ckpt_iter = latest_ckpt.split(".")[0]
latest_ckpt_model = latest_ckpt_iter + ".ckpt"


# ----- Construct dictionary ----- #
dictionary_file = "lists/plantclef2022_dictionary_trusted_train.txt"
image_dir = "PATH_TO_PlantClef2022_TRAINING_DATA"
pkl_file_dir = "PATH_TO_SAVE_DICTIONARY_DIR"
numclasses = 80000
dictionary_pkl_file = os.path.join(pkl_file_dir, "dictionary_80000_" + latest_ckpt_iter + ".pkl")


# ----- Validate test set ----- #
test_image_dir = "PATH_TO_PlantClef2022_TEST_IMAGES"
test_field_file = "lists/plantclef2022_trusted_validation.txt"
prediction_file = "mrr_score.txt"
prediction_dir = "PATH_TO_SAVE_PREDICTION_FILES_DIR"
prediction_csv = os.path.join(prediction_dir, "run4_predictions_" + latest_ckpt_iter + ".csv")


def dt():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')




        
print(f'[{dt()}] Constructing dictionary using', latest_ckpt_model)
construct_dictionary(
                dictionary_file,
                image_dir,
                checkpoint_model,
                dictionary_pkl_file,
                numclasses
                )

print(f'[{dt()}] Dictionary constructed...\n')


print(f'[{dt()}] Validating test set using', latest_ckpt_model)


validate_testset(
                test_image_dir,
                test_field_file,
                checkpoint_model,
                dictionary_pkl_file,
                prediction_file,
                prediction_csv,
                numclasses
                )

print(f'[{dt()}] Test set validated...\n')















    
