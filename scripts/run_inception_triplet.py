# -*- coding: utf-8 -*-
"""
NEUON AI PLANTCLEF 2022
"""

from inception_module_triplet_loss import inception_module as incept_net


image_dir_parent_train = "PATH_TO_PlantClef2022_TRAINING_DATA"
image_dir_parent_test = "PATH_TO_PlantClef2022_TRAINING_DATA"

train_file = "lists/plantclef2022_trusted_train.txt"
test_file = "lists/plantclef2022_trusted_validation.txt"

checkpoint_model = "PATH_TO_PRETRAINED_MODEL_inception_v4_2016_09_09/inception_v4.ckpt"
checkpoint_save_dir = "PATH_TO_CHECKPOINT_SAVE_DIR"


batch = 32
input_size = (299,299,3)
numclasses = 80000
learning_rate = 0.0001
iterbatch = 4
max_iter = 500000
val_freq = 200
val_iter = 10


network = incept_net(
        batch = batch,
        iterbatch = iterbatch,
        numclasses = numclasses,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file = train_file,
        test_file = test_file,      
        checkpoint_model = checkpoint_model,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )