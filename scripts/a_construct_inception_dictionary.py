# -*- coding: utf-8 -*-
"""
NEUON AI PLANTCLEF 2022
"""


import sys
sys.path.append("PATH_TO_models/research/slim")
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
from nets.inception_v4 import inception_v4
from nets import inception_utils
from six.moves import cPickle
import os
import datetime



class construct_dictionary_module(object):
    def __init__(self,
                 train_dictionary_file,
                 dictionary_dir,
                 checkpoint_model,
                 pkl_file,
                 numclasses):
        
        
        self.train_dictionary_file = train_dictionary_file,
        self.dictionary_dir = dictionary_dir,
        self.checkpoint_model = checkpoint_model
        self.pkl_file = pkl_file
        self.numclasses = numclasses

        
        tf.reset_default_graph()

        data_dict = {}
        mean_emb_dict = {}

        # ----- Load data ----- #
        with open(train_dictionary_file,'r') as fid:
            h_lines = [x.strip() for x in fid.readlines()]
            
        data_paths = [os.path.join(dictionary_dir,
                                        x.split(' ')[0]) for x in h_lines]
        data_labels = [int(x.split(' ')[5]) for x in h_lines]
        
        for key, value in zip(data_labels, data_paths):
            if key not in data_dict:
                data_dict[key] = [] 
        
            data_dict[key].append(value)      
        
        

        
        # ----- Network hyperparameters ----- #
        global_batch = 6
        batch = 60
        input_size = (299,299,3)
        
        
        # ----- Initiate tensors ----- #
        is_training = tf.placeholder(tf.bool)
        is_train = tf.placeholder(tf.bool, name="is_training")      
        tf_filepath2 =  tf.placeholder(tf.string,shape=(global_batch,))
        
        
        def datetimestr():
            return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        

        def read_images(p):
            im = tf.io.read_file(p)    
            im =  tf.cast(tf.image.resize_images(tf.image.decode_png(
                im, channels=3, dtype=tf.uint8),(299,299)),tf.float32)
            
            im1 = im[0:260,0:260,:]
            im2 = im[0:260,-260:,:]
            im3 = im[-260:,0:260,:]
            im4 = im[-260:,-260:,:]
            im5 = im[19:279,19:279,:]
            
            im1 =  tf.cast(tf.image.resize_images(im1,(299,299)),tf.float32)
            im2 =  tf.cast(tf.image.resize_images(im2,(299,299)),tf.float32)
            im3 =  tf.cast(tf.image.resize_images(im3,(299,299)),tf.float32)
            im4 =  tf.cast(tf.image.resize_images(im4,(299,299)),tf.float32)
            im5 =  tf.cast(tf.image.resize_images(im5,(299,299)),tf.float32)
            
            im6 = tf.image.flip_left_right(im1)
            im7 = tf.image.flip_left_right(im2)
            im8 = tf.image.flip_left_right(im3)
            im9 = tf.image.flip_left_right(im4)
            im10 = tf.image.flip_left_right(im5)
            
            return tf.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9,im10])
        

        ims = tf.map_fn(fn=read_images,elems=tf_filepath2,dtype=np.float32)
        ims = tf.reshape(ims,(batch,)+input_size)/255.0
        
        
        # ----- Image preprocessing methods ----- #
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,input_size[0],input_size[1],is_training=True)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,input_size[0],input_size[1],is_training=False)  
        
        def data_in_train():
            return tf.map_fn(fn = train_preproc,elems = ims,dtype=np.float32)      
        
        def data_in_test():
            return tf.map_fn(fn = test_preproc,elems = ims,dtype=np.float32)
        
        
        data_in = tf.cond(
                is_training,
                true_fn = data_in_train,
                false_fn = data_in_test
                )

          
        
        
        # ----- Construct network ----- #            
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in,
                                        num_classes=self.numclasses,
                                        is_training=is_training
                                        )
                                    
            field_embs = endpoints['PreLogitsFlatten']
            
            field_bn = tf.layers.batch_normalization(field_embs, training=is_train)

            field_feat = tf.contrib.layers.fully_connected(
                            inputs=field_bn,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='field'
                    )   
            
            field_feat = tf.math.l2_normalize(
                                    field_feat,
                                    axis=1      
                                )   
            

            
        variables_to_restore  = slim.get_variables_to_restore()    
        restorer = tf.train.Saver(variables_to_restore)
        
        
        

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
        
        print(f"[{datetimestr()}] Start process")
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, checkpoint_model)
            
            counter = 0
            # ----- Iterate each class ----- #
            for key_class in data_dict.keys():
                

                embedding_list = []
                embedding_np_list = []
                class_mean_embedding = []
                
                counter += 1
                print(f"[{datetimestr()}] COUNTER: ", counter, " ----------- Current class: ", key_class)
                
                current_class_files = data_dict[key_class]     
                
                iter_run = len(current_class_files)//global_batch
                
                if len(current_class_files) > (iter_run * global_batch):            
                    iter_run += 1
                    padded = (iter_run * global_batch) - len(current_class_files) 
                    current_class_files = current_class_files + ([current_class_files[0]] * padded)
                else:
                    padded = 0
                    
                for n in range(iter_run):

                    paths = current_class_files[n*global_batch:(n*global_batch)+global_batch]            

        
                    sample_embedding = sess.run(
                                field_feat, 
                                feed_dict = {
                                            tf_filepath2:paths,   
                                            is_training : False,
                                            is_train : False
                                        }
                            )
                    
                        
                    sample_embedding = np.reshape(sample_embedding,(global_batch,10,-1))
                    average_corner_crops = np.mean(sample_embedding,axis=1)
                    if n == (iter_run - 1):                
                        for a in average_corner_crops[0:(global_batch-padded)]:
                            embedding_list.append(a)  
                    else:            
                        for a in average_corner_crops:
                            embedding_list.append(a)  
        
                print(f"[{datetimestr()}] Getting class mean embeddings...")
                embedding_np_list = np.array(embedding_list)
                class_mean_embedding = embedding_np_list.mean(axis=0)
        
                # ----- Save mean embs into dict ----- #
                print(f"[{datetimestr()}] Saving class mean embeddings to dictionary...")
                if key_class not in mean_emb_dict:
                    mean_emb_dict[key_class] = [] 
            
                mean_emb_dict[key_class].append(class_mean_embedding) 
                
        
        
               
        with open(pkl_file,'wb') as fid:
            cPickle.dump(mean_emb_dict,fid,protocol=cPickle.HIGHEST_PROTOCOL)
            print(f"[{datetimestr()}] Pkl file created")        
       
        
        
































