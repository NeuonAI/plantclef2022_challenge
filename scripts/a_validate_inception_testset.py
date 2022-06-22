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
import csv
from nets.inception_v4 import inception_v4
from nets import inception_utils
import os
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime



class validate_testset_module(object):
    def __init__(self,
                 image_dir_parent_test,
                 test_field_file,
                 checkpoint_model,
                 field_pkl_file,
                 prediction_file,
                 numclasses
                 ):

        self.image_dir_parent_test = image_dir_parent_test
        self.test_field_file = test_field_file
        self.checkpoint_model = checkpoint_model
        self.field_pkl_file  = field_pkl_file
        self.prediction_file = prediction_file
        self.numclasses = numclasses

        
        
        tf.reset_default_graph()
        

        # ----- Load field ----- #
        with open(self.test_field_file,'r') as fid2:
            f_lines = [x.strip() for x in fid2.readlines()]
            
        field_paths = [os.path.join(self.image_dir_parent_test,
                                        x.split(' ')[0]) for x in f_lines]
        field_labels = [int(x.split(' ')[5]) for x in f_lines]
                
         
            
        # ----- Read dictionary pkl file ----- #
        with open(self.field_pkl_file,'rb') as fid1:
        	field_dictionary = cPickle.load(fid1)
            
        
        # ----- Network hyperparameters ----- #
        global_batch = 6 
        batch = 60
        input_size = (299,299,3)
        topN =  5
        
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
        
        
        
        def match_field_dictionary(test_embedding_list, field_emb_list):
            similarity = cosine_similarity(test_embedding_list, field_emb_list)
                
            k_distribution = []
            # 1 - Cosine
            print("Get probability distribution")
            for sim in similarity:
                new_distribution = []
                for d in sim:
                    new_similarity = 1 - d
                    new_distribution.append(new_similarity)
                k_distribution.append(new_distribution)
                
            k_distribution = np.array(k_distribution)
                
                      
            softmax_list = []
            # Inverse weighting
            for d in k_distribution:
                inverse_weighting = (1/np.power(d,5))/np.sum(1/np.power(d,5))
                softmax_list.append(inverse_weighting)
            
            softmax_list = np.array(softmax_list)    
            
            return softmax_list
        
        
        # ----- Construct network ----- #
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in,
                                        num_classes=numclasses,
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
            
        
            
        variables_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(variables_to_restore)
        
        sample_im = ims * 1
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
        print("Start process")
        
        # ----- Write csv ----- #
        with open(self.prediction_file, 'a', newline='', encoding="utf-8") as csv1:
            writer = csv.writer(csv1, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['filepath', 'groundtruth', 'top1 species', 'top1 probability', "top5 species", "top5 probability", "top1 score", "top5 score"]
            writer.writerow(header)  
            
            
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                
                sess.run(tf.global_variables_initializer())
                restorer.restore(sess, self.checkpoint_model)
                
                test_batches = [field_paths[i:i + batch] for i in range(0, len(field_paths), batch)]
                ground_truth_species_batches = [field_labels[i:i + batch] for i in range(0, len(field_labels), batch)]
                
                # ------ Get field dictionary ----- #
                field_emb_list = []
  
                for field_class, field_emb in field_dictionary.items():
                    field_emb_list.append(np.squeeze(field_emb))
                
                field_emb_list = np.array(field_emb_list)
                
                top1_species_counter = 0
                top5_species_counter = 0
                
                filepath_list = []
                batch_counter = 0
                            
                            
                # ----- Iterate each observation id ----- #
                for current_batch_files, ground_truths_species in zip(test_batches, ground_truth_species_batches):
                    print("Batch counter:", batch_counter)                
    
                    
                    iter_run = len(current_batch_files)//global_batch
                    
                    print(f"[{datetimestr()}] Files:{len(current_batch_files)}")
            
                    if len(current_batch_files) > (iter_run * global_batch):            
                        iter_run += 1
                        padded = (iter_run * global_batch) - len(current_batch_files) 
                        current_batch_files = current_batch_files + ([current_batch_files[0]] * padded)
                        ground_truths_species = ground_truths_species + ([ground_truths_species[0]] * padded)
                    else:
                        padded = 0
                    
    
                    for n in range(iter_run):
                        # print(n)
                        paths = current_batch_files[n*global_batch:(n*global_batch)+global_batch]
                        gtruths_species = ground_truths_species[n*global_batch:(n*global_batch)+global_batch]
                        # print(paths)
                        ret = sess.run(sample_im,feed_dict = {
                                                tf_filepath2:paths})
                        
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
                            score1 = 0
                            score5 = 0
                            for i,a in enumerate(average_corner_crops[0:(global_batch-padded)]):
                                current_species = int(gtruths_species[i])
                                current_embedding = a.reshape(1,500) 
                                embedding_prediction = match_field_dictionary(current_embedding, field_emb_list)
                                embedding_squeezed = np.squeeze(embedding_prediction)
                                embedding_squeezed = np.float32(embedding_squeezed)                            
                                filepath_list.append(paths[i])
                                
                                topN_species = embedding_squeezed.argsort()[-topN:][::-1]
                                topN_species_probability = np.sort(embedding_squeezed)[-topN:][::-1]
            
                                if current_species in topN_species:
                                    top5_species_counter += 1
                                    score5 = 1
                                else:
                                    score5 = 0
                                if current_species == topN_species[0]:
                                    top1_species_counter += 1
                                    score1 = 1
                                else:
                                    score1 = 0
                                    
                                writer.writerow([paths[i], current_species, topN_species[0], topN_species_probability[0], topN_species, topN_species_probability, score1, score5])
                               
    
                        else: 
                            score1 = 0
                            score5 = 0                            
                            for i,a in enumerate(average_corner_crops):
                                current_species = int(gtruths_species[i])
                                current_embedding = a.reshape(1,500)
                                embedding_prediction = match_field_dictionary(current_embedding, field_emb_list)
                                embedding_squeezed = np.squeeze(embedding_prediction)
                                embedding_squeezed = np.float32(embedding_squeezed)                            
                                filepath_list.append(paths[i])
    
                                topN_species = embedding_squeezed.argsort()[-topN:][::-1]
                                topN_species_probability = np.sort(embedding_squeezed)[-topN:][::-1]
            
                                if current_species in topN_species:
                                    top5_species_counter += 1
                                    score5 = 1
                                else:
                                    score5 = 0
                                if current_species == topN_species[0]:
                                    top1_species_counter += 1
                                    score1 = 1
                                else:
                                    score1 = 0
                                    
                                writer.writerow([paths[i], current_species, topN_species[0], topN_species_probability[0], topN_species, topN_species_probability, score1, score5])                          
    
    
    
                    print("Top-1 species:", round(top1_species_counter / len(filepath_list),4), top1_species_counter,"/",len(filepath_list), " ----- Top-5 species:", round(top5_species_counter / len(filepath_list),4), top5_species_counter,"/",len(filepath_list))
                    
                    batch_counter += 1
            

               


        

       