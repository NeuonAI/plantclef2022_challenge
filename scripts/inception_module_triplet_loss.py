# -*- coding: utf-8 -*-
"""
NEUON AI PLANTCLEF 2022
"""


import os 
import sys
sys.path.append("PATH_TO_models/research/slim")
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
from nets.inception_v4 import inception_v4
from nets import inception_utils
from database_module_triplet_loss_familybatch import database_module


class inception_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclasses,
                 input_size,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file,
                 test_file,
                 checkpoint_model,
                 save_dir,
                 learning_rate,                 
                 max_iter,
                 val_freq,
                 val_iter):
        
        self.batch = batch
        self.iterbatch = iterbatch
        self.image_dir_parent_train = image_dir_parent_train
        self.image_dir_parent_test = image_dir_parent_test
        self.train_file = train_file
        self.test_file = test_file
        self.input_size = input_size
        self.numclasses = numclasses
        self.checkpoint_model = checkpoint_model
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter
        
        
        
        print('Initiating database...')
        # ----- Database module ----- #
        self.train_database = database_module(
                image_source_dir = self.image_dir_parent_train,
                database_file = self.train_file,
                batch = self.batch,
                input_size = self.input_size,
                numclasses = self.numclasses,
                shuffle = True)

        self.test_database = database_module(
                image_source_dir = self.image_dir_parent_test,
                database_file = self.test_file,
                batch = self.batch,
                input_size = self.input_size,
                numclasses = self.numclasses,
                shuffle = True)
        
        
        
         
           
        
        # ----- Tensors ------ #
        print('Initiating tensors...')
        x = tf.placeholder(tf.float32,(None,) + self.input_size)
        y = tf.placeholder(tf.int32, (None,))        
        field_embs = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        is_train = tf.placeholder(tf.bool, name="is_training")
        
        
     
        # ----- Image pre-processing methods ----- #      
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False) 
        
        def data_in_train():
            return tf.map_fn(fn = train_preproc,elems = x, dtype=np.float32)
        
        def data_in_test():
            return tf.map_fn(fn = test_preproc,elems = x, dtype=np.float32)
        
        
        data_in = tf.cond(
                self.is_training,
                true_fn = data_in_train,
                false_fn = data_in_test
                )
        

              
        print('Constructing network...')               
        # ----- Network construction ----- #        
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in,
                                            num_classes=self.numclasses,
                                            is_training=self.is_training
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
           
    

        # ----- Get all variables ----- #
        self.variables_to_restore = tf.trainable_variables()
        self.var_list_front = self.variables_to_restore[0:-12]
        self.var_list_end = self.variables_to_restore[-12:]        
 
        self.var_list_train = self.var_list_front + self.var_list_end

        
        
        
        # ----- Network losses ----- #
        with tf.name_scope("loss_calculation"): 
            with tf.name_scope("triplets_loss"):
                self.triplets_loss = tf.reduce_mean(
                        tf.contrib.losses.metric_learning.triplet_semihard_loss(
                                labels=y, embeddings=field_feat, margin=1.0))

            with tf.name_scope("L2_reg_loss"):
                self.regularization_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.var_list_train]) * 0.00004 
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.triplets_loss + self.regularization_loss
                
                        
            
        # ----- Create update operation ----- #
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    
        self.vars_ckpt = slim.get_variables_to_restore()

        
        # ----- Restore model 1 ----- #
        restore_fn = slim.assign_from_checkpoint_fn(
            self.checkpoint_model, self.variables_to_restore[:-12])       
        
       
        # ----- Training scope ----- #       
        with tf.name_scope("train"):
            loss_accumulator = tf.Variable(0.0, trainable=False)
            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)


            
            # ----- Separate vars ----- #
            self.accum_train_front = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_front] 
            self.accum_train_end = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_end]                                               
        
            self.zero_ops_train_front = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_front]
            self.zero_ops_train_end = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_end]
            
            # ----- Set up optimizer / Compute gradients ----- #
            with tf.control_dependencies(self.update_ops):

                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)
                optimizer_end_layers = tf.train.AdamOptimizer(self.learning_rate)
                                
                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_end_layers.compute_gradients(self.totalloss,self.var_list_end)
              

                gradient_only_front = [gc[0] for gc in gradient1]
                gradient_only_front,_ = tf.clip_by_global_norm(gradient_only_front, 1.25)
                
                gradient_only_back = [gc[0] for gc in gradient2]
                gradient_only_back,_ = tf.clip_by_global_norm(gradient_only_back, 1.25)
                
               
                self.accum_train_ops_front = [self.accum_train_front[i].assign_add(gc) for i,gc in enumerate(gradient_only_front)]
            
                self.accum_train_ops_end = [self.accum_train_end[i].assign_add(gc) for i,gc in enumerate(gradient_only_back)]




            # ----- Apply gradients ----- #
            self.train_step_front = optimizer.apply_gradients(
                    [(self.accum_train_front[i], gc[1]) for i, gc in enumerate(gradient1)])
      
            self.train_step_end = optimizer_end_layers.apply_gradients(
                    [(self.accum_train_end[i], gc[1]) for i, gc in enumerate(gradient2)])
            
            

        # ----- Global variables ----- #
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]             
        var_list += bn_moving_vars
        
        
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss) 
        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        
        
        # ----- Tensorboard writer--- #
        tensorboar_dir = 'tensorboard'

        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')


        print('Commencing training...')        
        # ----- Create session 1 ----- #
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn(sess)     
            
            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train_front)
                    sess.run(self.zero_ops_train_end)
                    sess.run([self.zero_op_loss])                    
                    
                    
                    
                    # ----- Validation ----- #
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img,lbl = self.test_database.read_batch()
                            sess.run(
                                        self.collect_loss,
                                        feed_dict = {x : img,
                                                     y : lbl,
                                                     self.is_training : False,
                                                     is_train : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss = sess.run(                        
                                [self.merged,self.average_loss],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 
                        
                        writer_test.add_summary(s, i)
                        print('[Valid] Epoch:%i Iter:%i Loss:%f'%(self.test_database.epoch,i,self.netLoss))


                        sess.run([self.zero_op_loss])
                        


                    # ----- Train ----- #
                    for j in range(self.iterbatch):
                        img,lbl = self.train_database.read_batch()
    
                        sess.run(
                                    [self.collect_loss,self.accum_train_ops_front,self.accum_train_ops_end],
                                    feed_dict = {x : img, 
                                                 y : lbl,
                                                 self.is_training : True,
                                                 is_train : True
                                    }                                
                                )
                        
                    s,self.netLoss = sess.run(
                            [self.merged,self.average_loss],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    sess.run([self.train_step_front])
                    sess.run([self.train_step_end])
                        
                    print('[Train] Epoch:%i Iter:%i Loss:%f'%(self.train_database.epoch,i,self.netLoss))


                    
                    if i % 10000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
                
            # ----- Save model --- #
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')




           

    
     

    
    
    
    
    
    
    
    
    
