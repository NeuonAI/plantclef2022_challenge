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
import cv2
from nets.inception_v4 import inception_v4
from nets import inception_utils
from PIL import Image


class inception_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclasses,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file,
                 test_file,
                 input_size,
                 checkpoint_model,
                 learning_rate,
                 save_dir,
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
        
         
        print('Initiating tensors...')
        x = tf.placeholder(tf.float32,(self.batch,) + self.input_size)
        y1 = tf.placeholder(tf.int32,(self.batch,)) # species
        y2 = tf.placeholder(tf.int32,(self.batch,)) # family
        y3 = tf.placeholder(tf.int32,(self.batch,)) # genus
        y4 = tf.placeholder(tf.int32,(self.batch,)) # class
        y5 = tf.placeholder(tf.int32,(self.batch,)) # order
        y_onehot1 = tf.one_hot(y1,self.numclasses)
        y_onehot2 = tf.one_hot(y2,483)
        y_onehot3 = tf.one_hot(y3,9603)
        y_onehot4 = tf.one_hot(y4,8)
        y_onehot5 = tf.one_hot(y5,84)
        self.is_training = tf.placeholder(tf.bool)


        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        def data_in_train():
            return tf.map_fn(fn = train_preproc,elems = x,dtype=np.float32)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False)        
        
        def data_in_test():
            return tf.map_fn(fn = test_preproc,elems = x,dtype=np.float32)
        
        data_in = tf.cond(
                self.is_training,
                true_fn = data_in_train,
                false_fn = data_in_test
                )

        print('Constructing network...')        
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in,
                                            num_classes=self.numclasses,
                                            is_training=self.is_training)
            
            logits_family = slim.fully_connected(endpoints['PreLogitsFlatten'],483,activation_fn=None,
                                        scope='Family')
            logits_genus = slim.fully_connected(endpoints['PreLogitsFlatten'],9603,activation_fn=None,
                                        scope='Genus')
            logits_class = slim.fully_connected(endpoints['PreLogitsFlatten'],8,activation_fn=None,
                                        scope='Class')
            logits_order = slim.fully_connected(endpoints['PreLogitsFlatten'],84,activation_fn=None,
                                        scope='Order')            
            
        with tf.name_scope("cross_entropy"): 
            with tf.name_scope("auxloss"):
                self.auxloss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=endpoints['AuxLogits'], labels=y_onehot1))
            with tf.name_scope("logits_loss"):
                self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits, labels=y_onehot1))

            with tf.name_scope("logits_loss_family"):
                self.loss_family = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_family, labels=y_onehot2))

            with tf.name_scope("logits_loss_genus"):
                self.loss_genus = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_genus, labels=y_onehot3))

            with tf.name_scope("logits_loss_class"):
                self.loss_class = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_class, labels=y_onehot4))
                                
            with tf.name_scope("logits_loss_order"):
                self.loss_order = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_order, labels=y_onehot5))                                

            with tf.name_scope("L2_reg_loss"):
                self.regularization_loss = 0.00004 * tf.add_n(tf.losses.get_regularization_losses(scope='InceptionV4'))
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.loss + self.loss_family + self.loss_genus + self.loss_class + self.loss_order + self.auxloss + self.regularization_loss

            

        with tf.name_scope("accuracy"):
            with tf.name_scope('accuracy_species'):
                prediction = tf.argmax(logits,1)
                match = tf.equal(prediction,tf.argmax(y_onehot1,1))
                self.accuracy = tf.reduce_mean(tf.cast(match,tf.float32))  
            with tf.name_scope('accuracy_family'):
                prediction2 = tf.argmax(logits_family,1)
                match = tf.equal(prediction2,tf.argmax(y_onehot2,1))            
                self.accuracy_family = tf.reduce_mean(tf.cast(match,tf.float32)) 
            with tf.name_scope('accuracy_genus'):       
                prediction3 = tf.argmax(logits_genus,1)
                match = tf.equal(prediction3,tf.argmax(y_onehot3,1))            
                self.accuracy_genus = tf.reduce_mean(tf.cast(match,tf.float32))    
            with tf.name_scope('accuracy_class'):       
                prediction4 = tf.argmax(logits_class,1)
                match = tf.equal(prediction4,tf.argmax(y_onehot4,1))            
                self.accuracy_class = tf.reduce_mean(tf.cast(match,tf.float32))
            with tf.name_scope('accuracy_order'):       
                prediction5 = tf.argmax(logits_order,1)
                match = tf.equal(prediction5,tf.argmax(y_onehot5,1))            
                self.accuracy_order = tf.reduce_mean(tf.cast(match,tf.float32))                   
            

        self.var_list = [v for v in tf.trainable_variables()]


        # ----- Get all variables ----- #
        self.variables_to_restore = tf.trainable_variables()
        self.var_list_front = self.variables_to_restore[0:-16]
        self.var_list_end = self.variables_to_restore[-16:]
        
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        

        self.variables_to_restore = slim.get_variables_to_restore()
        restore_fn = slim.assign_from_checkpoint_fn(
            self.checkpoint_model, self.variables_to_restore[0:-16])    
        
        with tf.name_scope("train"):

            loss_accumulator = tf.Variable(0.0, trainable=False)
            acc_accumulator = tf.Variable(0.0, trainable=False)
            acc_accumulator_family = tf.Variable(0.0, trainable=False)
            acc_accumulator_genus = tf.Variable(0.0, trainable=False)
            acc_accumulator_class = tf.Variable(0.0, trainable=False)
            acc_accumulator_order = tf.Variable(0.0, trainable=False)
            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
            self.collect_acc = acc_accumulator.assign_add(self.accuracy)
            self.collect_acc_family = acc_accumulator_family.assign_add(self.accuracy_family)
            self.collect_acc_genus = acc_accumulator_genus.assign_add(self.accuracy_genus)
            self.collect_acc_class = acc_accumulator_class.assign_add(self.accuracy_class)
            self.collect_acc_order = acc_accumulator_order.assign_add(self.accuracy_order)

                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            self.average_acc = tf.cond(self.is_training,
                                       lambda: acc_accumulator / self.iterbatch,
                                       lambda: acc_accumulator / self.val_iter)
            self.average_acc_family = tf.cond(self.is_training,
                                       lambda: acc_accumulator_family / self.iterbatch,
                                       lambda: acc_accumulator_family / self.val_iter)
            self.average_acc_genus = tf.cond(self.is_training,
                                       lambda: acc_accumulator_genus / self.iterbatch,
                                       lambda: acc_accumulator_genus / self.val_iter)
            self.average_acc_class = tf.cond(self.is_training,
                                       lambda: acc_accumulator_class / self.iterbatch,
                                       lambda: acc_accumulator_class / self.val_iter)
            self.average_acc_order = tf.cond(self.is_training,
                                       lambda: acc_accumulator_order / self.iterbatch,
                                       lambda: acc_accumulator_order / self.val_iter)            

            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)
            self.zero_op_acc = tf.assign(acc_accumulator,0.0)
            self.zero_op_acc_family = tf.assign(acc_accumulator_family,0.0)
            self.zero_op_acc_genus = tf.assign(acc_accumulator_genus,0.0)
            self.zero_op_acc_class = tf.assign(acc_accumulator_class,0.0)
            self.zero_op_acc_order = tf.assign(acc_accumulator_order,0.0)
            
            
            # ----- Separate vars ----- #      
            self.accum_train_front = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_front] 
            self.accum_train_end = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_end]                                               
        
            self.zero_ops_train_front = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_front]
            self.zero_ops_train_end = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_end]

            
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)
                optimizer_end_layers = tf.train.AdamOptimizer(self.learning_rate)

                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_end_layers.compute_gradients(self.totalloss,self.var_list_end)
                
                gradient_only_front = [gc[0] for gc in gradient1]
                gradient_only_front,_ = tf.clip_by_global_norm(gradient_only_front,1.25)
                
                gradient_only_back = [gc[0] for gc in gradient2]
                gradient_only_back,_ = tf.clip_by_global_norm(gradient_only_back,1.25)
               
                self.accum_train_ops_front = [self.accum_train_front[i].assign_add(gc) for i,gc in enumerate(gradient_only_front)]
                self.accum_train_ops_end = [self.accum_train_end[i].assign_add(gc) for i,gc in enumerate(gradient_only_back)]
                
                
                
            # ----- Apply gradients ----- #
            self.train_step_front = optimizer.apply_gradients(
                    [(self.accum_train_front[i], gc[1]) for i, gc in enumerate(gradient1)])
      
            self.train_step_end = optimizer_end_layers.apply_gradients(
                    [(self.accum_train_end[i], gc[1]) for i, gc in enumerate(gradient2)])

            

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss) 
        tf.summary.scalar('accuracy',self.average_acc) 
        tf.summary.scalar('accuracy_f',self.average_acc_family)
        tf.summary.scalar('accuracy_g',self.average_acc_genus)
        tf.summary.scalar('accuracy_c',self.average_acc_class)
        tf.summary.scalar('accuracy_o',self.average_acc_order)

        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),
                                        tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        tensorboar_dir = 'tensorboard'
        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')


        print('Commencing training...') 
        val_best = 0.0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn(sess)
            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train_front)
                    sess.run(self.zero_ops_train_end)
                    sess.run([self.zero_op_acc_family,self.zero_op_acc_genus,self.zero_op_acc,self.zero_op_acc_class,self.zero_op_acc_order,self.zero_op_loss])
                    
                    # validations
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img,lbl1,lbl2,lbl3,lbl4,lbl5 = self.test_database.read_batch()
                            sess.run(
                                        [self.collect_loss,self.collect_acc,self.collect_acc_family,self.collect_acc_genus,self.collect_acc_class, self.collect_acc_order],
                                        feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                     y4 : lbl4,
                                                     y5 : lbl5,
                                                     self.is_training : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss,self.netAccuracy,self.netAccuracyFamily,self.netAccuracyGenus,self.netAccuracyClass,self.netAccuracyOrder = sess.run(
                                [self.merged,self.average_loss,self.average_acc,self.average_acc_family,self.average_acc_genus,self.average_acc_class,self.average_acc_order],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 
                        writer_test.add_summary(s, i)
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy:%f'%(self.test_database.epoch,i,self.netLoss,self.netAccuracy)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Family:%f'%(self.test_database.epoch,i,self.netLoss,self.netAccuracyFamily)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Genus:%f'%(self.test_database.epoch,i,self.netLoss,self.netAccuracyGenus)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Class:%f'%(self.test_database.epoch,i,self.netLoss,self.netAccuracyClass)) 
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy Order:%f'%(self.test_database.epoch,i,self.netLoss,self.netAccuracyOrder)) 


                        sess.run([self.zero_op_acc_family,self.zero_op_acc_genus,self.zero_op_acc_class,self.zero_op_acc_order,self.zero_op_acc,self.zero_op_loss])
                        
                        if self.netAccuracy > val_best:
                            val_best = self.netAccuracy
                            saver.save(sess, os.path.join(self.save_dir,'best.ckpt'))
                            print('Model saved')





                    # training
                    for j in range(self.iterbatch):
                        img,lbl1,lbl2,lbl3,lbl4,lbl5 = self.train_database.read_batch()
                        

    
                        sess.run(
                                    [self.collect_loss,self.collect_acc,self.collect_acc_family,self.collect_acc_genus,self.collect_acc_class,self.collect_acc_order,self.accum_train_ops_front,self.accum_train_ops_end],
                                    feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                     y4 : lbl4,
                                                     y5 : lbl5,
                                                 self.is_training : True
                                    }                                
                                )
                        
                    s,self.netLoss,self.netAccuracy,self.netAccuracyFamily,self.netAccuracyGenus,self.netAccuracyClass,self.netAccuracyOrder = sess.run(
                            [self.merged,self.average_loss,self.average_acc,self.average_acc_family,self.average_acc_genus,self.average_acc_class,self.average_acc_order],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    
                    sess.run([self.train_step_front])
                    sess.run([self.train_step_end])
                        

                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracy))
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Family:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracyFamily)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Genus:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracyGenus)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Class:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracyClass)) 
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy Order:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracyOrder)) 

                    
                    if i % 5000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')
        
class database_module(object):
    def __init__(
                self,
                image_source_dir,
                database_file,
                batch,
                input_size,
                numclasses,
                shuffle = False
            ):
        
        self.image_source_dir = image_source_dir
        self.database_file = database_file
        self.batch = batch
        self.input_size = input_size
        self.numclasses = numclasses
        self.shuffle = shuffle

        self.load_data_list()
        
    def load_data_list(self):
        with open(self.database_file,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels2 = [int(x.split(' ')[-3]) for x in lines] # family
        self.data_labels3 = [int(x.split(' ')[-2]) for x in lines] # genus
        self.data_labels1 = [int(x.split(' ')[-1]) for x in lines] # species
        self.data_labels4 = [int(x.split(' ')[-5]) for x in lines] # class
        self.data_labels5 = [int(x.split(' ')[-4]) for x in lines] # order
        self.data_num = len(self.data_paths)
        self.data_idx = np.arange(self.data_num)
        self.cursor = 0
        self.epoch = 0
        self.reset_data_list()
        
    def shuffle_data_list(self):
        np.random.shuffle(self.data_idx)
    
    def reset_data_list(self):

        if self.shuffle:
            print('shuffling')
            print(self.data_idx[0:10])            
            np.random.shuffle(self.data_idx)
            print(self.data_idx[0:10])
        self.cursor = 0
        
    def read_batch(self):
        img = []
        lbl1 = []
        lbl2 = []
        lbl3 = []
        lbl4 = []
        lbl5 = []
        while len(img) < self.batch:
            try:
         
                im = cv2.imread(self.data_paths[self.data_idx[self.cursor]])
                if im is None:
                   im = cv2.cvtColor(np.asarray(Image.open(self.data_paths[self.data_idx[self.cursor]]).convert('RGB')),cv2.COLOR_RGB2BGR)
                im = cv2.resize(im,(self.input_size[0:2]))
                if np.ndim(im) == 2:
                    img.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
                else:
                    img.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                lbl1.append(self.data_labels1[self.data_idx[self.cursor]])
                lbl2.append(self.data_labels2[self.data_idx[self.cursor]])
                lbl3.append(self.data_labels3[self.data_idx[self.cursor]])
                lbl4.append(self.data_labels4[self.data_idx[self.cursor]])
                lbl5.append(self.data_labels5[self.data_idx[self.cursor]])
            except:
                pass
            
            self.cursor += 1
            if self.cursor >= self.data_num:
                self.reset_data_list()
                self.epoch += 1
        
        img = np.asarray(img,dtype=np.float32)/255.0
        lbl1 = np.asarray(lbl1,dtype=np.int32)
        lbl2 = np.asarray(lbl2,dtype=np.int32)
        lbl3 = np.asarray(lbl3,dtype=np.int32)
        lbl4 = np.asarray(lbl4,dtype=np.int32)
        lbl5 = np.asarray(lbl5,dtype=np.int32)
        return (img,lbl1,lbl2,lbl3,lbl4,lbl5)   
    
            
        
    
    
    
    
    
    
    
    
    
    
    
    
