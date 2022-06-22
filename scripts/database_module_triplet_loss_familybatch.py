# -*- coding: utf-8 -*-
"""
NEUON AI PLANTCLEF 2022
"""


import os
import numpy as np
from PIL import Image
import cv2
import random
import copy


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
        
        print("Initialising...")
        self.image_source_dir = image_source_dir
        self.database_file = database_file
        self.batch = batch
        self.input_size = input_size
        self.numclasses = numclasses
        self.shuffle = shuffle
        self.epoch = 0
        
        self.load_data_list()
        
    def load_data_list(self):        
        self.database_dict = {}
        
        # ----- Dataset ----- #
        with open(self.database_file,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels = [int(x.split(' ')[-1]) for x in lines]
        self.fam_labels = [int(x.split(' ')[-3]) for x in lines]

        for key_species, key_family, path in zip(self.data_labels, self.fam_labels, self.data_paths):
            if key_family not in self.database_dict:
                self.database_dict[key_family] = {}
            if key_species not in self.database_dict[key_family]:
                self.database_dict[key_family][key_species] = []
                
            self.database_dict[key_family][key_species].append(path)
                
                       
        self.database_dict_copy = copy.deepcopy(self.database_dict)

        self.unique_labels = list(set(self.fam_labels))
        self.unique_labels_copy = copy.deepcopy(self.unique_labels)
   
          

    def reset_dict(self):
        self.unique_labels_copy = copy.deepcopy(self.unique_labels)
        self.database_dict_copy = copy.deepcopy(self.database_dict)
        
        
    def read_image(self, filepath):
        try:
            im = cv2.imread(filepath)
        
            if im is None:
               im = cv2.cvtColor(np.asarray(Image.open(filepath).convert('RGB')),cv2.COLOR_RGB2BGR)
            im = cv2.resize(im,(self.input_size[0:2]))
        
            if np.ndim(im) == 2:
                im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                
            else:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) 
        except:
            pass

        return im



    def get_random_class(self, labels):
        random_class = random.choice(labels)        
        return random_class    
    
    def get_path(self, current_class, dictionary, species_class):
        
        try:
            current_path = random.choice(dictionary[current_class].get(species_class))
        except:
            current_path = None
            
        return current_path
     

    def remove_anchor(self,current_class, current_path, dictionary, current_species):
    
        # Remove used anchor from dictionary
        for key, value in dictionary.items():
            if key == current_class:
                if current_path in value[current_species]:
                    value[current_species].remove(current_path)
                    
        return current_class, current_path, current_species                 

    def get_species_len(self, current_class, dictionary, current_species):
        key_len = len(dictionary[current_class][current_species])
        return key_len    

    def get_family_len(self, current_class, dictionary):
        key_len = len(dictionary[current_class])
        return key_len   

    def check_reset_dictionary(self, dictionary):
        reset_dictionary = False
        
        counter = 0
        for key, value in list(dictionary.items()):
            if len(value) == 0:
                del dictionary[key] 
        for key, value in list(dictionary.items()):                
            for species in value:
                species_len = len(dictionary[key][species])
                counter += species_len
                    
        
        if counter < self.batch:
            reset_dictionary = True
        
        return reset_dictionary, dictionary
    
        
        
    def read_batch(self):        
        self.total_filepaths = []
        self.img = []
        self.lbl = [] # species
        self.lbl2 = [] # family
        
        current_species_labels = []
        

        while len(self.total_filepaths) < self.batch:
                        
            #   Select random family class
            current_family = self.get_random_class(self.unique_labels_copy)

    
            #   Get family species list
            species_list = list(self.database_dict_copy[current_family])
                
            
            #   Iterate species
            for current_species in species_list:
                                
                species_files = self.database_dict_copy[current_family][current_species]
                species_files_len = len(species_files)
                            
                if species_files_len >= 4:
                    n_iter = 4
                if species_files_len == 3 or species_files_len == 2:
                    n_iter = 2            
                if species_files_len < 2:
                    n_iter = 1
                    
                for i in range(n_iter):

                    current_path = self.get_path(current_family, self.database_dict_copy, current_species)
                    im = self.read_image(current_path)

                    if (current_path is None) or (im is None):
                        current_family, current_path, current_species = self.remove_anchor(current_family, current_path, self.database_dict_copy, current_species)
                    if (current_path is not None) and (im is not None) and (len(self.total_filepaths) < self.batch):
                        current_family, current_path, current_species = self.remove_anchor(current_family, current_path, self.database_dict_copy, current_species)
                        self.img.append(im)
                        self.lbl.append(current_species)
                        self.total_filepaths.append(current_path)
                        self.lbl2.append(current_family)
                        
                        # Append current class labels
                        if current_species not in current_species_labels:
                            current_species_labels.append(current_species)
                
                            
                    #   Check species len
                    species_files_len = self.get_species_len(current_family, self.database_dict_copy, current_species)
                    if species_files_len < 1:
                        del self.database_dict_copy[current_family][current_species]
                            
                if len(self.total_filepaths) == self.batch:
                    break 


            #   Check family dictionary len
            family_len = self.get_family_len(current_family, self.database_dict_copy)
            if family_len < 1:
                self.unique_labels_copy.remove(current_family)
            
            
            #   Check if all current class species == species labels
            all_species_list = [s for k, v in self.database_dict_copy.items() for s in v]
            check = all(item in current_species_labels for item in all_species_list) 
            
            if check:
                self.reset_dict()
                self.epoch += 1
                print("Reset)")                
            
            #   Check if family labels is sufficient
            if (len(self.unique_labels_copy) < len(self.unique_labels) // 2):
                reset_dictionary, self.database_dict_copy = self.check_reset_dictionary(self.database_dict_copy)
                if reset_dictionary == True:
                    self.reset_dict()
                    self.epoch += 1
                    print("Reset")
            
            #   Check if batch consits of all same species
            if len(current_species_labels) == 1 and len(self.total_filepaths) == self.batch:
                self.total_filepaths = []
                self.img = []
                self.lbl = [] # species
                self.lbl2 = [] # family
                
                current_species_labels = []
                
   

        self.img = np.asarray(self.img,dtype=np.float32)/255.0

            
        return (self.img, self.lbl)
















        