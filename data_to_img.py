import numpy as np
import glob
import random
import scipy.misc as scm
import os


class Npy2Img(object):
    def __init__(self, file_dir):
        self.file_dir=file_dir

    def get_len(self):
        npy_dir=self.file_dir
        self.file_list=glob.glob('{}/*'.format(npy_dir))
        return len(self.file_list)

    def create_data_dir(self):
        train_data_dir="./trainA/"
        test_data_dir="./testA/"
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
        self.train_directory=train_data_dir
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)
        self.test_directory=test_data_dir

    def label_list_npy(self):
        #get the labels (last part of string)
        file_list=self.file_list
        label_list=[]
        for item in file_list:
            a=(item.split("/")[-1]).split(".")[0]
            label_list.append(a)
        self.label_list=label_list
        return self.label_list

    def reject_labels(self):
        self.labels_to_reject=['animal migration','asparagus','basket','bracelet','bridge','broccoli',
                'broom','clarinet','diving board','drums','elbow','feather','fence','floor lamp','foot',
                'fork','garden','garden hose','giraffe','goatee','golf club','grass','hedgehog','hockey puck',
                'hockey stick','ladder','keyboard','jail','hurricane','hot tub','mosquito','paintbrush',
                'mermaid','motorbike','mouth','moustache','lobster','lollipop','ocean','necklace','nail',
                'paper clip','peas','pencil','pineapple','popsicle','raccoon','rainbow','rake','rifle',
                'river','roller coaster','screwdriver','see saw','squiggle','spreadsheet','snowflake',
                'snorkel','stethoscope','stitches','streetlight','string bean','swing set','tennis racquet',
                'The Eiffel Tower','The Great Wall of China','tiger','tornado','traffic light',
                'trombone','wristwatch','yoga','zebra','zigzag']

    def get_label_from_path(self, path):
        a=(path.split("/")[-1]).split(".")[0]
        return a

    def setup(self):
        self.create_data_dir()
        self.label_list_npy()
        self.reject_labels()

    def load_npy(self, Ntrain, Ntest,P):
        #load N random out of get_len
        self.accept_list=list(set(self.label_list_npy())-set(self.labels_to_reject))
        print("len self.file_list=",len(self.file_list))
        print("len self.accept_list", len(self.accept_list))
        train_files_to_load=random.sample(self.accept_list,Ntrain)
        test_files_to_load=random.sample(self.accept_list,Ntest)
        print("train_files_to_load=", train_files_to_load)
        print("test_files_to_load=", test_files_to_load)
        #change files to load to get refactored classes
        idx=0
        for item in train_files_to_load:
            file_path=self.file_dir+str("/")+item+".npy"
            file_i=np.load(file_path)
            batch_i, dim_i=file_i.shape
            newd_i=int(dim_i**0.5)
            file_i=file_i.reshape(batch_i,newd_i,newd_i)
            #print("{}: file shape={}".format(item,file_i.shape))
            batch_to_chose=np.random.choice(batch_i,P)
            for i in batch_to_chose:
                #scm.imsave(self.train_directory+'output_{}-{}.png'.format(idx,i),np.invert(file_i[i,:,:]))
                scm.imsave(self.train_directory+'output_{}-{}.jpg'.format(idx,i),np.invert(file_i[i,:,:]))
            idx+=1
        idx=0
        for item in test_files_to_load:
            file_path=self.file_dir+str("/")+item+".npy"
            file_i=np.load(file_path)
            batch_i, dim_i=file_i.shape
            newd_i=int(dim_i**0.5)
            file_i=file_i.reshape(batch_i,newd_i,newd_i)
            batch_to_chose=np.random.choice(batch_i,P)
            for i in batch_to_chose:
                #scm.imsave(self.test_directory+'output_{}-{}.png'.format(idx,i),np.invert(file_i[i,:,:]))
                scm.imsave(self.test_directory+'output_{}-{}.jpg'.format(idx,i),np.invert(file_i[i,:,:]))
            idx+=1




root="/Users/aroushan/Documents/cs236/project/data_npy"
npy2img=Npy2Img(root)
l=npy2img.get_len()
print("num of files:",l)
npy2img.setup()
params={
        'Ntrain':100,
        'Ntest':10,
        'P':500
        }
npy2img.load_npy(**params)
