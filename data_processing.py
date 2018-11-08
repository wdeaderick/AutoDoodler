import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import glob
print("imports done!")


class Dataset(data.Dataset):
    def __init__(self, file_list, labels):
        self.file_list=file_list
        self.labels=labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        X= np.load(self.file_list[index])
        Y=str(self.file_list[index]).strip().split(".")[0] #Y in string

        return X,Y

def get_labels(files):
    #print(str(files).split("/")[-1].strip())
    file_label=(str(files).split("/")[-1]).split(".")[0] #Y in string
    return str(files).split("/")[-1],str(file_label)

use_cuda=torch.cuda.is_available()
device=torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark=True
params={'batch_size':64,
        'shuffle':True,
        'num_workers':6
        }
max_epochs=100

directory="/Users/aroushan/Documents/cs236/project/data_npy"#use the directory where saved the data
directory_files=glob.glob('{}/*'.format(directory))
train_files, train_labels=list(), list()
for files in directory_files:
    x,y=get_labels(files)
#    print(x,y)
    train_files+=x
    train_labels+=y
print(train_files)
#print(train_labels)
training_set=Dataset(train_files, train_labels)
train_loader=DataLoader(training_set, batch_size=params['batch_size'], shuffle=params['shuffle'], num_workers=params['num_workers'])


#optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2_reg)

