from __future__ import division # floating point division
from os import listdir
from os.path import isfile, join
import sklearn.datasets
import numpy as np
import nltk as nltk
import re
import sys
from copy import deepcopy
from random import randint

def return_filelist_dir(dirname):
     files = [f for f in listdir(postrainpath) if isfile(join(postrainpath, f))]
     return files

def loadfiles(filename):
    #print filename,type(filename)
    train=np.genfromtxt(filename,dtype=str,delimiter="/n")
    return train
if __name__ == "__main__":

 trainloc="/home/011/s/sx/sxd170431/bigram/A2_Data/aclImdb/train/"
 testloc="/home/011/s/sx/sxd170431/bigram/A2_Data/aclImdb/test/"
 postrainpath=trainloc+"pos/"
 negtrainpath=trainloc+"neg/"
 train_data=[]
 train_label=[]
 test_data=[]
 test_label=[]

 """Read the file lists in the pos and neg folders"""
 pos_trainfiles=return_filelist_dir(postrainpath)
 neg_trainfiles=return_filelist_dir(negtrainpath)
 pos_testfiles=return_filelist_dir(testloc+"pos/")
 neg_testfiles=return_filelist_dir(testloc+"neg/")

 """Load the contents of the files in an array"""
 for i in range(0,len(pos_trainfiles)):
     train_data.append(loadfiles(postrainpath+pos_trainfiles[i]))
     train_label.append(1)
 #train_data=np.asarray(train_data)
 #train_label=np.asarray(train_label)
 
  for i in range(0,len(neg_trainfiles)):
      train_data.append(loadfiles(negtrainpath+neg_trainfiles[i]))
      train_label.append(0)
  train_data=np.asarray(train_data)
  train_label=np.asarray(train_label)
  print train_data.shape,train_label.shape


