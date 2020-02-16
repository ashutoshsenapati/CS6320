from __future__ import division # floating point division
import numpy as np
import re
import sys
from copy import deepcopy
from random import randint

"""
Created on Sat Feb 15 01:30:45 2020

@author: Srijita
"""
trainpath="D:/Grad Studies/NLP/A1/Data/"
testpath="D:/Grad Studies/NLP/A1/Data/"
vocab={}
vocab_temp={}
bigram={}
bigram_prob={}

def readfile(filename):
    data=[]
    file1 = open(filename,"r") 
    for line in file1.readlines():
        data.append(line)
    return data
    
#The main python module starts here
if __name__== "__main__":
   init_token=['<s>','</s>'] 
   train=readfile(trainpath+"train.txt")
   test= readfile(trainpath+"test.txt")
   test_to_write=deepcopy(test)
   #print "The length of train is", len(train)
   #print "The length of test is", len(test)
   #If mode is 0, then no smoothening, else smoothening
   mode=int(sys.argv[1])
   print "The mode is", mode
   for i in range(0,len(train)):
       #remove all unnecessary punctuations
       train[i]=train[i].replace(".", "")
       train[i]=train[i].replace("(", " ")
       train[i]=train[i].replace(")", " ")
       train[i]=train[i].replace("[", " ")
       train[i]=train[i].replace("]", " ")
       train[i]=train[i].replace("'", "")
       train[i]=train[i].replace('"', "")
       train[i]=train[i].replace("-","")
       train[i]=train[i].replace(",","")
       train[i]=train[i].replace(";","")
       #strip spaces from beg and end
       train[i]=train[i].strip()
       #delimit every line by a beginning and ending token
       train[i]='<s> '+train[i]
       train[i]=train[i]+" </s>"
       vocab_train=train[i].split()
       for j in vocab_train:
           if j not in vocab_temp.keys():
              vocab_temp[j]=0
   vocab_size=len(vocab_temp)    
   print "The train vocab size is", vocab_size
   #Finding the probabilities of the test set   
   bigram_test_count=[]
   bigram_test_prob=[]
   prob_test=[]
   for i in range(0,len(test)):
       #remove all unnecessary punctuations
       test[i]=test[i].replace(".", "")
       test[i]=test[i].replace("(", " ")
       test[i]=test[i].replace(")", " ")
       test[i]=test[i].replace("[", " ")
       test[i]=test[i].replace("]", " ")
       test[i]=test[i].replace("'", "")
       test[i]=test[i].replace('"', "")
       test[i]=test[i].replace("-","")
       test[i]=test[i].replace(",","")
       test[i]=test[i].replace(";","")
       #strip spaces from beg and end
       test[i]=test[i].strip()
       #delimit every line by a beginning and ending token
       test[i]='<s> '+test[i]
       test[i]=test[i]+" </s>"
#       vocab_list=test[i].split()
#       for j in vocab_list:
#           if j not in vocab.keys():
#              vocab[j]=0
#              
   #populate the keys of the bigram model           
   for key in vocab_temp.keys():
       bigram[key]={}
       for key1 in vocab_temp.keys():
           bigram[key][key1]=[0,0]
   
   #populate the count of the unigrams in vocab
   for key in vocab_temp.keys():
        if mode==0:
           key_count=0
        else:
           key_count=vocab_size 
        for j in range(0,len(train)):
            if (" "+key+" " in train[j]) and (key not in init_token):
               key_count=key_count+train[j].count(" "+key+" ")
            else:
               if (key in train[j]) and (key in init_token):
                  key_count=key_count+1 
        vocab[key]=key_count    
   
   #print "Vocab is", vocab 
   #populate the bigram counts and probabilties     
   for key in bigram.keys():
       for key1 in bigram[key].keys():   
           if mode==0:
              bigram_cnt=0
           else:
              bigram_cnt=1 
           for i in range(0,len(train)):
              if (key not in init_token) and (key1 not in init_token):  
                if (" "+key+" "+key1+" " in train[i]): 
                   bigram_cnt=bigram_cnt+train[i].count(" "+key+" "+key1+" ")
                elif (" "+key+"  "+key1+" " in train[i]): 
                     bigram_cnt=bigram_cnt+train[i].count(" "+key+"  "+key1+" ")
                elif (" "+key+" "+key1+" " in train[i]) and (" "+key+"  "+key1+" " in train[i]): 
                     bigram_cnt=bigram_cnt+train[i].count(" "+key+" "+key1+" ")+train[i].count(" "+key+"  "+key1+" ")     
                   #if key=="john" and key1=="boulnois":
                   #   print train[i] 
              else:
                if (key+" "+key1 in train[i]) or (key+"  "+key1 in train[i]):
                   bigram_cnt=bigram_cnt+1  
                   
           bigram[key][key1][0]=(bigram_cnt)
           bigram[key][key1][1]=(bigram_cnt/vocab[key])  
           print key, key1
   
   #print "Bigram is", bigram    
   
   #Calculating the prob of each sentence in the test set
   for i in range(0,len(test)):  
       vocab_test=test[i].split()
       probsum=1
       bigram_test_count_each=[]
       bigram_test_prob_each=[]
       for j in range(0,len(vocab_test)-1):
           prob=bigram[vocab_temp[j]][vocab_temp[j+1]][1]
           bigram_test_count_each.append((vocab_temp[j+1]+"|"+vocab_temp[j],bigram[vocab_temp[j]][vocab_temp[j+1]][0]))
           bigram_test_prob_each.append((vocab_temp[j+1]+"|"+vocab_temp[j],bigram[vocab_temp[j]][vocab_temp[j+1]][1]))
           probsum=probsum*prob
       bigram_test_count.append(bigram_test_count_each) 
       bigram_test_prob.append(bigram_test_prob_each) 
       prob_test.append((test_to_write[i],probsum))
       #prob_test.append(probsum)
           
   #Show all the metric of the test corpus
   print "*******************************************************************"
   print "The bigram counts for each sentence are",        bigram_test_count
   print "The bigram probabilities for each sentence are", bigram_test_prob
   print "The probability of each sentence is",            prob_test
   print "*******************************************************************"
    
       
    