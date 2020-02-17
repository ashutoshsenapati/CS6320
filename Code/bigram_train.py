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
#trainpath="C://Users//sxd170431//Desktop//Work//NLP//A1//Data//"
#testpath="C://Users//sxd170431//Desktop//Work//NLP//A1//Data//"
vocab={}
bigram={}

def readfile(filename):
    data=[]
    file1 = open(filename,"r") 
    for line in file1.readlines():
        data.append(line)
    return data
    
#The main python module starts here
if __name__== "__main__":
   trainpath=sys.argv[1]
   trainpath=sys.argv[2]
   mode=int(sys.argv[3]) 
   init_token=['<s>','</s>'] 
   train=readfile(trainpath+"//train.txt")
   test= readfile(trainpath+"//test.txt")
   test_to_write=deepcopy(test)
   #print "The length of train is", len(train)
   #print "The length of test is", len(test)
   #If mode is 0, then no smoothening, else smoothening
   
   if mode==0:
      print "bigram probabilities without laplace smoothing, mode= ", mode
   else:
      print "bigram probabilities WITH laplace smoothing, mode= ", mode 
    
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
       #populate the unigram dictionary
       for j in vocab_train:
           if j not in vocab.keys():
              vocab[j]=0
       #populate the bigram dictionary
       #print train[i]
       for j in range(0,len(vocab_train)-1):
           key=vocab_train[j]
           key1=vocab_train[j+1]
           #print key, key1
           if key not in bigram.keys():
              bigram[key]={}
           if key1 not in bigram[key].keys():   
              bigram[key][key1]=[0,0]
           #print bigram
           #raw_input()
           
   #print "The size of unigram is", len(vocab)
   #print "The size of bigram is", len(bigram)
   
   vocab_size=len(vocab)
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

   #populate the count for the unigram model" 
   for key in vocab.keys():
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
   print "The unigram dictionary got populated"
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
              else:
                if (key+" "+key1 in train[i]) or (key+"  "+key1 in train[i]):
                   bigram_cnt=bigram_cnt+1  
                   
           bigram[key][key1][0]=(bigram_cnt)
           bigram[key][key1][1]=(bigram_cnt/vocab[key])  
           #print key, key1
   print "The bigram dictionary got populated"   

   print "Evaluating the sentences in the test set"     
   for i in range(0,len(test)):  
       vocab_test=test[i].split()
       probsum=1
       bigram_test_count_each=[]
       bigram_test_prob_each=[]
       for j in range(0,len(vocab_test)-1):
           prob=bigram[vocab_test[j]][vocab_test[j+1]][1]
           bigram_test_count_each.append((vocab_test[j]+","+vocab_test[j+1],bigram[vocab_test[j]][vocab_test[j+1]][0]))
           bigram_test_prob_each.append((vocab_test[j+1]+"|"+vocab_test[j],bigram[vocab_test[j]][vocab_test[j+1]][1]))
           probsum=probsum*prob
       bigram_test_count.append(bigram_test_count_each) 
       bigram_test_prob.append(bigram_test_prob_each) 
       prob_test.append((test_to_write[i],probsum))
           
   #Show all the metric of the test corpus
   print "*******************************************************************"
   print "BIGRAM COUNTS FOR THE SENTENCES ARE" 
   for i in range(0,len(test)):
       print "                                                    "       
       print bigram_test_count[i]
   print "*****************************************************************"
   print "BIGRAM PROBABILITIES FOR THE SENTENCES ARE" 
   for i in range(0,len(test)):
       print "                                                    "  
       print bigram_test_prob[i]
   print "*******************************************************************"
   print "PROBABILITY OF EACH SENTENCE IS"
   for i in range(0,len(test)):
       print "                                                    "
       print prob_test[i]
   print "*******************************************************************"
    
       
    