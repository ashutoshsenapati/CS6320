from __future__ import division # floating point division
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from os.path import isfile, join
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from nltk.corpus import stopwords 
import sklearn.datasets
import numpy as np
import nltk 
from nltk.corpus import stopwords
import re
import sys
from copy import deepcopy
from random import randint

def return_filelist_dir(dirname):
     files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
     return files

def loadfiles(filename):
    #print filename,type(filename)
    train=np.genfromtxt(filename,dtype=str,delimiter="/n")
    return train

def build_classifier(Xfeat,Xlabel,classifier,reg="l2"):

 print "The features are", Xfeat.shape,Xlabel.shape, classifier, reg
 if classifier=="nbayes":
    clf = MultinomialNB()
    print "Building NB classifier"
    #clf.fit(Xfeat, Xlabel)

 else:
   if classifier=="regression":
      if reg=="no":
         clf=LogisticRegression(C=1e5)
	 print "Building Logistic Regression no regularization"
      elif reg=="l1":
         clf=LogisticRegression(penalty=reg)
	 print "Building Logistic Regression with l1 regularization"
      elif reg=="l2":
         clf=LogisticRegression()
	 print "Building Logistic Regression with l2 regularisation"

 clf.fit(Xfeat, Xlabel)
 return clf


if __name__ == "__main__":

 #trainloc="/home/011/s/sx/sxd170431/bigram/A2_Data/aclImdb/train/"
 #testloc="/home/011/s/sx/sxd170431/bigram/A2_Data/aclImdb/test/"
 trainloc=sys.argv[1]
 testloc=sys.argv[2]
 postrainpath=trainloc+"/pos/"
 negtrainpath=trainloc+"/neg/"
 postestpath=testloc+"/pos/"
 negtestpath=testloc+"/neg/"
 print "The positive abd negative instance path locations are",postrainpath,negtrainpath
 train_data=[]
 train_label=[]
 test_data=[]
 test_label=[]
 print "The command line arguments are", sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
 if sys.argv[4]=="nbayes":
    reg="None"
 else:
    reg=sys.argv[6] 

 """Read the file lists in the pos and neg folders"""
 pos_trainfiles=return_filelist_dir(postrainpath)
 #print "Length of posfile", len(pos_trainfiles)
 neg_trainfiles=return_filelist_dir(negtrainpath)
 #print "Length of negfile", len(neg_trainfiles)
 pos_testfiles=return_filelist_dir(postestpath)
 neg_testfiles=return_filelist_dir(negtestpath)

 """Load the contents of the train files in np array"""
 for i in range(0,len(pos_trainfiles)):
     train_data.append(loadfiles(postrainpath+pos_trainfiles[i]))
     train_label.append(1)
 for i in range(0,len(neg_trainfiles)):
     train_data.append(loadfiles(negtrainpath+neg_trainfiles[i]))
     train_label.append(0)
 train_label=np.asarray(train_label)
 
 """Load the contents of the test files in np array"""
 for i in range(0,len(pos_testfiles)):
      test_data.append(loadfiles(postestpath+pos_testfiles[i]))
      test_label.append(1)
 for i in range(0,len(neg_testfiles)):
     test_data.append(loadfiles(negtestpath+neg_testfiles[i]))
     test_label.append(0)

 test_label=np.asarray(test_label)

 """Preprocessing the training strings"""
 for i in range(0,len(train_data)):
   
     np.char.replace(train_data[i],".", "")
     np.char.replace(train_data[i],"(", " ")
     np.char.replace(train_data[i],")", " ")
     np.char.replace(train_data[i],"[", " ")
     np.char.replace(train_data[i],"]", " ")
     np.char.replace(train_data[i],"'", "")
     np.char.replace(train_data[i],'"', "")
     np.char.replace(train_data[i],"-","")
     np.char.replace(train_data[i],",","")
     np.char.replace(train_data[i],";","")
     #strip spaces from beg and end
     np.char.strip(train_data[i])

 #train_data=np.asarray(train_data) 
 """Preprocessing for test strings"""
 for i in range(0,len(test_data)):
     np.char.replace(test_data[i],".", "")
     np.char.replace(test_data[i],"(", " ")
     np.char.replace(test_data[i],")", " ")
     np.char.replace(test_data[i],"[", " ")
     np.char.replace(test_data[i],"]", " ")
     np.char.replace(test_data[i],"'", "")
     np.char.replace(test_data[i],'"', "")
     np.char.replace(test_data[i],"-","")
     np.char.replace(test_data[i],",","")
     np.char.replace(test_data[i],";","")
     #strip spaces from beg and end 
     np.char.strip(test_data[i])

 #test_data=np.asarray(test_data)
 #print "The shapes are", train_data.shape,test_data.shape
 #Some reformatting'
 train_data1=[]
 test_data1=[]

 #if int(sys.argv[5])==1:
 #   """get the list of stopwords to remove"""
 #   print "Getting list of Stopwords"
 #   stop_words = set(stopwords.words('english'))
 #   print stop_words

 for i in range(0,len(train_data)):
     train_str=train_data[i].tostring()
     #if int(sys.argv[5])==1:
     #   """Remove stopwords from traindata"""
     #   wordlist=train_str.split()
#	word_to_remove=[]
#	for words in wordlist:
#	    if words in stop_words:
#	       word_to_remove.append(words)
#	for words in word_to_remove:
#	    wordlist.remove(words)
 #       train_data1.append(" ".join(wordlist))
  #   else:
     train_data1.append(train_str)

 for i in range(0,len(test_data)):
     test_str=test_data[i].tostring()
     #if int(sys.argv[5])==1:
      #   """Remove stopwords from test data"""
      #   wordlist=test_str.split()
#	 word_to_remove=[]
#	 for words in wordlist:
 #            if words in stop_words:
#		word_to_remove.append(words)
#	 for words in word_to_remove:
#	     wordlist.remove(words)
#	 test_data1.append(" ".join(wordlist))
 #    else:
     test_data1.append(test_str)

 """Feature creation from text"""
 if sys.argv[3]=="bow":
    print "Creating bag of words features"
    if int(sys.argv[5])==1:
       count_vect = CountVectorizer(stop_words='english')
       print "Removing stopwords"
    else:
       count_vect = CountVectorizer()
    train_data_feat = count_vect.fit_transform(train_data1)
    test_data_feat=count_vect.transform(test_data1)
 else:
    print "Creating tfidf features"
    if int(sys.argv[5])==1:
       count_vect = CountVectorizer(stop_words='english')
       print "Removing stopwords"
    else:
       count_vect = CountVectorizer()
    train_data_count = count_vect.fit_transform(train_data1)
    tfidf_transformer = TfidfTransformer()
    train_data_feat = tfidf_transformer.fit_transform(train_data_count)

    test_data_count = count_vect.transform(test_data1)
    test_data_feat = tfidf_transformer.transform(test_data_count)


 print "THe train feature, train label, test feature, test label are", train_data_feat.shape,train_label.shape,test_data_feat.shape,test_label.shape

 """Building of classifier"""

 model=build_classifier(train_data_feat,train_label,sys.argv[4],reg)
 """getting output from learnt classifier"""
 test_predict=model.predict(test_data_feat)

 """Jotting down the various metric"""
 accuracy=accuracy_score(test_label,test_predict)
 precision=precision_score(test_label,test_predict)
 recall=recall_score(test_label,test_predict)
 fscore=f1_score(test_label,test_predict)

 print "THe accuracy,precision,recall and F1-score are", accuracy,precision, recall,fscore




