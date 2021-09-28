import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import random
import tensorflow as tf
import torch

os.chdir("/Users/claudia/Documents/mt-ganbert/mttransformer/tsv_files")
path = "/Users/claudia/Documents/mt-ganbert/mttransformer/tsv_files/tsv_transformed"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

apply_gan=False
number_labeled_examples=0 #0-100-200-500


#path train and test dataset of the task
tsv_task_train = 'haspeede_TW-train.tsv'
tsv_task_test = 'haspeede_TW-reference.tsv'
AMI = "A" #if task is AMI, choose between AMI A or AMI B
sentipolc = 1 #if task is sentipolc, choose between sentipolc 1 or sentipolc 2

if "haspeede" in tsv_task_train:
    df_train = pd.read_csv(tsv_task_train, delimiter='\t', names=('id','sentence','label'))
    df_train = df_train[['id']+['label']+['sentence']]
    df_test = pd.read_csv(tsv_task_test, delimiter='\t', names=('id','sentence','label'))
    df_test = df_test[['id']+['label']+['sentence']]
elif "AMI2018" in tsv_task_train and AMI == "A":
    df_train = pd.read_csv(tsv_task_train, delimiter='\t')
    df_train = df_train[['id']+['misogynous']+['text']]
    df_train = df_train.rename(columns={'misogynous': 'label','text': 'sentence'})
    df_test = pd.read_csv(tsv_task_test, delimiter='\t')
    df_test = df_test[['id']+['misogynous']+['text']]
    df_test = df_test.rename(columns={'misogynous': 'label','text': 'sentence'})
elif "AMI2018" in tsv_task_train and AMI == "B":
    df_train = pd.read_csv(tsv_task_train, delimiter='\t')
    df = pd.DataFrame(columns=['id', 'misogyny_category', 'text'])
    for ind in df_train.index:
      if df_train.misogynous[ind]==1:
        if df_train.misogyny_category[ind] == 'stereotype':
          df = df.append({'id' : df_train['id'][ind], 'misogyny_category' : 0, 'text' : df_train['text'][ind] }, ignore_index=True)
        #elif df_train.misogyny_category[ind] == 'dominance':
          #df = df.append({'id' : df_train['id'][ind], 'misogyny_category' : 1, 'text' : df_train['text'][ind] }, ignore_index=True)
        #elif df_train.misogyny_category[ind] == 'derailing':
          #df = df.append({'id' : df_train['id'][ind], 'misogyny_category' : 2, 'text' : df_train['text'][ind] }, ignore_index=True)
        elif df_train.misogyny_category[ind] == 'sexual_harassment':
          df = df.append({'id' : df_train['id'][ind], 'misogyny_category' : 1, 'text' : df_train['text'][ind] }, ignore_index=True)
        elif df_train.misogyny_category[ind] == 'discredit':
          df = df.append({'id' : df_train['id'][ind], 'misogyny_category' : 2, 'text' : df_train['text'][ind] }, ignore_index=True)
    df_train = df
    df_train = df_train.rename(columns={'misogyny_category': 'label','text': 'sentence'})
    df_test = pd.read_csv(tsv_task_test, delimiter='\t')
    df = pd.DataFrame(columns=['id', 'misogyny_category', 'text'])
    for ind in df_test.index:
      if df_test.misogynous[ind]==1:
        if df_test.misogyny_category[ind] == 'stereotype':
          df = df.append({'id' : df_test['id'][ind], 'misogyny_category' : 0, 'text' : df_test['text'][ind] }, ignore_index=True)
        #elif df_test.misogyny_category[ind] == 'dominance':
          #df = df.append({'id' : df_test['id'][ind], 'misogyny_category' : 1, 'text' : df_test['text'][ind] }, ignore_index=True)
        #elif df_test.misogyny_category[ind] == 'derailing':
          #df = df.append({'id' : df_test['id'][ind], 'misogyny_category' : 2, 'text' : df_test['text'][ind] }, ignore_index=True)
        elif df_test.misogyny_category[ind] == 'sexual_harassment':
          df = df.append({'id' : df_test['id'][ind], 'misogyny_category' : 1, 'text' : df_test['text'][ind] }, ignore_index=True)
        elif df_test.misogyny_category[ind] == 'discredit':
          df = df.append({'id' : df_test['id'][ind], 'misogyny_category' : 2, 'text' : df_test['text'][ind] }, ignore_index=True)
    df_test = df
    df_test = df_test.rename(columns={'misogyny_category': 'label', 'text': 'sentence'})
elif "dankmemes" in tsv_task_train:
    df_train = pd.read_csv(tsv_task_train, delimiter=',')
    df_train = df_train[['File']+['Hate Speech']+['Text']]
    df_train = df_train.rename(columns={'File':'id','Hate Speech': 'label', 'Text': 'sentence'})
    df_test = pd.read_csv(tsv_task_test, delimiter=',')
    df_test = df_test[['File']+['Hate Speech']+['Text']]
    df_test = df_test.rename(columns={'File':'id','Hate Speech': 'label', 'Text': 'sentence'})
elif "sentipolc16" in tsv_task_train and sentipolc == 1:
    df_train = pd.read_csv(tsv_task_train, delimiter=',')
    df_train = df_train[['idtwitter']+['subj']+['text']]
    df_train = df_train.rename(columns={'idtwitter':'id','subj': 'label', 'text': 'sentence'})
    df_test = pd.read_csv(tsv_task_test, delimiter=',')
    df_test = df_test[['idtwitter']+['subj']+['text']]
    df_test = df_test.rename(columns={'idtwitter':'id','subj': 'label', 'text': 'sentence'})
    for ind in df_train.index:
      if "\t" in df_train.text[ind]:
        df_train = df_train.replace(to_replace='\t', value='', regex=True)
elif "sentipolc16" in tsv_task_train and sentipolc == 2:
    df_train = pd.read_csv(tsv_SENTIPOLC2016_train, delimiter=',')
    df = pd.DataFrame(columns=['idtwitter', 'polarity', 'text'])
    for ind in df_train.index:
      if df_train['subj'][ind] == 1:
        if df_train['opos'][ind] == 1 and df_train['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_train['idtwitter'][ind], 'polarity' : 0, 'text' : df_train['text'][ind] }, ignore_index=True)
        elif df_train['opos'][ind] == 0 and df_train['oneg'][ind] == 1:
          df = df.append({'idtwitter' : df_train['idtwitter'][ind], 'polarity' : 1, 'text' : df_train['text'][ind] }, ignore_index=True)
        elif df_train['opos'][ind] == 0 and df_train['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_train['idtwitter'][ind], 'polarity' : 2, 'text' : df_train['text'][ind] }, ignore_index=True)
      else:
        if df_train['opos'][ind] == 0 and df_train['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_train['idtwitter'][ind], 'polarity' : 2, 'text' : df_train['text'][ind] }, ignore_index=True)
    df_train = df
    df_train = df_train.rename(columns={'idtwitter':'id','polarity': 'label', 'text': 'sentence'})
    for ind in df_train.index:
      if "\t" in df_train.text[ind]:
        df_train = df_train.replace(to_replace='\t', value='', regex=True)
    df_test = pd.read_csv(tsv_SENTIPOLC2016_test, delimiter=',')
    df = pd.DataFrame(columns=['idtwitter', 'polarity', 'text'])
    for ind in df_test.index:
      if df_test['subj'][ind] == 1:
        if df_test['opos'][ind] == 1 and df_test['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_test['idtwitter'][ind], 'polarity' : 0, 'text' : df_test['text'][ind] }, ignore_index=True)
        elif df_test['opos'][ind] == 0 and df_test['oneg'][ind] == 1:
          df = df.append({'idtwitter' : df_test['idtwitter'][ind], 'polarity' : 1, 'text' : df_test['text'][ind] }, ignore_index=True)
        elif df_test['opos'][ind] == 0 and df_test['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_test['idtwitter'][ind], 'polarity' : 2, 'text' : df_test['text'][ind] }, ignore_index=True)
      else:
        if df_test['opos'][ind] == 0 and df_test['oneg'][ind] == 0:
          df = df.append({'idtwitter' : df_test['idtwitter'][ind], 'polarity' : 2, 'text' : df_test['text'][ind] }, ignore_index=True)
    df_test = df
    df_test = df_test.rename(columns={'idtwitter':'id','polarity': 'label', 'text': 'sentence'})


#split train dev
train_dataset, dev_dataset = train_test_split(df_train, test_size=0.2, shuffle = True)

#reduction
if number_labeled_examples!=0:
  if number_labeled_examples==100:
    labeled = train_dataset.sample(n=100)
    unlabeled = train_dataset
    cond = unlabeled['id'].isin(labeled['id'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
  elif number_labeled_examples==200:
    labeled = train_dataset.sample(n=200)
    unlabeled = train_dataset
    cond = unlabeled['id'].isin(labeled['id'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
  elif number_labeled_examples==500:
    labeled = train_dataset.sample(n=500)
    unlabeled = train_dataset
    cond = unlabeled['id'].isin(labeled['id'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
  #model with or without gan
  if apply_gan == True:
    print("GANBERT")
    #dataset unlabeled with label -1
    unlabeled['label'] = unlabeled['label'].replace(0,-1)
    unlabeled['label'] = unlabeled['label'].replace(1,-1)
    train = pd.concat([labeled, unlabeled])
    dev = dev_dataset
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train),len(labeled), len(unlabeled)))
    print("Size of Dev dataset is {} ".format(len(dev)))
  else:
    print("BERT-based model, with reduction dataset")
    train = labeled
    dev = dev_dataset
    print("Size of Train dataset is {} ".format(len(labeled)))
    print("Size of Dev dataset is {} ".format(len(dev)))
else:
  print("BERT-based model")
  train = train_dataset
  dev = dev_dataset
  print("Size of Train dataset is {} ".format(len(train)))
  print("Size of Dev dataset is {} ".format(len(dev)))

os.chdir(path)


#The code is using surfix to distinguish what type of set it is ("_train","_dev" and "_test"). So:
#1.   make sure your train set is named as "TASK_train" (replace TASK with your task name)

#2.   make sure your dev set and test set ends with "_dev" and "_test".
#3.   add your task into task define config (task_def file):

#  Here is a piece of example task define config :
#  <pre>haspeede-TW:
#    data_format: PremiseOnly
#    ensable_san: false
#    labels:
#    - contradiction
#    - neutral
#    - entailment
#    metric_meta:
#    - ACC
#    loss: CeCriterion
#    n_class: 3
#    task_type: Classification</pre>

#Choose the correct data format based on your task, in this notebook are used 2 types of data formats, coresponds to different tasks:
#  1. "PremiseOnly" : single text, i.e. premise. Data format is "id" \t "label" \t "premise" .
#  2. "Gan" : single text, i.e. premise. Data format is "id" \t "label" \t "premise" .

#ensable_san: Set "true" if you would like to use Stochastic Answer Networks(SAN) for your task.

#If you prefer using readable labels (text), you can specify what labels are there in your data set, under "labels" field.

#More details about metrics,please refer to [data_utils/metrics.py](../data_utils/metrics.py);

#You can choose loss (for BERT-based model and MT-DNN, the GANBERT loss is in the model), from pre-defined losses in file [mt_dnn/loss.py](../mt_dnn/loss.py), and you can implement your customized losses into this file and specify it in the task config.

#Specify what task type it is in your own task, choose one from types in:
#    1. Classification
#    2. Regression
#    3. Ranking
#    4. Span
#    5. SeqenceLabeling
#    6. MaskLM
#  More details in [data_utils/task_def.py](../data_utils/task_def.py)

#Also, specify how many classes in total in your task, under "n_class" field.


#train
name_train = "haspeede-TW_train.tsv"  #example for haspeede task
id_train = train.id
label_train = train.label
sentence_train = train.sentence

#dev
name_dev = "haspeede-TW_dev.tsv"
id_dev = dev.id
label_dev = dev.label
sentence_dev = dev.sentence

#test
name_test = "haspeede-TW_test.tsv"
id_test = df_test.id
label_test = df_test.label
sentence_test = df_test.sentence

#task_def
name_file = 'haspeede-TW_task_def.yml' #the task def file can be composed of all tasks


f = open(name_train, 'w')
with f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(id_train,label_train,sentence_train))

f = open(name_dev, 'w')
with f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(id_dev,label_dev,sentence_dev))

f = open(name_test, 'w')
with f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(id_test,label_test,sentence_test))

task = "haspeede-TW:\n"

f = open(name_file, 'w')
with f:
    f.write(task)
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 2\n") #change the number of classes based on the task
    f.write("  task_type: Classification\n")
