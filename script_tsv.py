import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import random
import tensorflow as tf
import torch

#directory of tasks dataset
os.chdir("original_data")

#destination path to create tsv files, dipends on data cutting
path_0 = "/Users/claudia/Documents/mt-ganbert/mttransformer/data/0"
path_100 = "/Users/claudia/Documents/mt-ganbert/mttransformer/data/100"
path_200 = "/Users/claudia/Documents/mt-ganbert/mttransformer/data/200"
path_500 = "/Users/claudia/Documents/mt-ganbert/mttransformer/data/500"

#if you use a model with gan the flag "apply_gan" is True, else False
apply_gan=False
#data cutting
number_labeled_examples=0 #0-100-200-500
#if you want activate balancing, that is used only in the model Multi-task, MT-DNN and MT-GANBERT
balancing=False

#path train and test dataset of the task
tsv_haspeede_train = 'haspeede_TW-train.tsv'
tsv_haspeede_test = 'haspeede_TW-reference.tsv'
tsv_AMI2018_train = 'AMI2018_it_training.tsv'
tsv_AMI2018_test = 'AMI2018_it_testing.tsv'
tsv_AMI2018_train = 'AMI2018_it_training.tsv'
tsv_AMI2018_test = 'AMI2018_it_testing.tsv'
tsv_DANKMEMES2020_train = 'dankmemes_task2_train.csv'
tsv_DANKMEMES2020_test = 'hate_test.csv'
tsv_SENTIPOLC2016_train = 'training_set_sentipolc16.csv'
tsv_SENTIPOLC2016_test = 'test_set_sentipolc16_gold2000.csv'
tsv_SENTIPOLC2016_train = 'training_set_sentipolc16.csv'
tsv_SENTIPOLC2016_test = 'test_set_sentipolc16_gold2000.csv'

#Upload the dataset of all task as dataframes
#haspeede_TW
df_train = pd.read_csv(tsv_haspeede_train, delimiter='\t', names=('id','sentence','label'))
df_train = df_train[['id']+['label']+['sentence']]
df_test = pd.read_csv(tsv_haspeede_test, delimiter='\t', names=('id','sentence','label'))
df_test = df_test[['id']+['label']+['sentence']]
#AMI2018A
df_train2 = pd.read_csv(tsv_AMI2018_train, delimiter='\t')
df_train2 = df_train2[['id']+['misogynous']+['text']]
df_test2 = pd.read_csv(tsv_AMI2018_test, delimiter='\t')
df_test2 = df_test2[['id']+['misogynous']+['text']]
#AMI2018B
df_train3 = pd.read_csv(tsv_AMI2018_train, delimiter='\t')
df = pd.DataFrame(columns=['id', 'misogyny_category', 'text'])
for ind in df_train3.index:
  if df_train3.misogynous[ind]==1:
    if df_train3.misogyny_category[ind] == 'stereotype':
      df = df.append({'id' : df_train3['id'][ind], 'misogyny_category' : 0, 'text' : df_train3['text'][ind] }, ignore_index=True)
    #elif df_train3.misogyny_category[ind] == 'dominance':
      #df = df.append({'id' : df_train3['id'][ind], 'misogyny_category' : 1, 'text' : df_train3['text'][ind] }, ignore_index=True)
    #elif df_train3.misogyny_category[ind] == 'derailing':
      #df = df.append({'id' : df_train3['id'][ind], 'misogyny_category' : 2, 'text' : df_train3['text'][ind] }, ignore_index=True)
    elif df_train3.misogyny_category[ind] == 'sexual_harassment':
      df = df.append({'id' : df_train3['id'][ind], 'misogyny_category' : 1, 'text' : df_train3['text'][ind] }, ignore_index=True)
    elif df_train3.misogyny_category[ind] == 'discredit':
      df = df.append({'id' : df_train3['id'][ind], 'misogyny_category' : 2, 'text' : df_train3['text'][ind] }, ignore_index=True)

df_train3 = df
df_test3 = pd.read_csv(tsv_AMI2018_test, delimiter='\t')
df = pd.DataFrame(columns=['id', 'misogyny_category', 'text'])
for ind in df_test3.index:
  if df_test3.misogynous[ind]==1:
    if df_test3.misogyny_category[ind] == 'stereotype':
      df = df.append({'id' : df_test3['id'][ind], 'misogyny_category' : 0, 'text' : df_test3['text'][ind] }, ignore_index=True)
    #elif df_test3.misogyny_category[ind] == 'dominance':
      #df = df.append({'id' : df_test3['id'][ind], 'misogyny_category' : 1, 'text' : df_test3['text'][ind] }, ignore_index=True)
    #elif df_test3.misogyny_category[ind] == 'derailing':
      #df = df.append({'id' : df_test3['id'][ind], 'misogyny_category' : 2, 'text' : df_test3['text'][ind] }, ignore_index=True)
    elif df_test3.misogyny_category[ind] == 'sexual_harassment':
      df = df.append({'id' : df_test3['id'][ind], 'misogyny_category' : 1, 'text' : df_test3['text'][ind] }, ignore_index=True)
    elif df_test3.misogyny_category[ind] == 'discredit':
      df = df.append({'id' : df_test3['id'][ind], 'misogyny_category' : 2, 'text' : df_test3['text'][ind] }, ignore_index=True)

df_test3 = df
#DANKMEMES2020
df_train4 = pd.read_csv(tsv_DANKMEMES2020_train, delimiter=',')
df_train4 = df_train4[['File']+['Hate Speech']+['Text']]
df_test4 = pd.read_csv(tsv_DANKMEMES2020_test, delimiter=',')
df_test4 = df_test4[['File']+['Hate Speech']+['Text']]
#SENTIPOLC20161
df_train5 = pd.read_csv(tsv_SENTIPOLC2016_train, delimiter=',')
df_train5 = df_train5[['idtwitter']+['subj']+['text']]
df_test5 = pd.read_csv(tsv_SENTIPOLC2016_test, delimiter=',')
df_test5 = df_test5[['idtwitter']+['subj']+['text']]

for ind in df_train5.index:
  if "\t" in df_train5.text[ind]:
    df_train5 = df_train5.replace(to_replace='\t', value='', regex=True)
#SENTIPOLC20162
df_train6 = pd.read_csv(tsv_SENTIPOLC2016_train, delimiter=',')
df = pd.DataFrame(columns=['idtwitter', 'polarity', 'text'])
for ind in df_train6.index:
  if df_train6['subj'][ind] == 1:
    if df_train6['opos'][ind] == 1 and df_train6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_train6['idtwitter'][ind], 'polarity' : 0, 'text' : df_train6['text'][ind] }, ignore_index=True)
    elif df_train6['opos'][ind] == 0 and df_train6['oneg'][ind] == 1:
      df = df.append({'idtwitter' : df_train6['idtwitter'][ind], 'polarity' : 1, 'text' : df_train6['text'][ind] }, ignore_index=True)
    elif df_train6['opos'][ind] == 0 and df_train6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_train6['idtwitter'][ind], 'polarity' : 2, 'text' : df_train6['text'][ind] }, ignore_index=True)
  else:
    if df_train6['opos'][ind] == 0 and df_train6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_train6['idtwitter'][ind], 'polarity' : 2, 'text' : df_train6['text'][ind] }, ignore_index=True)

df_train6 = df
for ind in df_train6.index:
  if "\t" in df_train6.text[ind]:
    df_train6 = df_train6.replace(to_replace='\t', value='', regex=True)

df_test6 = pd.read_csv(tsv_SENTIPOLC2016_test, delimiter=',')
df = pd.DataFrame(columns=['idtwitter', 'polarity', 'text'])
for ind in df_test6.index:
  if df_test6['subj'][ind] == 1:
    if df_test6['opos'][ind] == 1 and df_test6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_test6['idtwitter'][ind], 'polarity' : 0, 'text' : df_test6['text'][ind] }, ignore_index=True)
    elif df_test6['opos'][ind] == 0 and df_test6['oneg'][ind] == 1:
      df = df.append({'idtwitter' : df_test6['idtwitter'][ind], 'polarity' : 1, 'text' : df_test6['text'][ind] }, ignore_index=True)
    elif df_test6['opos'][ind] == 0 and df_test6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_test6['idtwitter'][ind], 'polarity' : 2, 'text' : df_test6['text'][ind] }, ignore_index=True)
  else:
    if df_test6['opos'][ind] == 0 and df_test6['oneg'][ind] == 0:
      df = df.append({'idtwitter' : df_test6['idtwitter'][ind], 'polarity' : 2, 'text' : df_test6['text'][ind] }, ignore_index=True)

df_test6 = df

#split train dev, in all tasks
train_dataset, dev_dataset = train_test_split(df_train, test_size=0.2, shuffle = True)
train_dataset2, dev_dataset2 = train_test_split(df_train2, test_size=0.2, shuffle = True)
train_dataset3, dev_dataset3 = train_test_split(df_train3, test_size=0.2, shuffle = True)
train_dataset4, dev_dataset4 = train_test_split(df_train4, test_size=0.2, shuffle = True)
train_dataset5, dev_dataset5 = train_test_split(df_train5, test_size=0.2, shuffle = True)
train_dataset6, dev_dataset6 = train_test_split(df_train6, test_size=0.2, shuffle = True)

#reduction of datasets in case of data cutting 100, 200, 500
if number_labeled_examples!=0:
  if number_labeled_examples==100:
    labeled = train_dataset.sample(n=100)
    unlabeled = train_dataset
    labeled2 = train_dataset2.sample(n=100)
    unlabeled2 = train_dataset2
    labeled3 = train_dataset3.sample(n=100)
    unlabeled3 = train_dataset3
    labeled4 = train_dataset4.sample(n=100)
    unlabeled4 = train_dataset4
    labeled5 = train_dataset5.sample(n=100)
    unlabeled5 = train_dataset5
    labeled6 = train_dataset6.sample(n=100)
    unlabeled6 = train_dataset6
    cond = unlabeled['id'].isin(labeled['id'])
    cond2 = unlabeled2['id'].isin(labeled2['id'])
    cond3 = unlabeled3['id'].isin(labeled3['id'])
    cond4 = unlabeled4['File'].isin(labeled4['File'])
    cond5 = unlabeled5['idtwitter'].isin(labeled5['idtwitter'])
    cond6 = unlabeled6['idtwitter'].isin(labeled6['idtwitter'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
    unlabeled2.drop(unlabeled2[cond2].index, inplace = True)
    unlabeled3.drop(unlabeled3[cond3].index, inplace = True)
    unlabeled4.drop(unlabeled4[cond4].index, inplace = True)
    unlabeled5.drop(unlabeled5[cond5].index, inplace = True)
    unlabeled6.drop(unlabeled6[cond6].index, inplace = True)
  elif number_labeled_examples==200:
    labeled = train_dataset.sample(n=200)
    unlabeled = train_dataset
    labeled2 = train_dataset2.sample(n=200)
    unlabeled2 = train_dataset2
    labeled3 = train_dataset3.sample(n=200)
    unlabeled3 = train_dataset3
    labeled4 = train_dataset4.sample(n=200)
    unlabeled4 = train_dataset4
    labeled5 = train_dataset5.sample(n=200)
    unlabeled5 = train_dataset5
    labeled6 = train_dataset6.sample(n=200)
    unlabeled6 = train_dataset6
    cond = unlabeled['id'].isin(labeled['id'])
    cond2 = unlabeled2['id'].isin(labeled2['id'])
    cond3 = unlabeled3['id'].isin(labeled3['id'])
    cond4 = unlabeled4['File'].isin(labeled4['File'])
    cond5 = unlabeled5['idtwitter'].isin(labeled5['idtwitter'])
    cond6 = unlabeled6['idtwitter'].isin(labeled6['idtwitter'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
    unlabeled2.drop(unlabeled2[cond2].index, inplace = True)
    unlabeled3.drop(unlabeled3[cond3].index, inplace = True)
    unlabeled4.drop(unlabeled4[cond4].index, inplace = True)
    unlabeled5.drop(unlabeled5[cond5].index, inplace = True)
    unlabeled6.drop(unlabeled6[cond6].index, inplace = True)
  elif number_labeled_examples==500:
    labeled = train_dataset.sample(n=500)
    unlabeled = train_dataset
    labeled2 = train_dataset2.sample(n=500)
    unlabeled2 = train_dataset2
    labeled3 = train_dataset3.sample(n=500)
    unlabeled3 = train_dataset3
    labeled4 = train_dataset4.sample(n=500)
    unlabeled4 = train_dataset4
    labeled5 = train_dataset5.sample(n=500)
    unlabeled5 = train_dataset5
    labeled6 = train_dataset6.sample(n=500)
    unlabeled6 = train_dataset6
    cond = unlabeled['id'].isin(labeled['id'])
    cond2 = unlabeled2['id'].isin(labeled2['id'])
    cond3 = unlabeled3['id'].isin(labeled3['id'])
    cond4 = unlabeled4['File'].isin(labeled4['File'])
    cond5 = unlabeled5['idtwitter'].isin(labeled5['idtwitter'])
    cond6 = unlabeled6['idtwitter'].isin(labeled6['idtwitter'])
    unlabeled.drop(unlabeled[cond].index, inplace = True)
    unlabeled2.drop(unlabeled2[cond2].index, inplace = True)
    unlabeled3.drop(unlabeled3[cond3].index, inplace = True)
    unlabeled4.drop(unlabeled4[cond4].index, inplace = True)
    unlabeled5.drop(unlabeled5[cond5].index, inplace = True)
    unlabeled6.drop(unlabeled6[cond6].index, inplace = True)
  #model with or without gan
  if apply_gan == True:
    print("MT-GANBERT")
    #dataset unlabeled with label -1
    unlabeled['label'] = unlabeled['label'].replace(0,-1)
    unlabeled['label'] = unlabeled['label'].replace(1,-1)
    unlabeled2['misogynous'] = unlabeled2['misogynous'].replace(0,-1)
    unlabeled2['misogynous'] = unlabeled2['misogynous'].replace(1,-1)
    unlabeled3['misogyny_category'] = unlabeled3['misogyny_category'].replace(0,-1)
    unlabeled3['misogyny_category'] = unlabeled3['misogyny_category'].replace(1,-1)
    unlabeled3['misogyny_category'] = unlabeled3['misogyny_category'].replace(2,-1)
    unlabeled3['misogyny_category'] = unlabeled3['misogyny_category'].replace(3,-1)
    unlabeled3['misogyny_category'] = unlabeled3['misogyny_category'].replace(4,-1)
    unlabeled4['Hate Speech'] = unlabeled4['Hate Speech'].replace(0,-1)
    unlabeled4['Hate Speech'] = unlabeled4['Hate Speech'].replace(1,-1)
    unlabeled5['subj'] = unlabeled5['subj'].replace(0,-1)
    unlabeled5['subj'] = unlabeled5['subj'].replace(1,-1)
    unlabeled6['polarity'] = unlabeled6['polarity'].replace(0,-1)
    unlabeled6['polarity'] = unlabeled6['polarity'].replace(1,-1)
    unlabeled6['polarity'] = unlabeled6['polarity'].replace(2,-1)
    train = pd.concat([labeled, unlabeled])
    train2 = pd.concat([labeled2, unlabeled2])
    train3 = pd.concat([labeled3, unlabeled3])
    train4 = pd.concat([labeled4, unlabeled4])
    train5 = pd.concat([labeled5, unlabeled5])
    train6 = pd.concat([labeled6, unlabeled6])
    dev = dev_dataset
    dev2 = dev_dataset2
    dev3 = dev_dataset3
    dev4 = dev_dataset4
    dev5 = dev_dataset5
    dev6 = dev_dataset6
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train),len(labeled), len(unlabeled)))
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train2),len(labeled2), len(unlabeled2)))
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train3),len(labeled3), len(unlabeled3)))
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train4),len(labeled4), len(unlabeled4)))
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train5),len(labeled5), len(unlabeled5)))
    print("Size of Train dataset is {}, with {} labeled and {} not labeled ".format(len(train6),len(labeled6), len(unlabeled6)))
    print("Size of Dev dataset is {} ".format(len(dev)))
    print("Size of Dev dataset is {} ".format(len(dev2)))
    print("Size of Dev dataset is {} ".format(len(dev3)))
    print("Size of Dev dataset is {} ".format(len(dev4)))
    print("Size of Dev dataset is {} ".format(len(dev5)))
    print("Size of Dev dataset is {} ".format(len(dev6)))
  else:
    print("MT-DNN, with reduction dataset")
    train = labeled
    train2 = labeled2
    train3 = labeled3
    train4 = labeled4
    train5 = labeled5
    train6 = labeled6
    dev = dev_dataset
    dev2 = dev_dataset2
    dev3 = dev_dataset3
    dev4 = dev_dataset4
    dev5 = dev_dataset5
    dev6 = dev_dataset6
    print("Size of Train dataset is {} ".format(len(labeled)))
    print("Size of Train dataset is {} ".format(len(labeled2)))
    print("Size of Train dataset is {} ".format(len(labeled3)))
    print("Size of Train dataset is {} ".format(len(labeled4)))
    print("Size of Train dataset is {} ".format(len(labeled5)))
    print("Size of Train dataset is {} ".format(len(labeled6)))
    print("Size of Dev dataset is {} ".format(len(dev)))
    print("Size of Dev dataset is {} ".format(len(dev2)))
    print("Size of Dev dataset is {} ".format(len(dev3)))
    print("Size of Dev dataset is {} ".format(len(dev4)))
    print("Size of Dev dataset is {} ".format(len(dev5)))
    print("Size of Dev dataset is {} ".format(len(dev6)))
else:
  print("MT-DNN")
  train = train_dataset
  train2 = train_dataset2
  train3 = train_dataset3
  train4 = train_dataset4
  train5 = train_dataset5
  train6 = train_dataset6
  dev = dev_dataset
  dev2 = dev_dataset2
  dev3=dev_dataset3
  dev4=dev_dataset4
  dev5=dev_dataset5
  dev6=dev_dataset6
  print("Size of Train dataset is {} ".format(len(train)))
  print("Size of Train dataset is {} ".format(len(train2)))
  print("Size of Train dataset is {} ".format(len(train3)))
  print("Size of Train dataset is {} ".format(len(train4)))
  print("Size of Train dataset is {} ".format(len(train5)))
  print("Size of Train dataset is {} ".format(len(train6)))
  print("Size of Dev dataset is {} ".format(len(dev)))
  print("Size of Dev dataset is {} ".format(len(dev2)))
  print("Size of Dev dataset is {} ".format(len(dev3)))
  print("Size of Dev dataset is {} ".format(len(dev4)))
  print("Size of Dev dataset is {} ".format(len(dev5)))
  print("Size of Dev dataset is {} ".format(len(dev6)))


#Balancing for:
#- MT-DNN, trained on the total dataset of each task
#- MT-GAN, trained on the chosen data cutting of each task

if balancing==True:
    if apply_gan== True:
      print("MT-GAN")
      max_train_un = max(len(unlabeled), len(unlabeled2), len(unlabeled3), len(unlabeled4), len(unlabeled5), len(unlabeled6))
      print(max_train_un)
    else:
      print("MT-DNN")
      unlabeled=train
      unlabeled2=train2
      unlabeled3=train3
      unlabeled4=train4
      unlabeled5=train5
      unlabeled6=train6
      max_train_un = max(len(unlabeled), len(unlabeled2), len(unlabeled3), len(unlabeled4), len(unlabeled5), len(unlabeled6))
      print(max_train_un)
    #double dataset
    df = pd.DataFrame(columns=['id', 'label', 'sentence'])
    count=0
    if len(unlabeled)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled):
          df = df.append({'id' : unlabeled.iloc[i, 0], 'label' : unlabeled.iloc[i, 1], 'sentence' : unlabeled.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled):
            df = df.append({'id' : unlabeled.iloc[count, 0], 'label' : unlabeled.iloc[count, 1], 'sentence' : unlabeled.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'id' : unlabeled.iloc[count, 0], 'label' : unlabeled.iloc[count, 1], 'sentence' : unlabeled.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled = df
    if apply_gan== True:
      train = pd.concat([labeled, unlabeled])
    else:
      train=unlabeled
    df = pd.DataFrame(columns=['id', 'misogynous', 'text'])
    count=0
    if len(unlabeled2)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled2):
          df = df.append({'id' : unlabeled2.iloc[i, 0], 'misogynous' : unlabeled2.iloc[i, 1], 'text' : unlabeled2.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled2):
            df = df.append({'id' : unlabeled2.iloc[count, 0], 'misogynous' : unlabeled2.iloc[count, 1], 'text' : unlabeled2.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'id' : unlabeled2.iloc[count, 0], 'misogynous' : unlabeled2.iloc[count, 1], 'text' : unlabeled2.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled2 = df
    if apply_gan==True:
      train2 = pd.concat([labeled2, unlabeled2])
    else:
      train2=unlabeled2
    df = pd.DataFrame(columns=['id', 'misogyny_category', 'text'])
    count=0
    if len(unlabeled3)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled3):
          df = df.append({'id' : unlabeled3.iloc[i, 0], 'misogyny_category' : unlabeled3.iloc[i, 1], 'text' : unlabeled3.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled3):
            df = df.append({'id' : unlabeled3.iloc[count, 0], 'misogyny_category' : unlabeled3.iloc[count, 1], 'text' : unlabeled3.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'id' : unlabeled3.iloc[count, 0], 'misogyny_category' : unlabeled3.iloc[count, 1], 'text' : unlabeled3.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled3 = df
    if apply_gan==True:
      train3 = pd.concat([labeled3, unlabeled3])
    else:
      train3=unlabeled3
    df = pd.DataFrame(columns=['File', 'Hate Speech', 'Text'])
    count=0
    if len(unlabeled4)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled4):
          df = df.append({'File' : unlabeled4.iloc[i, 0], 'Hate Speech' : unlabeled4.iloc[i, 1], 'Text' : unlabeled4.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled4):
            df = df.append({'File' : unlabeled4.iloc[count, 0], 'Hate Speech' : unlabeled4.iloc[count, 1], 'Text' : unlabeled4.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'File' : unlabeled4.iloc[count, 0], 'Hate Speech' : unlabeled4.iloc[count, 1], 'Text' : unlabeled4.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled4 = df
    if apply_gan==True:
      train4 = pd.concat([labeled4, unlabeled4])
    else:
      train4=unlabeled4
    df = pd.DataFrame(columns=['idtwitter', 'subj', 'text'])
    count=0
    if len(unlabeled5)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled5):
          df = df.append({'idtwitter' : unlabeled5.iloc[i, 0], 'subj' : unlabeled5.iloc[i, 1], 'text' : unlabeled5.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled5):
            df = df.append({'idtwitter' : unlabeled5.iloc[count, 0], 'subj' : unlabeled5.iloc[count, 1], 'text' : unlabeled5.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'idtwitter' : unlabeled5.iloc[count, 0], 'subj' : unlabeled5.iloc[count, 1], 'text' : unlabeled5.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled5 = df
    if apply_gan==True:
      train5 = pd.concat([labeled5, unlabeled5])
    else:
      train5=unlabeled5
    df = pd.DataFrame(columns=['idtwitter', 'polarity', 'text'])
    count=0
    if len(unlabeled6)<max_train_un:
      for i in range(max_train_un):
        if i < len(unlabeled6):
          df = df.append({'idtwitter' : unlabeled6.iloc[i, 0], 'polarity' : unlabeled6.iloc[i, 1], 'text' : unlabeled6.iloc[i, 2] }, ignore_index=True)
        else:
          if count < len(unlabeled6):
            df = df.append({'idtwitter' : unlabeled6.iloc[count, 0], 'polarity' : unlabeled6.iloc[count, 1], 'text' : unlabeled6.iloc[count, 2] }, ignore_index=True)
            count = count+1
          else:
            count = 0
            df = df.append({'idtwitter' : unlabeled6.iloc[count, 0], 'polarity' : unlabeled6.iloc[count, 1], 'text' : unlabeled6.iloc[count, 2] }, ignore_index=True)
            count = count+1
      unlabeled6 = df
    if apply_gan==True:
      train6 = pd.concat([labeled6, unlabeled6])
    else:
      train6=unlabeled6

#create directory, dipends on chosen data cutting of tasks, where the task .tsv files are placed
if number_labeled_examples=0:
    try:
        os.mkdir(path_0)
    except OSError:
        print ("Creation of the directory %s failed" % path_0)
    else:
        print ("Successfully created the directory %s " % path_0)
    os.chdir(path_0)
elif number_labeled_examples=100:
     try:
         os.mkdir(path_100)
     except OSError:
         print ("Creation of the directory %s failed" % path_100)
     else:
         print ("Successfully created the directory %s " % path_100)
     os.chdir(path_100)
elif number_labeled_examples=200:
     try:
         os.mkdir(path_200)
     except OSError:
         print ("Creation of the directory %s failed" % path_200)
     else:
         print ("Successfully created the directory %s " % path_200)
     os.chdir(path_200)
elif number_labeled_examples=500:
     try:
         os.mkdir(path_500)
     except OSError:
         print ("Creation of the directory %s failed" % path_500)
     else:
         print ("Successfully created the directory %s " % path_500)
     os.chdir(path_500)

#The code is using surfix to distinguish what type of set it is ("_train","_dev" and "_test"). So:
#1.   make sure your train set is named as "TASK_train" (replace TASK with your task name)
#2.   make sure your dev set and test set ends with "_dev" and "_test".
#3.   add your task into task define config (task_def file):

#  Here is a piece of example task define config :
#  <pre>haspeede-TW:
#    data_format: PremiseOnly
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

#You can choose loss (for BERT-based model and MT-DNN, the GANBERT the loss is in the model implementation), from pre-defined losses in file [mt_dnn/loss.py](../mt_dnn/loss.py), and you can implement your customized losses into this file and specify it in the task config.

#In this case the task_type are Classification task, but the MT-DNN implementation allows you to choose between different task type. More details in data_utils/task_def.py

#Also, specify how many classes in total in your task, under "n_class" field.

#Creation of files, for each task:
#-train: .tsv file
#-test: .tsv file
#-dev: .tsv file
#-task_def: .yml file

#haspeede_TW
#train
name_train = "haspeede-TW_train.tsv"
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
name_file = 'haspeede-TW_task_def.yml'


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
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")

#AMI2018A
#train
name_train = "AMI2018A_train.tsv"
id_train = train2.id
label_train = train2.misogynous
sentence_train = train2.text
#dev
name_dev = "AMI2018A_dev.tsv"
id_dev = dev2.id
label_dev = dev2.misogynous
sentence_dev = dev2.text
#test
name_test = "AMI2018A_test.tsv"
id_test = df_test2.id
label_test = df_test2.misogynous
sentence_test = df_test2.text
#task_def
name_file = 'AMI2018A_task_def.yml'

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


task = "AMI2018A:\n"

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
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")

#AMI2018B
#train
name_train = "AMI2018B_train.tsv"
id_train = train3.id
label_train = train3.misogyny_category
sentence_train = train3.text
#dev
name_dev = "AMI2018B_dev.tsv"
id_dev = dev3.id
label_dev = dev3.misogyny_category
sentence_dev = dev3.text
#test
name_test = "AMI2018B_test.tsv"
id_test = df_test3.id
label_test = df_test3.misogyny_category
sentence_test = df_test3.text
#task_def
name_file = 'AMI2018B_task_def.yml'


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


task = "AMI2018B:\n"

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
    f.write("  n_class: 3\n")
    f.write("  task_type: Classification\n")

#DANKMEMES2020
#train
name_train = "DANKMEMES2020_train.tsv"
id_train = train4.File
label_train = train4["Hate Speech"]
sentence_train = train4.Text
#dev
name_dev = "DANKMEMES2020_dev.tsv"
id_dev = dev4.File
label_dev = dev4["Hate Speech"]
sentence_dev = dev4.Text
#test
name_test = "DANKMEMES2020_test.tsv"
id_test = df_test4.File
label_test = df_test4["Hate Speech"]
sentence_test = df_test4.Text
#task_def
name_file = 'DANKMEMES2020_task_def.yml'


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


task = "DANKMEMES2020:\n"

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
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")

#SENTIPOLC20161
#train
name_train = "SENTIPOLC20161_train.tsv"
id_train = train5.idtwitter
label_train = train5.subj
sentence_train = train5.text
#dev
name_dev = "SENTIPOLC20161_dev.tsv"
id_dev = dev5.idtwitter
label_dev = dev5.subj
sentence_dev = dev5.text
#test
name_test = "SENTIPOLC20161_test.tsv"
id_test = df_test5.idtwitter
label_test = df_test5.subj
sentence_test = df_test5.text
#task_def
name_file = 'SENTIPOLC20161_task_def.yml'


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


task = "SENTIPOLC20161:\n"

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
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")

#SENTIPOLC20162
#train
name_train = "SENTIPOLC20162_train.tsv"
id_train = train6.idtwitter
label_train = train6.polarity
sentence_train = train6.text
#dev
name_dev = "SENTIPOLC20162_dev.tsv"
id_dev = dev6.idtwitter
label_dev = dev6.polarity
sentence_dev = dev6.text
#test
name_test = "SENTIPOLC20162_test.tsv"
id_test = df_test6.idtwitter
label_test = df_test6.polarity
sentence_test = df_test6.text
#task_def
name_file = 'SENTIPOLC20162_task_def.yml'


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


task = "SENTIPOLC20162:\n"

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
    f.write("  n_class: 3\n")
    f.write("  task_type: Classification\n")


#task_def of all tasks
name_file = "haspeede-TW_AMI2018A_AMI2018B_DANKMEMES2020_SENTIPOLC20161_SENTIPOLC20162_task_def.yml"

f = open(name_file, 'w')

with f:

    f.write("haspeede-TW:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")
    f.write("AMI2018A:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")
    f.write("AMI2018B:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 3\n")
    f.write("  task_type: Classification\n")
    f.write("DANKMEMES2020:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")
    f.write("SENTIPOLC20161:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 2\n")
    f.write("  task_type: Classification\n")
    f.write("SENTIPOLC20162:\n")
    if apply_gan == True:
      f.write("  data_format: Gan\n")
    else:
      f.write("  data_format: PremiseOnly\n")
    f.write("  enable_san: false\n")
    f.write("  metric_meta:\n")
    f.write("  - F1MAC\n")
    f.write("  - ACC\n")
    f.write("  loss: CeCriterion\n")
    f.write("  n_class: 3\n")
    f.write("  task_type: Classification\n")
