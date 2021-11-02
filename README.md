# mt-ganbert

### Author: Claudia Breazzano - Danilo Croce

This repository contains the code of the MT-GANBERT model: Multi-Task and Generative Adversarial Learning for sustainable Language Processing. The implementation is in Pytorch and it started by using the code available of the model MT-DNN (https://github.com/namisan/mt-dnn), and inside GANBERT (https://github.com/crux82/ganbert) was implemented.
So in order to use this model you use that code with added parameters that allow you to activate GANBERT rather than the combined version. In this folder there are 4 python books:

0. BERT-based model.ipynb, in which you can launch the training of a simple model based solely on a transformer, in particular Italian Bert-base model, UmBERTo;
1. MT-DNN model.ipynb, in which you can launch the training of a model that applies multi task learning, MT-DNN, again based on UmBERTo;
2. GANBERT model.ipynb, in which you can launch the training of GANBERT;
3. MT-GANBERT model.ipynb, in which you can launch the training of MT-GANBERT;
These notebooks are pretty much the same, only they give you the ability to train each model.

# DATASETS:
In this work the following 6 tasks are considered:
1.   Hate Spech Detection. In particular the Twitter dataset of HaSpeeDe (http://www.di.unito.it/~tutreeb/haspeede-evalita18/) is used, the task of Hate Spech Detection organized within Evalita 2018, the 6th evaluation campaign of Natural Language Processing and Speech tools for Italian;
2.   Automatic Misogyny Identification, in particular the identification of misogyny and non-misogyny tweets, using the dataset of one of the tasks proposed by AMI 2018 (https://amievalita2018.wordpress.com), the task on Automatic Misogyny Identification organized in Evalita 2018;
3.   Identification of misogynistic categories: stereotype, sexual_harassment, discredit, using the dataset of one of the tasks proposed by AMI 2018 (https://amievalita2018.wordpress.com);
4.   Hate Speech Identification, using the dataset of the sub-task of Hate Speech Identification in DANKMEMEs (https://dankmemes2020.fileli.unipi.it), the task of MEMEs recognition, organized in Evalita 2020;
5.   Sentiment analysis, in particular recognition of objective and subjective tweets, using the dataset of SENTIPOLC 2016 (http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html), organized in Evalita 2016;
6.   Sentiment analysis, in particular recognition of positive, negative and neutral tweets, using the dataset of SENTIPOLC 2016 (http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html), organized in Evalita 2016.

For the misogyny tasks and for the Sentiment Analysis tasks, two sub-tasks of the AMI 2018 and, respectively, of SENTIPOLC 2016 are considered, so for everyone two distinct datasets were generated from a single dataset, in order to constitute the datasets of two different tasks. In fact, in the repository of original dataset, there is only one file (respectively train and test), Misogyny task and one of Sentiment analysis task.
The repository that contains the original datasets is "mt-ganbert/mttransformer/tsv_files".
Through the script "script_tsv.py", the original the datasets are transformed into new files, based on the amount of data you want to use, based on the model you want to use in notebooks made available in this github. The files .tsv generated by the script are placed in "mt-ganbert/mttransformer/data".
