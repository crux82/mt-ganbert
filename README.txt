# mt-ganbert

This repository contains the code of the MT-GANBERT model: Multi-Task and Generative Adversarial Learning for sustainable Language Processing. The implementation is in Pytorch and it started by using the code available of the model MT-DNN (https://github.com/namisan/mt-dnn), and inside ganbert was implemented.
So in order to use this model you use that code with added parameters that allow you to activate GANBERT rather than the combined version. In this folder there are 4 python books:
- 0. BERT-based model.ipynb, in which you can launch the training of a simple model based solely on a transformer, in particular Italian Bert-base model, UmBERTo;
- 1. MT-DNN model.ipynb, in which you can launch the training of a model that applies multi task learning, MT-DNN, again based on UmBERTo;
- 2. GANBERT model.ipynb, in which you can launch the training of GANBERT;
- 3. MT-GANBERT model.ipynb, in which you can launch the training of MT-GANBERT;
These notebooks are pretty much the same, only they give you the ability to train each model.

#DATASETS:
In this work the following 6 tasks are considered:
1.   HaSpeeDe: Hate Spech Recognition
2.   AMI A: Automatic Misogyny Identification (misogyny, not mysogyny)
3.   AMI B: Automatic Misogyny Identification (misogyny_category: stereotype, sexual_harassment, discredit)
4.   DANKMEMEs: Hate Spech Recognition in MEMEs sentences
5.   SENTIPOLC 1: Sentiment Polarity Classification (objective, subjective)
6.   SENTIPOLC 2: Sentiment Polarity Classification (polarity: positive, negative, neutral)

The original datasets of each task were taken from the following competitions ... Through the script "script_tsv.py", the .tsv files to be used in notebooks for model training are obtained.
