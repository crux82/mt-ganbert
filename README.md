# mt-ganbert

### Author: Claudia Breazzano - Danilo Croce

This repository contains the code of the paper **[MT-GANBERT model: Multi-Task and Generative Adversarial Learning for sustainable Language Processing](http://ceur-ws.org/Vol-3015/paper133.pdf
)** published in the *5th Workshop on Natural Language for Artificial Intelligence* (NL4AI2021).

This code is implemented in Pytorch and it started by using the code available of the model [MT-DNN](https://github.com/namisan/mt-dnn), and inside [GANBERT](https://github.com/crux82/ganbert).

## Repository description

This repository contains:

- the `mt-transformer` folder which contains the actual architecture implementation code (tha is a MT-DNN implementation modified with the addition of GAN-BERT);


- four Pythonbooks that show how to implement Generative Adversarial and/or Multi-task learning.

	1. `0_BERT-based_model.ipynb` implements a *basic* model, i.e., a straigth  BERT implementation. It is a baseline model that does not use nor GANBERT or Multi-task learning. Moreover, since the addressed tasks in the papare involve the Italian Language, here an Italian pre-trained Bert-base model is used: [UmBERTo](https://github.com/musixmatchresearch/umberto).


	2. `1_MT-DNN_model.ipynb` implements a model that applies multi task learning, MT-DNN, again based on UmBERTo (but NOT GANBERT). Again, this is a baseline model.

	3. `2_GANBERT_model.ipynb`, implements the training of GANBERT, without Multi-task learning. This is the last baseline model.

	4. `3_MT-GANBERT_model.ipynb` **implements MT-GANBERT** that combines Multi-task learning and Semi-supervised Learning.


-  and the `script-tsv.py` *load the training and test material* from the folder `original_data` and pre-process them to be used by library.
**Note**: this script is included in this repository only to ensure the reproducibility of the experiments. The training and test material is already included in the `data` folder in a tsv format that can be directly readed by the Pythonbooks.


## How to use it

This code shows how to train a single BERT based models on six different tasks at the same time, using Multi-task learning.

You can just refer to the specific python book here reported to evaualate MT-GANBERT or the specific (BERT, MT-DNN and GANBERT) baselines.

Each python book will load the data (already pre-processed by `script-tsv.py`) and replicate the different measures.


## How to cite this work

If you think that MT-GANBERT is useful for your research, please refers to the following paper. Thank you!

```
@inproceedings{DBLP:conf/aiia/BreazzanoC021,
  author    = {Claudia Breazzano and
               Danilo Croce and
               Roberto Basili},
  editor    = {Elena Cabrio and
               Danilo Croce and
               Lucia C. Passaro and
               Rachele Sprugnoli},
  title     = {{MT-GAN-BERT:} Multi-Task and Generative Adversarial Learning for
               Sustainable Language Processing},
  booktitle = {Proceedings of the Fifth Workshop on Natural Language for Artificial
               Intelligence {(NL4AI} 2021) co-located with 20th International Conference
               of the Italian Association for Artificial Intelligence (AI*IA 2021),
               Online event, November 29, 2021},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {3015},
  publisher = {CEUR-WS.org},
  year      = {2021},
  url       = {http://ceur-ws.org/Vol-3015/paper133.pdf},
  timestamp = {Wed, 24 Nov 2021 15:16:29 +0100},
  biburl    = {https://dblp.org/rec/conf/aiia/BreazzanoC021.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Datasets and task:

MT-GANBERT is trained on six tasks involving the automatic recognition of Sentiment and Abusive Language over Italian texts.

Here the list of datasets

1.   [Hate Spech Detection](http://www.di.unito.it/~tutreeb/haspeede-evalita18/) (HaSpeeDe) according to the Hate Spech Detection task organized within Evalita 2018, the 6th evaluation campaign of Natural Language Processing and Speech tools for Italian;
2.   [Automatic Misogyny Identification](https://amievalita2018.wordpress.com) (AMI), according to the task of identification of misogyny and non-misogyny tweets, organized in Evalita 2018;
3.   [Automatic Misogyny Identification](https://amievalita2018.wordpress.com) (AMI), according to the task of the identification of misogynistic categories: stereotype, sexual_harassment, discredit, that was organized in Evalita 2018;
4.   [Hate Speech Identification](https://dankmemes2020.fileli.unipi.it), the task of MEMEs recognition, organized in Evalita 2020;
5.   [Sentiment analysis](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html) (SENTIPOLC), according to the task of recognition of objective and subjective tweets, organized in Evalita 2016;
6.   [Sentiment analysis](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html) (SENTIPOLC), according to the task of recognition of positive, negative and neutral tweets, organized in Evalita 2016.

## References

- [MT-GANBERT](http://ceur-ws.org/Vol-3015/paper133.pdf) paper
- [GAN-BERT](https://github.com/crux82/ganbert) repository
- [GAN-BERT](https://aclanthology.org/2020.acl-main.191.pdf) paper
- [MT-DNN](https://github.com/namisan/mt-dnn) repository
- [MT-DNN](https://arxiv.org/pdf/1901.11504.pdf) paper
