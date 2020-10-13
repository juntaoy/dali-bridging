# Multi-task Learning Based Neural Bridging Reference Resolution

## Introduction
This repository contains code introduced in the following paper:
 
**[Multi-task Learning Based Neural Bridging Reference Resolution](https://arxiv.org/abs/2003.03666)**  
Juntao Yu and Massimo Poesio 
In *Proceedings of he 28th International Conference on Computational Linguistics (COLING)*, 2020

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that modify and run `extract_bert_features/extract_bert_features.sh` to compute the BERT embeddings for your training or testing.
* You also need to download context-independent word embeddings such as fasttext or GloVe embeddings that required by the system.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://www.dropbox.com/s/3yu3qoyv3wf9j54/best_model_rst_coling2020_dali_bridging.zip?dl=0). We provide pre-trained models for ARRAU RST reported in our paper, if you need other models please contact me.
* Choose the model you want to use and copy them to the `logs/` folder.
* Modifiy the *test_path* accordingly in the `experiments.conf`:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file is a batch of sentences and must in the following format:
   
   ```
   {
  "clusters": [[[0,0],[5,5]],[[2,3],[7,8]], #Coreference
  "bridging_pairs"[[[14,15],[2,3]],....] #Bridging 
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["He", "washed", "the", "car", "yesteday","."],["How","is","the", "left", "wheel","?"]],
  "speakers": [["sp1", "sp1", "sp1", "sp1", "sp1"], ["sp1", "sp1", "sp1", "sp1", "sp1","sp1"],["sp2","sp2","sp2","sp2","sp2","sp2","sp2"]] #Optional
  }
  ```
  
  * For coreference the mentions only contain two properties \[start_index, end_index\] the indices are counted in document level and both inclusive.
  * For bridging pairs, each pair contains two mentions the first one is the anaphora and the second one is the antecedent.
* Then use `python evaluate.py config_name` to start your evaluation

## To train your own model
* You will need additionally to create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you can start training by using `python train.py config_name`
