
# coding: utf-8

# In[1]:

import os
import json
from glove import Glove
from glove import Corpus
import re


# # Data

# In[2]:

data_home = "../data4bilstm"


# In[3]:

experiment_name = "webnlg"
emb_dim = 300


# In[4]:

data_dir = os.path.join(data_home, experiment_name)
train_data_path = os.path.join(data_dir, "train_data.json")
valid_data_path = os.path.join(data_dir, "valid_data.json")
test_data_dir = os.path.join(data_dir, "test_data")
test_data_path_dict = {}
for path, folds, files in os.walk(test_data_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        file_name = re.match("(.*?)\.json", file_name).group(1)
        test_data_path_dict[file_name] = file_path


# In[5]:

train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))
test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding = "utf-8"))


# In[6]:

all_data = train_data + valid_data
for data in list(test_data_dict.values()):
    all_data.extend(data)
    
corpus = [sample["text"].split(" ") for sample in all_data]
len(corpus)


# # Glove

# In[7]:

def train_glove_emb(corpus, window = 10, emb_dim = 100, learning_rate = 0.05, epochs = 10, thr_workers = 6):
    corpus_model = Corpus()
    corpus_model.fit(corpus, window = window)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    
    glove = Glove(no_components = emb_dim, learning_rate = learning_rate)
    glove.fit(corpus_model.matrix, 
              epochs = epochs,
              no_threads = thr_workers, 
              verbose = True)
    glove.add_dictionary(corpus_model.dictionary)
    return glove


# In[8]:

# glove
golve = train_glove_emb(corpus, emb_dim = emb_dim)


# In[9]:

# save
save_path = os.path.join("../pretrained_word_emb", "glove_{}_{}.emb".format(emb_dim, experiment_name))
golve.save(save_path)


# In[10]:

golve.most_similar('university', number = 10)


# In[11]:

golve.word_vectors.shape


# In[12]:

# Quick Start

# # get similar words
# golve.most_similar('Massachusetts', number = 10)

# # emb matrix shape
# golve.word_vectors.shape

# # get id 
# golve.dictionary['Virginia']

# # # 指定词条词向量
# # glove.word_vectors[glove.dictionary['university']]

# # save
# save_path = os.path.join(data_home, "pretrained_word_embeddings", "glove_100.emb")
# glove.save(save_path)

# # load
# glove = Glove()
# glove = glove.load(save_path)

