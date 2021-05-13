
# coding: utf-8

# In[16]:

import unicodedata
import re
import os
import json
from tqdm import tqdm
import random
from pprint import pprint


# In[8]:

data_home = "../ori_data"


# # WebNLG

# In[9]:

# data_dir = os.path.join(data_home, "webnlg")
# train_data_path = os.path.join(data_dir, "train_data.json")
# valid_data_path = os.path.join(data_dir, "valid_data.json")
# test_data_path = os.path.join(data_dir, "test_data", "test.json")
#
#
# # In[10]:
#
# train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
# valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))
# test_data = json.load(open(test_data_path, "r", encoding = "utf-8"))


# In[19]:

# bad_samples = []
# for sample in train_data + valid_data + test_data:
#      for spo in sample["spo_list"]:
#         if spo[0] == "``" or spo[2] == "``":
#             bad_samples.append(bad_samples)
#             print(" ".join(sample["tokens"]))
#             pprint(sample["spo_list"])


# # Raw NYT

# In[2]:

nyt_data_dir = "../ori_data/nyt_raw"
nyt_train_data_path = os.path.join(nyt_data_dir, "raw_train.json")
nyt_valid_data_path = os.path.join(nyt_data_dir, "raw_valid.json")
nyt_test_data_path = os.path.join(nyt_data_dir, "raw_test.json")


# In[3]:

def get_data(path):
    with open(path, "r", encoding = "utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


# In[6]:

nyt_train_data = get_data(nyt_train_data_path)
nyt_valid_data = get_data(nyt_valid_data_path)
nyt_test_data = get_data(nyt_test_data_path)


# In[7]:

def remove_stress_mark(text):
    text = "".join([c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"])
    return text


# In[12]:

for sample in tqdm(nyt_train_data + nyt_valid_data + nyt_test_data):
    # remove stress marks
    for rel in sample["relationMentions"]:
        rel["em1Text"] = remove_stress_mark(rel["em1Text"])
        rel["em2Text"] = remove_stress_mark(rel["em2Text"])


# In[13]:

# fix a bad sample
for sample in tqdm(nyt_train_data + nyt_valid_data + nyt_test_data):
    if "XXXXXXXXXX" in sample["sentText"]:
        sample["sentText"] = re.sub("X+", "", sample["sentText"])


# In[20]:

# Output to Disk
train_data_path = os.path.join(nyt_data_dir, "train_data.json")
valid_data_path = os.path.join(nyt_data_dir, "valid_data.json")
test_data_dir = os.path.join(nyt_data_dir, "test_data")
if not os.path.exists(test_data_dir):
    os.mkdir(test_data_dir)
test_data_path = os.path.join(test_data_dir, "test_data.json")

json.dump(nyt_train_data, open(train_data_path, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(nyt_valid_data, open(valid_data_path, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(nyt_test_data, open(test_data_path, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

