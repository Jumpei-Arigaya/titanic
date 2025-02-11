#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[28]:


train = pd.read_csv("./data/train.tsv", sep="\t", index_col=0)
test = pd.read_csv("./data/test.tsv", sep="\t", index_col=0)
sample_submit = pd.read_csv("./data/sample_submit.tsv", sep="\t", index_col=0, header=None)


# In[34]:


train.head()

