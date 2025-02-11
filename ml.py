#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd


# In[108]:


train = pd.read_csv("./data/train.tsv", sep="\t", index_col=0)
test = pd.read_csv("./data/test.tsv", sep="\t", index_col=0)
sample_submit = pd.read_csv("./data/sample_submit.tsv", sep="\t", index_col=0, header=None)


# In[109]:


train["survived"].count()
train["survived"].value_counts()


# In[102]:


test.head()


# In[119]:


train = train[["survived", "sibsp", "parch", "fare"]]
test = test[["sibsp", "parch", "fare"]]


# In[120]:


y = train["survived"]
X = train.drop(["survived"], axis=1)


# In[121]:


X


# In[124]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)


# In[138]:


pred = model.predict_log_proba(test)[:, 1]
print(pred[:5])


# In[166]:


sample_submit[1] = pred
sample_submit.to_csv("submit.tsv", header=None, sep="\t")

