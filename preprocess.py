#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import numpy as np
import shutil


# In[23]:


source1 = "images/knife/"
dest11 = "images/examples/"
files = os.listdir(source1)
for f in files:
    if np.random.rand(1) <= 0.2:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)


# In[24]:


source2 = "images/scissors/"
files = os.listdir(source2)
for f in files:
    if np.random.rand(1) <= 0.2:
        shutil.move(source2 + '/'+ f, dest11 + '/'+ f)


# In[ ]:




