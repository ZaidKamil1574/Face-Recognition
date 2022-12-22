#!/usr/bin/env python
# coding: utf-8

# In[23]:


from deepface import DeepFace
import matplotlib.pyplot as plt 
import pandas as pd


# In[24]:


img1_path = 'deepface/deepface/tests/dataset/img6.jpg'
img2_path = 'deepface/deepface/tests/dataset/img9.jpg'


# In[25]:


img1 = DeepFace.detectFace(img1_path)
img2 = DeepFace.detectFace(img2_path)


# In[26]:


plt.imshow(img1)


# In[27]:


plt.imshow(img2)


# In[28]:


model_name = 'Facenet' #using Facenet model


# In[29]:


resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name) 


# In[30]:


resp #false image pair is same person. Facial recognition task 


# In[31]:


model_name = 'ArcFace' #using ArcFace model


# In[32]:


resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name) #image pair as input


# In[33]:


resp #true image pair is same person. Facial verify function have O(1) time complexity


# In[34]:


df = DeepFace.find(img_path = 'source.jpg', db_path = 'deepface/deepface/tests/dataset')


# In[35]:


df.head #find function have On time complexity. where n is the number of instances in database


# In[36]:


DeepFace.analyze(img_path = img1_path) #input single image then make prediction for emotion, race


# In[37]:


resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name)


# In[38]:


resp


# In[ ]:




