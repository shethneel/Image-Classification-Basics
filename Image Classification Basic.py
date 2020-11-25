#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import numpy as np


# In[3]:


filename = '/Users/neelsheth/Downloads/Unknown.jpg'


# # Image loading techniques

# Loading Image : Method 1

# In[4]:


from IPython.display import Image
Image(filename = '/Users/neelsheth/Downloads/Unknown.jpg', width = 224, height = 224)


# Loading Image : Method 2

# In[5]:


from tensorflow.keras.preprocessing import image
img = image.load_img (filename, target_size = (224,224))


# To print image im method 2 is we have to use matplotlib

# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.imshow(img)


# Loading image using PILLOW : Method 3

# In[8]:


from PIL import Image


# In[9]:


imgg = Image.open(filename)


# In[10]:


imgg = imgg.resize ((224,224))


# In[11]:


plt.imshow(imgg)


# # Let's load deep learning model

# Mobilenet is one of the best deep learning architecture in tensorfloe for image classification so we'll use it

# In[12]:


mobile = tf.keras.applications.mobilenet.MobileNet()


# Now we have four phases
# 1. Create Model
# 2. Train Model
# 3. Test Model
# 4. Prediction

# Firstly we have to pre-process the image cause we can't use it as it is

# We'll select any one method of image loading from above to work further
# 
# I'll select the method with tensorflow

# In[13]:


from tensorflow.keras.preprocessing import image
img = image.load_img (filename, target_size = (224,224))


# In[14]:


plt.imshow(img)


# In[15]:


resized_img = image.img_to_array(img)


# In[16]:


resized_img.shape


# To use deep learning model, we need four dimenssion but here we have only three, so we'll bring 4th dimenssion using Numpy

# In[17]:


final_img = np.expand_dims(resized_img, axis = 0)


# In[18]:


final_img.shape


# Boom!!! We have 4 dimenssion now......

# Now let's pre-process the 4 dimenssion image

# In[19]:


final_img = tf.keras.applications.mobilenet.preprocess_input(final_img)


# Pre-processed data is fitted to Model
# 
# Now let's do the predicitons

# In[20]:


pred = mobile.predict(final_img)


# now if we print pred directly then we'll get a big confusing stuff
# 
# still if you want then you can try : print (pred)

# so to check some accurate result, we'll use imagenet_utils lib from keras

# In[21]:


from tensorflow.keras.applications import imagenet_utils


# In[22]:


results = imagenet_utils.decode_predictions(pred)


# In[23]:


print (results)


# In[24]:


plt.imshow(img)


# SO AS YOU SAW THAT THESE ARE THE TOP 5 PREDICTIONS FROM MOBILENET WHICH HAVE ACCURACY OF 80%
# 
# BUT WE HAVE IT'S ANOTHER MORE ACCURATE VERSION MOBILENETV_2

# # Let's again load more accurate model

# In[25]:


mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()


# In[26]:


pred = mobile.predict(final_img)


# In[27]:


results = imagenet_utils.decode_predictions(pred)


# In[28]:


print (results)


# # IN THIS CASE, MODEL - 1 IS MORE ACCURATE THAN VERSION 2
