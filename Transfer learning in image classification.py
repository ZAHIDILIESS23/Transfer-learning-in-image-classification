#!/usr/bin/env python
# coding: utf-8

# ## Transfer learning in image classification

# In this notebook we will use transfer learning and take pre-trained model from google's Tensorflow Hub and re-train that on flowers dataset. Using pre-trained model saves lot of time and computational budget for new classification problem at hand

# In[1]:


get_ipython().system(' pip install tensorflow_hub')


# In[2]:


import numpy as np
import cv2

import PIL.Image as Image
import os

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ## Make predictions using ready made model (without any training)

# In[5]:


IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])


# In[9]:


gold_fish = Image.open("goldfish.jpg").resize(IMAGE_SHAPE)
gold_fish


# In[10]:


gold_fish = np.array(gold_fish)/255.0
gold_fish.shape


# In[11]:


gold_fish[np.newaxis, ...]


# In[12]:


result = classifier.predict(gold_fish[np.newaxis, ...])
result.shape


# In[13]:


predicted_label_index = np.argmax(result)
predicted_label_index


# In[15]:


tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')


# ## Load flowers dataset

# In[19]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it


# In[20]:


data_dir


# In[21]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[22]:


list(data_dir.glob('*/*.jpg'))[:5]


# In[23]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[24]:


roses = list(data_dir.glob('roses/*'))
roses[:5]


# In[26]:


import PIL
PIL.Image.open(str(roses[1]))


# In[27]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# ## Read flowers images from disk into numpy array using opencv

# In[28]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[29]:


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[30]:



flowers_images_dict['roses'][:5]


# In[31]:


str(flowers_images_dict['roses'][0])


# In[32]:



img = cv2.imread(str(flowers_images_dict['roses'][0]))


# In[33]:


img.shape


# In[34]:


cv2.resize(img,(224,224)).shape


# In[35]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[36]:



X = np.array(X)
y = np.array(y)


# ## Train test split

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ## Preprocessing: scale images

# In[39]:


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# ## Make prediction using pre-trained model on new flowers dataset

# In[40]:


X[0].shape


# In[41]:


IMAGE_SHAPE+(3,)


# In[42]:


x0_resized = cv2.resize(X[0], IMAGE_SHAPE)
x1_resized = cv2.resize(X[1], IMAGE_SHAPE)
x2_resized = cv2.resize(X[2], IMAGE_SHAPE)


# In[43]:


plt.axis('off')
plt.imshow(X[0])


# In[44]:


predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized]))
predicted = np.argmax(predicted, axis=1)
predicted


# ## Now take pre-trained model and retrain it using flowers images

# In[46]:



feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# In[47]:


num_of_flowers = 5

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)
])

model.summary()


# In[48]:



model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)


# In[ ]:




