#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from glob import glob
from itertools import chain
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(tf.test.is_gpu_available())


# In[2]:


DATA_DIR = '/home/deepcovidxr/nih_data/images/images'
image_size = 224
batch_size = 16


# In[3]:


df = pd.read_csv(f'/home/deepcovidxr/nih_data/Data_Entry_2017.csv')


# In[4]:


data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, '*.png'))}


# In[5]:


df['path'] = df['Image Index'].map(data_image_paths.get)


# In[6]:


df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))


# In[7]:


labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]


# In[8]:


for label in labels:
    if len(label) > 1:
        df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)


# In[9]:


df.head()


# In[10]:


labels = [label for label in labels if df[label].sum() > 1000]


# In[11]:


# labels


# In[12]:


train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['Finding Labels'].map(lambda x: x[:4]))


# In[13]:


train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)


# In[14]:


print(train_df.shape)
print(valid_df.shape)


# In[15]:


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def preprocess (img):
    img /= 255
    centered = np.subtract(img, imagenet_mean)
    standardized = np.divide(centered, imagenet_std)
    return standardized 
    return img


# In[16]:


core_idg = ImageDataGenerator(preprocessing_function=preprocess,
                            height_shift_range=0.05,
                            width_shift_range=0.05,
                            rotation_range=20,
                            zoom_range=0.05,
                            brightness_range=[0.9,1.1],
                            # fill_mode='constant',
                            horizontal_flip=True)

train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df,
                                                       directory=None,
                                                       x_col='path',
                                                       y_col='labels',
                                                       class_mode='categorical',
                                                       batch_size=1024,
                                                       classes=labels,
                                                       target_size=(image_size, image_size)))


# In[18]:


# # Change the model to the one that you're training

import efficientnet.tfkeras as efn 
import tensorflow.keras as keras

base_model = efn.EfficientNetB2(include_top=False, weights= 'imagenet', input_shape=(224,224,3),
                             backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
x = keras.layers.Dropout(0.125)(x)
predictions = keras.layers.Dense(len(labels), activation="sigmoid", name='last')(x)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)
# model.summary() 
print('......finish loading model')


# In[ ]:


from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(learning_rate=0.001, 
                                                                   momentum=0.9,
                                                                   nesterov=True,
                                                                   ), 
              metrics=['acc', 
                       tf.keras.metrics.AUC(name='auc'), 
                       tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall')])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

weightpath = '/home/deepcovidxr/Yunan/NIH224/saved_weights/nih_224.hdf5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=weightpath,
    verbose=1,
    monitor='val_auc',
    save_best_only=True,
    mode='max')
rlr = ReduceLROnPlateau(monitor='val_auc',
                        mode='max',
                        factor=0.1,
                        patience=2)
es = EarlyStopping(monitor='val_auc', 
                   verbose=1, 
                   patience=10, 
                   min_delta=0.001, 
                   mode='max')


# In[ ]:


model.fit(train_gen,
              steps_per_epoch=100,
              validation_data=(test_X, test_Y),
              epochs=50,
              callbacks=[checkpoint,rlr,es])


# In[ ]:




