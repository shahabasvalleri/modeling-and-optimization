#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# In[9]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D


from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img


# In[10]:


print(os.listdir('/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion'))


# In[11]:


Facial_train='/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/train'


# In[12]:


Facial_test='/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/test'


# In[13]:


emotion_names= sorted(os.listdir(Facial_train))
print(emotion_names)


# In[14]:


emotion_names2= sorted(os.listdir(Facial_test))
print(emotion_names2)


# In[15]:


batch_size = 64
target_size = (48,48)

Face_train_datagen = ImageDataGenerator(rescale=1./255)
Face_test_datagen   = ImageDataGenerator(rescale=1./255)

Face_train_generator = Face_train_datagen.flow_from_directory(
       Facial_train,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

Face_test_generator = Face_test_datagen.flow_from_directory(
        Facial_test,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


# In[16]:


#Data Preparation
input_shape = (48,48,1) # img_rows, img_colums, color_channels
number_of_classes = 7


# In[17]:


print(os.listdir('/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/train'))


# In[18]:


#Dividing all data class variable

def plot_images(image_directory, top=10):
    Total_picture_directoris = os.listdir(image_directory)
    picture_files = [os.path.join(image_directory, file) for file in Total_picture_directoris][:5]
  
    plt.figure(figsize=(12, 12))
  
    for idx, image_path in enumerate(picture_files):
        plt.subplot(5, 5, idx+1)
        img = plt.imread(image_path)
        plt.tight_layout()         
        plt.imshow(img, cmap='plasma')


# In[19]:


print('Angry Pictures ')
print()
plot_images(Facial_train+'/angry')


# In[20]:


print('Disgusted Pictures ')
print()
plot_images(Facial_train+'/disgusted')


# In[21]:


print('Fear Pictures')
plot_images(Facial_train+'/fearful')
print()


# In[22]:


print('Haapy Pictures')
plot_images(Facial_train+'/happy')
print()


# In[23]:


print('Neutral Pictures')
plot_images(Facial_train+'/neutral')
print()


# In[24]:


print('Sad Pictures')
plot_images(Facial_train+'/sad')
print()


# In[25]:


print('Train and Test Surprised Pictures')
plot_images(Facial_train+'/surprised')
print()
#print('Surprised Pictures')
plot_images(Facial_test+'/surprised')
print()


# In[26]:


#Data Visuallisation
expressions = os.listdir('/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/train')
for emotion in expressions:
    count = len(os.listdir(f'/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/train/{emotion}'))
    print(f'{emotion} faces={count}')


# In[27]:


expressions = os.listdir('/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/test')
for emotion in expressions:
    count = len(os.listdir(f'/Users/Shahabas/Documents/CW Facial EXPRESSION/Emotion/Facial_emotion/test/{emotion}'))
    print(f'{emotion} faces={count}')


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

expressions = os.listdir('C:/Users/Shahabas/Downloads/Emotion/Facial_emotion/train')
Label_values = [len(os.listdir(f'C:/Users/Shahabas/Downloads/Emotion/Facial_emotion/train/{emotion}')) for emotion in expressions]
fig = plt.figure(figsize = (10, 5))
sns.barplot(expressions, Label_values ,palette = 'plasma')
plt.xlabel("Emotions",fontsize = 16)
plt.ylabel("Number of images",fontsize = 16)
plt.title("Number of pictures of each category", fontsize = 16)
plt.show()


# In[29]:


expressions = os.listdir('C:/Users/Shahabas/Downloads/Emotion/Facial_emotion/test')
Label_values2  = [len(os.listdir(f'C:/Users/Shahabas/Downloads/Emotion/Facial_emotion/test/{emotion}')) for emotion in expressions]
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
sns.barplot(expressions, Label_values2 ,palette = 'rocket')
plt.xlabel("Emotions",fontsize = 16)
plt.ylabel("Number of images",fontsize = 16)
plt.title("Number of pictures of each category", fontsize = 16)
plt.show()


# In[30]:


#Model Building
#CNN

import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
model= tf.keras.models.Sequential()
model.add(Conv2D(32,(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(512,(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512,(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten()) 
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
    
model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(7, activation='softmax'))
model.summary()


# In[31]:


Face_train_datagen = ImageDataGenerator(width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         validation_split = 0.2
                                        )
Face_test_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)


# In[32]:


img_size=48


# In[33]:


Face_train_generator = Face_train_datagen.flow_from_directory(directory = Facial_train,
                                                    target_size = (img_size,img_size),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                    subset = "training"
                                                   )
Face_test_generator = Face_test_datagen.flow_from_directory( directory = Facial_test,
                                                              target_size = (img_size,img_size),
                                                              batch_size = 64,
                                                              color_mode = "grayscale",
                                                              class_mode = "categorical",
                                                              subset = "validation"
                                                             )


# In[34]:


from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
model.compile(
    optimizer = Adam(lr=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
  )


# In[35]:


epochs = 10
batch_size = 64


# In[ ]:


history = model.fit(x = Face_train_generator,epochs = epochs,validation_data = Face_test_generator)


# In[ ]:


cnn_score = model.evaluate_generator(Face_test_generator) 
print('Test loss: ', cnn_score[0])
print('Test accuracy: ', cnn_score[1]*100)


# In[ ]:


y_pred = model.predict(Face_test_generator)


# In[ ]:


figure, axs = plt.subplots(1, 2, figsize=(20, 7))
plt.subplot(1, 2, 1)
axs = axs.ravel()
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
axs[0].plot(epochs, loss, 'b', label='loss_train',linewidth=3)
axs[0].plot(epochs, loss_val, 'r', label='loss_val',linewidth=3)
axs[0].set_title('Test accuracy loss')

axs[0].set_ylabel('Test accuracy loss ')
axs[0].legend()
axs[0].grid()
acc = history.history['accuracy']
acc_val = history.history['val_accuracy']
axs[1].plot(epochs, acc, 'b', label='accuracy_train',linewidth=3)
axs[1].plot(epochs, acc_val, 'r', label='accuracy_val',linewidth=3)
axs[1].set_title('Accuracy')

axs[1].set_ylabel('Test accuracy')
axs[1].legend()
axs[1].grid()
plt.show()


# In[ ]:


#ANN

model_classifier = Sequential()
model_classifier.add(Dense(128, input_shape=input_shape, activation='relu'))
model_classifier.add(Dropout(0.4))
model_classifier.add(Dense(64, activation='relu'))
model_classifier.add(Dropout(0.6))
model_classifier.add(Flatten())
model_classifier.add(Dense(7, activation='softmax'))
model_classifier.summary()


# In[68]:


model_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[69]:


num_epochs = 10

x_train = Face_train_generator.n//Face_train_generator.batch_size
y_test   = Face_test_generator.n//Face_test_generator.batch_size


# In[70]:


model_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[71]:


history = model_classifier.fit(x=Face_train_generator, steps_per_epoch=x_train, epochs=num_epochs, batch_size=batch_size, validation_data=Face_test_generator, validation_steps=y_test)
history


# In[639]:


ANN_score = model_classifier.evaluate_generator(Face_test_generator) 
print('Test loss: ', ANN_score[0])
print('Test accuracy: ', ANN_score[1]*100)


# In[72]:


num_epochs = 10
x_train = Face_train_generator.n//Face_train_generator.batch_size
y_test   = Face_test_generator.n//Face_test_generator.batch_size


# In[73]:


from keras.models import Model, Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Bidirectional


# In[74]:


#LSTM

model_LSTM = Sequential()
model_LSTM.add(LSTM(256, return_sequences=True, input_shape=(4,189)))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(128, return_sequences=True))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(64, return_sequences=True))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(64))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(units=1))

model_LSTM.add(Dense(10, activation='softmax'))
model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_LSTM.summary()


# In[75]:


history = model_LSTM.fit(x=Face_train_generator, steps_per_epoch=x_train, epochs=num_epochs, batch_size=batch_size, validation_data=Face_test_generator, validation_steps=y_test)


# In[36]:


Score = model_ann.evaluate_generator(Face_test_generator) 
print('Test loss: ', Score[0])
print('Test accuracy: ', Score[1]*100)


# In[35]:


figure, axs = plt.subplots(1, 2, figsize=(10, 7))
plt.subplot(1, 2, 1)
axs = axs.ravel()
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
axs[0].plot(epochs, loss, 'b', label='loss_train',linewidth=3)
axs[0].plot(epochs, loss_val, 'r', label='loss_val',linewidth=3)
axs[0].set_title('Test accuracy loss')

axs[0].set_ylabel('Test accuracy loss ')
axs[0].legend()
axs[0].grid()
acc = history.history['accuracy']
acc_val = history.history['val_accuracy']
axs[1].plot(epochs, acc, 'b', label='accuracy_train',linewidth=3)
axs[1].plot(epochs, acc_val, 'r', label='accuracy_val',linewidth=3)
axs[1].set_title('Accuracy')

axs[1].set_ylabel('Test accuracy')
axs[1].legend()
axs[1].grid()
plt.show()


# In[34]:


numberof_pictures = 1
plt.figure(figsize=(19, 5))
for i in range(numberof_pictures):
    #ax = plt.subplot(1, numberof_pictures, i+1)
    plot_images(Facial_test+'/surprised')
    plot_images(Facial_train+'/surprised')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

print()


# In[ ]:




