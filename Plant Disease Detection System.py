#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns


# # data processing

# # training image processing

# In[2]:


training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# # valdation image processing

# In[4]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[5]:


training_set


# In[6]:


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# # building model

# In[7]:


from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D, Flatten,Dropout
from tensorflow.keras.models import Sequential


# In[8]:


model = Sequential()


# In[9]:


# BUILDING CONVOLUTION LAYER 
#to boost our training speed we remove padding from every convolution layer 


# In[10]:


model.add(Conv2D(filters = 32,kernel_size = 3, padding = 'same', activation = 'relu', input_shape = [128,128,3]))
model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))


# In[11]:


model.add(Conv2D(filters=64,kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters= 64, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))


# In[12]:


model.add(Conv2D(filters=128,kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters= 128, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))


# In[13]:


model.add(Conv2D(filters=256,kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters= 256, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))


# In[14]:


model.add(Conv2D(filters=512,kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters= 512, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))


# In[15]:


model.add(Dropout(0.25)) # avoid overfitting 


# In[16]:


model.add(Flatten())


# In[17]:


model.add(Dense(units = 1024, activation = 'relu'))


# In[18]:


model.add(Dropout(0.4))


# In[19]:


#output layer
model.add(Dense(units = 38 , activation = 'softmax'))


# # compling model

# In[20]:


model.compile(optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[21]:


model.summary()


# # Model Training

# In[22]:


training_history = model.fit(x = training_set, validation_data = validation_set, epochs = 10)


# In[ ]:





# # Model Evaluation

# In[23]:


#model evluation on Training set 
train_loss,train_acc = model.evaluate(training_set)


# In[24]:


print(train_loss,train_acc)


# In[25]:


#model on validation set 
val_loss,val_acc = model.evaluate(validation_set)


# In[26]:


print(val_loss,val_acc)


# # SAVING MODEL

# In[28]:


#model.save("trained_model.h5")


# In[71]:


#saving trained moodel
model.save("trained_model.keras")


# In[39]:


training_history.history


# In[40]:


#Recording history in json
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)


# In[51]:


training_history.history['val_accuracy']


# # Accuracy Visualization

# In[33]:


epochs = (i for i in range (1,11))
epochs


# In[62]:


epochs = range(1,11)

#training_accuracy = training_history.history['accuracy']
#validation_accuracy = training_history.history['val_accuracy']

plt.plot(epochs, training_accuracy, color='red', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, color='blue', label='Validation Accuracy')

plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Visualization of Accuracy Results")
plt.legend()
plt.show()


# # some others metrics for model evaluation

# In[1]:


class_name = training_set.class_names
class_name


# In[73]:


test_set = validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[75]:


y_pred = model.predict(test_set)
y_pred,y_pred.shape


# In[77]:


predict_categories = tf.argmax(y_pred,axis = 1)


# In[78]:


predict_categories


# In[80]:


true_categories = tf.concat([y for x ,y in test_set], axis = 0)
true_categories


# In[81]:


y_true = tf.argmax(true_categories,axis = 1)
y_true


# In[ ]:





# ![image.png](attachment:image.png)

# In[85]:


from sklearn.metrics import classification_report,confusion_matrix


# In[84]:


print(classification_report(y_true,predict_categories, target_names = class_name))


# In[88]:


cm = confusion_matrix(y_true,predict_categories)
cm


# # confusion matrix visualozation

# In[97]:


plt.figure(figsize = (40,40))
sns.heatmap(cm,annot = True, annot_kws = {'size':10})
plt.xlabel("Predicted class", fontsize = 20)
plt.ylabel("Actual class", fontsize = 20)
plt.title("plant disease prediction confusion matrix", fontsize = 25)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




