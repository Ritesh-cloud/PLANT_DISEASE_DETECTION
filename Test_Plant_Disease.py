#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import numpy as np  

import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# # loading Model

# In[2]:


model = tf.keras.models.load_model('trained_model.h5')


# In[3]:


model.summary()


# In[ ]:





# # visualizing single image of test set

# In[1]:


import cv2
image_path = r"C:\Users\rites\Downloads\New Plant Diseases Dataset(Augmented)\test\TomatoYellowCurlVirus1.JPG"
#reading image 
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # convert BGR image to RGB

#displaying image 
plt.imshow(img)
plt.title("test iamge")
plt.xticks([])
plt.yticks([])
plt.show()


# # Testing Model

# In[5]:


image = tf.keras.preprocessing.image.load_img(image_path,target_size = (128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) #convert isingle image to a batch 
print(input_arr.shape)


# In[6]:


prediction = model.predict(input_arr)
prediction,prediction.shape


# In[7]:


result_index = np.argmax(prediction)
result_index


# In[8]:


class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# In[10]:


#Displaying Result of Disease Prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()


# In[11]:


model_prediction


# In[ ]:





# In[ ]:




