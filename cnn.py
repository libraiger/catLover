import tensorflow as tf

# Step 1: Initialising the CNN
model = tf.keras.models.Sequential()

# Step 2: ADDING CONVOLUTION LAYER
# 32 feature detectors with 3*3 dimensions so the convolution layer compose of 32 feature maps
# 128 by 128 dimensions with colored image(3 channels)  (tensorflow backend)
input_size = (128, 128)
model.add(tf.keras.layers.Convolution2D(32, 3, 3, input_shape = (*input_size, 3), activation = 'relu'))

# STEP 3: ADDING POOLING LAYER
# reduce the size of feature maps and therefore reduce the number of nodes in the future fully connected layer (reduce time complexity, less compute intense without losing the performace). 2 by 2 deminsion is the recommended option
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

# STEP 4: ADDING SECOND CONVOLUTION LAYER WITH POOLIING
model.add(tf.keras.layers.Convolution2D(32, 3, 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

# STEP 5: ADDING FLATTENING LAYER
# flatten all the feature maps in the pooling layer into single vector
model.add(tf.keras.layers.Flatten())

# STEP 6: ADDING A FULLY CONNECTED LAYER
# making classic ann which compose of fully connected layers
# number of nodes in hidden layer (output_dim) (common practice is to take the power of 2)
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# STEP 7: COMPILING THE MODEL
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# STEP 8: FITTING THE CNN TO THE IMAGES
# image augmentation technique to enrich our dataset(training set) without adding more images so get good performance  results with little or no overfitting even with the small amount of images
# used from keras documentation (flow_from_directory method)

from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
# image augmentation part
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# create training set
# wanna get higher accuracy -> inccrease target_size
training_set = train_datagen.flow_from_directory('/content/CNN-for-image-Classification/dataset/training_set',
                                                 target_size = input_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

# create test set
# wanna get higher accuracy -> inccrease target_size
test_set = test_datagen.flow_from_directory('/content/CNN-for-image-Classification/dataset/test_set',
                                            target_size = input_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

# fit the cnn model to the trainig set and testing it on the test set
model.fit(training_set,
          steps_per_epoch = 8000/batch_size,
          epochs = 35,
          validation_data = test_set,
          validation_steps = 2000/batch_size)

# STEP 9: MAKING NEW PREDICTIONS
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/content/CNN-for-image-Classification/dataset/single_prediction/cat_or_dog_4.jpg', target_size= input_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)