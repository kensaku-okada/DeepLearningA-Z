#convolutional neural network (CNN)
#building the VNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the CNN
classifier = Sequential()

#1 add convolution layers
#since we are using tensorflow backend, filters = number of feature detectors, kernel_size = the size of feature detectors, input_shape = the size of the input variables (input pictures)
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=(64, 64, 3), activation = 'relu'))

#2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#####option: add another convolution and pooling layer for better performance. the code runs without this part.
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
###########################################################################################

#3 flattening
classifier.add(Flatten())

#4 full connection
classifier.add(Dense(units=128, activation= 'relu'))
#since the output node is binary (cat or dog) we use sigmoid, not softmax
classifier.add(Dense(units=1, activation= 'sigmoid'))

#compile the CNN
#since the function corresponds to logarithmic loss (common in CNN) and we have a binary outcome, we use binary_cross_entropy instead of categorical_corss_entropy
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

###########################################################
#fit the CNN to the images (preprocess the images to prevent overfitting) start 
###########################################################
#the whole code reference is at https://keras.io/preprocessing/image/

from keras.preprocessing.image import ImageDataGenerator

#image augumentation preparation for training dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#image augumentation for test dataset, which is also normalizing the data
test_datagen = ImageDataGenerator(rescale=1./255)

#create the training set
#target_size should be same as the input shape defined at classifier.add.
#this function also resized the training dataset
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#create the test set
#this function also resized the test dataset
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#fit the parameters (feature detector and the weights at the ANN) with the training set and test the trained model with the test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)

#for more improvement, you can revise number of epochs, layers, feature detectors, input_shape

###########################################################
#fit the CNN to the images (preprocess the images to prevent overfitting) end
###########################################################

###########################################################
#make new single predictions start
###########################################################
import numpy as np
from keras.preprocessing import image

test_image = image.load_img(path='dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
#maek the image 3 arrays (black&white to color)
test_image_3d = image.img_to_array(test_image)
#add another dimenstion for batch size
test_image_3d = np.expand_dims(test_image_3d, axis = 0)
#run prediction
result = classifier.predict(test_image_3d)
#get the labels to each output variable
labels = training_set.class_indices
labels = {v:k for k, v in labels.items()}
if result[0][0] == 1:
#    dog
    prediction = labels[1]
else:
#    cat
    prediction = labels[0]
print(prediction)
###########################################################
#make new single predictions end
###########################################################

