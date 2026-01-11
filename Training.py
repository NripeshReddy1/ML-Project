from sklearn.datasets import load_files       
from keras.utils import to_categorical
import numpy as np
from glob import glob

tar=5
path='./dataset/'
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets =to_categorical(np.array(data['target']), tar)
    return files, targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset(path)

test_files=train_files
test_targets = train_targets


burn_classes = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
# print statistics about the dataset
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total  images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))



for file in train_files: assert('.DS_Store' not in file)



from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to \4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 


# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 

import matplotlib.pyplot as plt

img_width, img_height = 224, 224
batch_size = 4
epoch=20


img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 3
lr = 0.0005
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import callbacks
import time
from keras.layers import SeparableConv2D, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# First Convolution Block
model.add(SeparableConv2D(nb_filters1, (conv1_size, conv1_size), padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# First Separable Convolution Layer
model.add(SeparableConv2D(nb_filters2, (conv2_size, conv2_size), padding='same'))
model.add(Activation("relu"))
'''

# Second Separable Convolution Layer
model.add(SeparableConv2D(nb_filters2, (conv2_size, conv2_size), padding='same'))
model.add(Activation("relu"))

'''

# Pooling after Separable Convolution layers
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(tar, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

hist=model.fit(train_tensors, train_targets ,validation_split=0.3, epochs=15, batch_size=10)

# Print the accuracy after training
train_acc = hist.history['accuracy'][-1]
#val_acc = hist.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {train_acc:.4f}")
#print(f"Final Validation Accuracy: {val_acc:.4f}")


show_history_graph(hist)


#model.save('color_trained_modelDNN.h5')
model.save('trained_model_DNN1.h5')


##
##import numpy as np
##from sklearn.datasets import load_files
##from keras.utils import np_utils
##from glob import glob
##from tensorflow.keras.preprocessing import image
##from tqdm import tqdm
##import matplotlib.pyplot as plt
##from tensorflow.keras.applications import VGG16
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import Flatten, Dense, Dropout
##from tensorflow.keras.callbacks import EarlyStopping
##from tensorflow.keras.preprocessing.image import ImageDataGenerator
##from tensorflow.keras import optimizers
##from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
##
### Set the number of classes and path
##tar = 5
##path = './dataset/'
##
### Function to load dataset
##def load_dataset(path):
##    data = load_files(path)
##    files = np.array(data['filenames'])
##    targets = np_utils.to_categorical(np.array(data['target']), tar)
##    return files, targets
##
### Load datasets
##train_files, train_targets = load_dataset(path)
##test_files = train_files
##test_targets = train_targets
##
### Check for .DS_Store files
##for file in train_files:
##    assert('.DS_Store' not in file)
##
### Function to convert image paths to tensors
##def path_to_tensor(img_path, width=224, height=224):
##    img = image.load_img(img_path, target_size=(width, height))
##    x = image.img_to_array(img)
##    return np.expand_dims(x, axis=0)
##
##def paths_to_tensor(img_paths, width=224, height=224):
##    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
##    return np.vstack(list_of_tensors)
##
### Pre-process the data for Keras
##train_tensors = paths_to_tensor(train_files).astype('float32') / 255
##
### Split train_tensors and train_targets into training and validation sets
##from sklearn.model_selection import train_test_split
##train_tensors, val_tensors, train_targets, val_targets = train_test_split(train_tensors, train_targets, test_size=0.3, random_state=42)
##
### Data augmentation
##train_datagen = ImageDataGenerator(
##    rotation_range=20,
##    width_shift_range=0.2,
##    height_shift_range=0.2,
##    shear_range=0.2,
##    zoom_range=0.2,
##    horizontal_flip=True,
##    fill_mode='nearest'
##)
##
### Pre-trained model
##base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
##base_model.trainable = False  # Freeze the base model
##
### Create the model
##model = Sequential()
##model.add(base_model)
##model.add(Flatten())
##model.add(Dense(256, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(tar, activation='softmax'))
##
##### Compile the model
####model.compile(loss='categorical_crossentropy', 
####              optimizer=optimizers.RMSprop(learning_rate=0.0002), 
####              metrics=['accuracy'])
####
##### Set up early stopping
####early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
####
##### Fit the model using augmented data
####history = model.fit(train_datagen.flow(train_tensors, train_targets, batch_size=10),
####                    validation_data=(val_tensors, val_targets), 
####                    epochs=20, 
####                    callbacks=[early_stopping])
##
##
##### Fine-tuning: Unfreeze the last few layers of the base model
####for layer in base_model.layers[-4:]:
####    layer.trainable = True
### Set up early stopping
##early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
##
### Set up reduce learning rate on plateau
##reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
##
##
##for layer in base_model.layers[-4:]:
##    layer.trainable = True
##
### Compile the model again after unfreezing layers
##model.compile(loss='categorical_crossentropy',
##              optimizer=optimizers.RMSprop(learning_rate=1e-5),  # Lower learning rate for fine-tuning
##              metrics=['accuracy'])
##
### Fit the model again
##history_fine = model.fit(train_datagen.flow(train_tensors, train_targets, batch_size=10),
##                          validation_data=(val_tensors, val_targets),
##                          epochs=20,
##                          callbacks=[early_stopping, reduce_lr])
##
##
##
##
### Function to show training history
##def show_history_graph(history):
##    plt.figure(figsize=(12, 4))
##    
##    # Accuracy
##    plt.subplot(1, 2, 1)
##    plt.plot(history.history['accuracy'], label='train')
##    plt.plot(history.history['val_accuracy'], label='validation')
##    plt.title('Model Accuracy')
##    plt.ylabel('Accuracy')
##    plt.xlabel('Epoch')
##    plt.legend(loc='upper left')
##    
##    # Loss
##    plt.subplot(1, 2, 2)
##    plt.plot(history.history['loss'], label='train')
##    plt.plot(history.history['val_loss'], label='validation')
##    plt.title('Model Loss')
##    plt.ylabel('Loss')
##    plt.xlabel('Epoch')
##    plt.legend(loc='upper left')
##    
##    plt.show()
##
### Show training history
####show_history_graph(history)
### Show training history for fine-tuning
##show_history_graph(history_fine)
### Save the model
##model.save('trained_model_DNN.h5')
##





##
##import numpy as np
##from sklearn.datasets import load_files
##from keras.utils import np_utils
##from glob import glob
##from tensorflow.keras.preprocessing import image
##from tqdm import tqdm
##import matplotlib.pyplot as plt
##from tensorflow.keras.applications import VGG16
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import Flatten, Dense, Dropout
##from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
##from tensorflow.keras.preprocessing.image import ImageDataGenerator
##from tensorflow.keras import optimizers
##from sklearn.model_selection import train_test_split
##
### Set the number of classes and path
##tar = 5
##path = './dataset/'
##
### Function to load dataset
##def load_dataset(path):
##    data = load_files(path)
##    #print(data['target'])
##    files = np.array(data['filenames'])
##    targets = np_utils.to_categorical(np.array(data['target']), tar)
##    return files, targets
##
### Load datasets
##train_files, train_targets = load_dataset(path)
##test_files = train_files
##test_targets = train_targets
##
### Check for .DS_Store files
##for file in train_files:
##    assert('.DS_Store' not in file)
##
### Function to convert image paths to tensors
##def path_to_tensor(img_path, width=224, height=224):
##    img = image.load_img(img_path, target_size=(width, height))
##    x = image.img_to_array(img)
##    return np.expand_dims(x, axis=0)
##
##def paths_to_tensor(img_paths, width=224, height=224):
##    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
##    return np.vstack(list_of_tensors)
##
##
### Pre-process the data for Keras
##train_tensors = paths_to_tensor(train_files).astype('float32') / 255
##
### Split train_tensors and train_targets into training and validation sets
##train_tensors, val_tensors, train_targets, val_targets = train_test_split(train_tensors, train_targets, test_size=0.3, random_state=42)
##
##
### Data augmentation
##train_datagen = ImageDataGenerator(
##    rotation_range=40,              
##    width_shift_range=0.2,         
##    height_shift_range=0.2,        
##    shear_range=0.2,               
##    zoom_range=0.2,                
##    horizontal_flip=True,          
##    fill_mode='nearest',           
##    brightness_range=[0.8, 1.2],   
##    channel_shift_range=30.0       
##)
### Pre-trained model
##base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
##base_model.trainable = False  # Freeze the base model
##
### Create the model
##model = Sequential()
##model.add(base_model)
##model.add(Flatten())
##model.add(Dense(256, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(tar, activation='softmax'))
##
##
##base_model.trainable = True
##for layer in base_model.layers[:-2]:
##    layer.trainable = False
##
##
##
### Compile the model
##model.compile(loss='categorical_crossentropy', 
##              optimizer=optimizers.RMSprop(learning_rate=0.0005), 
##              metrics=['accuracy'])
##
### Set up early stopping and reduce learning rate
##early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
####reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
##reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
##
### Fit the model using augmented data
##history = model.fit(train_datagen.flow(train_tensors, train_targets, batch_size=256),
##                    validation_data=(val_tensors, val_targets), 
##                    epochs=20, 
##                    callbacks=[early_stopping, reduce_lr])
##
##
####history = model.fit(train_datagen.flow(train_tensors, train_targets, batch_size=256),
####                    validation_data=(val_tensors, val_targets), 
####                    epochs=20, 
####                    callbacks=[early_stopping, reduce_lr])
##
### Function to show training history
##def show_history_graph(history):
##    plt.figure(figsize=(12, 4))
##    
##    # Accuracy
##    plt.subplot(1, 2, 1)
##    plt.plot(history.history['accuracy'], label='train')
##    plt.plot(history.history['val_accuracy'], label='validation')
##    plt.title('Model Accuracy')
##    plt.ylabel('Accuracy')
##    plt.xlabel('Epoch')
##    plt.legend(loc='upper left')
##    
##    # Loss
##    plt.subplot(1, 2, 2)
##    plt.plot(history.history['loss'], label='tran')
##    plt.plot(history.history['val_loss'], label='validation')
##    plt.title('Model Loss')
##    plt.ylabel('Loss')
##    plt.xlabel('Epoch')
##    plt.legend(loc='upper left')
##    
##    plt.show()
##
### Show training history
##show_history_graph(history)
##
### Save the model
##model.save('trained_model_DNN.h5')

