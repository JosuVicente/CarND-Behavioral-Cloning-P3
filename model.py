import csv
import cv2
import numpy as np
import sklearn
import math
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

##### Models ######    
def get_initial_model():
    ### INITIAL
    model = Sequential()    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def get_lenet_model():
    ### LENET
    model = Sequential()        
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model
    
def get_nvidia_model():
    ### NVIDIA MODEL
    model = Sequential() 
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop image to only see section with road
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    
def get_comma_ai_model():
    ### COMMA.AI MODEL    
    model = Sequential()  
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model

##### Training ######
def train(sources, model_type, correction, epochs):    
    lines = []
    for source in sources:                
        isFirstLine = True
        with open(source) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if (isFirstLine == False):
                    lines.append(line)
                else:
                    isFirstLine = False

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    #steers = []
    
    def generator(samples, batch_size=32):
        num_samples = len(samples)
        #bln_store_images = False
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                measurements = []
                
                for batch_sample in batch_samples:
                    #Process each image (center, left, right) and apply correction
                    for i in range(3):
                        source_path = batch_sample[i]
                        filename = source_path.split('/')[-1].split('\\')[-1]
                        current_path = 'data/IMG/' + filename
                        image = cv2.imread(current_path)                        
                        images.append(image)

                        adjustment = float(line[3])
                        if (i==1):
                            adjustment += correction
                        elif (i==2):
                            adjustment -= correction
                        measurement = adjustment
                        measurements.append(measurement)
                        
                        #Flip horizontally to augment dataset
                        images.append(cv2.flip(image,1))
                        measurements.append(measurement*-1.0) 

                
                #steers.extend(measurements)
                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)    


    # choose model 
    if (model_type == 'initial'):
        model = get_initial_model()
    elif (model_type == 'lenet'):
        model = get_lenet_model()
    elif (model_type == 'nvidia'):
        model = get_nvidia_model()
    elif (model_type == 'comma_ai'):
        model = get_comma_ai_model()
        
    model.compile(loss='mse', optimizer='adam')
        
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=epochs)

    model.save('model.h5')
    
    model.summary()

    ### plot the training and validation loss for each epoch
    import matplotlib.pyplot as plt
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    

sources = []
#sources.append('data/loop1.csv')
#sources.append('data/loop4.csv')
#sources.append('data/loops2.csv')
#sources.append('data/loop3.csv')
#sources.append('data/reverse.csv')
#sources.append('data/sides.csv')
sources.append('data/driving_log.csv')
train(sources, 'nvidia', 0.08, 1)
