import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def Get_dataset():

    # Read csv data
    df = pd.read_csv('./dataset/icml_face_data.csv/icml_face_data.csv')

    # Initialize features and result lists
    X = []
    y = []

    image_size = 48
    training_data = []

    # Iterate over data and add each item to its specific list
    for i in tqdm(range(len(df[' pixels']))):

        # clear the useless data from the dataset and fix the format
        image = np.reshape(np.array(df[' pixels'][i].split(' ')).astype(int), (image_size, image_size))

        # get the labels of the dataset
        label = df['emotion'][i]
        
        # remove the disgust emotion for the dataset and fix the format
        if label == 1:
            pass
        else:
            if label != 0:
                training_data.append([image, label - 1])  
            else:
                training_data.append([image, label]) 
    # shuffle the data
    random.shuffle(training_data)

    # seperate the data to X and Y
    for features,label in tqdm(training_data):
        X.append(features)
        y.append(label)

    # reshape the features array
    X = np.array(X).reshape(-1, image_size, image_size, 1)

    # normalize the data
    X = X / 255.0

    # convert class vectors to binary class matrices
    y = to_categorical(y, 6)

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

def Show_plots(history, epochs):

    # visualizing losses and accuracy
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure()
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), val_loss)
    plt.title('Loss')
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend({'Test Data','Training Data'})

    plt.figure()
    plt.plot(range(epochs), train_acc)
    plt.plot(range(epochs), val_acc)
    plt.title('Accuracy')
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy') 
    plt.legend({'Test Data','Training Data'})