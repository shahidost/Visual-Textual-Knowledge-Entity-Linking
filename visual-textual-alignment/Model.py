import pickle
from collections import defaultdict
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from io import BytesIO

#Upload training data
with open('.../dataset/training_data.pkl', 'rb') as i:
    training_data = pickle.load(i)

#Upload testing data
with open('.../dataset/test_data.pkl', 'rb') as i:
    testing_data = pickle.load(i)

#Calculate the number of row and columns
def dimension(data):

    #column-> 5896 = (4096+500+500+500+300)
    col = 5896

    #number of rows
    row = 0
    for img_id in data:
        img_data = data[img_id]
        for cap_id,cap_data in img_data.items():
            row = row + len(cap_data)
    return row,col

#prepared the data for model
def prepare_data(data):

    row,col = dimension(data)

    x = np.zeros((row,col), dtype=np.float32)
    y = np.zeros((row,1), dtype=np.float32)

    index = 0
    for img_id in data:
        img_data = data[img_id]
        for cap_id, cap_data in img_data.items():
            for pair in cap_data:

                #visual features consist of 4096 vgg features and 500 google-images classes
                v1 = pair[0][0]
                v2 = pair[0][1]
                v = np.hstack((v1,v2))

                #textual features consists of sub and super classes of google-images (500,500) and 300 word2vec features
                t1 = pair[1][0]
                t2 = pair[1][1]
                t = np.hstack((t1,t2))

                x[index] =np.hstack((v,t))

                #alignment pair --> 1 for true (pair) or 0 for false (non-pair)
                y[index]=pair[2]
                index+=1

    return x,y,row,col

def Model(X,y):

    # define the neural network model with three hidden layer
    model = Sequential()
    model.add(Dense(1024, input_dim=5896, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=10, batch_size=32)

    return model

def evaluation():

    #------------training-------------------#
    X_train,y_train,row,col = prepare_data(training_data)

    # Model training
    model = Model(X_train,y_train)

    # Model evaluation (training)
    loss, accuracy = model.evaluate(X_train, y_train)
    print('Training accuracy: %.2f' % (accuracy*100))

    # Model predictions
    predictions = model.predict_classes(X_train)

    # Confusion metrix
    confusion = confusion_matrix(y_train, predictions)

    # Precsion and recall
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    TP = confusion[1][1]

    print('Precision:',100*(TP/(TP+FP)))
    print('Recall',100*(TP/(TP+FN)))

    #------------Testing-------------------#
    X_test,y_test,row,col = prepare_data(testing_data)

    # Model evaluation (testing)
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Testing accuracy: %.2f' % (accuracy*100))

    # Model predictions
    predictions = model.predict_classes(X_test)

    # Confusion metrix
    confusion = confusion_matrix(y_test, predictions)

    # Precsion, recall
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    TP = confusion[1][1]

    print('Precision:',100*(TP/(TP+FP)))
    print('Recall',100*(TP/(TP+FN)))

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test = evaluation()
