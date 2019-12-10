#!/usr/bin/env python
# coding: utf-8

# In[458]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.python.keras import metrics
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.python.keras.layers import Input, Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout,LSTM,Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
import pickle
import statistics
plt.style.use('ggplot')


def readFile(name):
    """Returns DataFrame

    Takes in a path of a csv file as a string
    Cleans the dataset
    Returns clean pandas DataFrame
    """
    df = pd.read_csv(name)
    df = df.fillna('N')
    df['subcls'] = df['subcls'].replace({'None': 'N'}) #None can be interpreted as a NaN object and not as a string, convert them to N
    for col in ['mag_u','mag_g','mag_r','mag_i','mag_z']:
        df=df.where(df[col] >-1000)
    df = df.dropna()
    for col in ['w1','w2','w3','w4']:
        df=df.where(df[col] <1000)
    df = df.dropna()
    
    df=df.where(df['cls'] == 'GALAXY')
    df = df.dropna()
    return df

def extractFeatures(df,ls):
    """Returns numpy matrix

    Takes in a dataframe and a list of feature columns
    applies the StandardScaler to the features 
    Returns the scaled features as an array of numpy arrays
    """
    scaler = StandardScaler()

    features = scaler.fit_transform(df[ls])
    return features

def extractCatTargets(df,col,ls):
    """Returns numpy matrix

    Takes in a dataframe,categorical column and a list of feature columns
    Returns categorical numpy array
    """
    df[col] = pd.Categorical(df[col])
    dfDummies = pd.get_dummies(df[col])
    arr = dfDummies[ls].values
    return arr


def makeModel():
    """Returs model

    Creates a model with 3 deep layers and sigmoid activation at the final layer
    Returns the model
    """
    
    model = Sequential([
        Dense(512, input_shape=(9,)),
        Activation('relu'),

        Dense(512),
        Activation('relu'),

        Dense(124),
        Activation('relu'),

        ])
    model.add(Dense(3, activation = 'softmax'))
    return model

def oneHotEncode(x):
    """Returs array

    Takes in an array of logits
    Returns a one hot encoded array
    """

    for i in range(len(x)):
        x[i][np.argmax(x[i])] = 1
        for j in range(len(x[i])):
            if x[i][j] != 1:
                x[i][j]=0
    return x


def confMatrix(matrix,cmap =plt.cm.plasma ):
    """Displays numpy matrix

    Takes in a confusion matrix
    Nomalises the matrix
    Displays the normalised matrix with a chosen cmap
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_conf_mx = matrix/row_sums
    plt.matshow(norm_conf_mx, cmap=cmap)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.colorbar()
    plt.show()


def errorConfMatrix(matrix,cmap =plt.cm.plasma):
    """Displays numpy matrix

    Takes in a confusion matrix
    Nomalises the matrix
    Sets diagonal elements to 0 so only misclassified elements are displayed
    Displays the normalised matrix with a chosen cmap
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_conf_mx = matrix/row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=cmap)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    df = readFile('MyTable_All_Dato1998.csv') #read in spectroscopic data


    df.groupby(['subcls']).size()


    df_n = df[df['subcls']=='N']
    df_sb = df[df['subcls']=='STARBURST']
    df_sf = df[df['subcls']=='STARFORMING']


    df_t1=df_n[::60]
    df_t2=df_sb[::2]
    df_t3=df_sf[::8]


    df_balanced = pd.concat([df_t1,df_t2,df_t3])

    df_balanced.groupby(['subcls']).size()

    df_unbalanced=pd.concat([df,df_balanced]).drop_duplicates(keep=False)


    df_unbalanced.groupby(['subcls']).size()


    features_FR_b = extractFeatures(df_balanced,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    targets_FR_b = extractCatTargets(df_balanced,'subcls',['N','STARFORMING','STARBURST'])
    features_FR_u = extractFeatures(df_unbalanced,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    targets_FR_u = extractCatTargets(df_unbalanced,'subcls',['N','STARFORMING','STARBURST'])

    features_FR_u, test_set_features, targets_FR_u, test_set_targets = train_test_split(
        features_FR_u,targets_FR_u, test_size=0.2, random_state=42, shuffle = True)


    train_features_FR_u, test_features_FR_u, train_targets_FR_u, test_targets_FR_u = train_test_split(
        features_FR_u,targets_FR_u, test_size=0.2, random_state=42, shuffle = True)

    train_features_FR_u, val_features_FR_u, train_targets_FR_u, val_targets_FR_u = train_test_split(
        train_features_FR_u, train_targets_FR_u, test_size=0.2, random_state=42, shuffle = True)


    #split data by train, validation and test
    train_features_FR_b, test_features_FR_b, train_targets_FR_b, test_targets_FR_b = train_test_split(
        features_FR_b,targets_FR_b, test_size=0.2, random_state=42, shuffle = True)

    train_features_FR_b, val_features_FR_b, train_targets_FR_b, val_targets_FR_b = train_test_split(
        train_features_FR_b, train_targets_FR_b, test_size=0.2, random_state=42, shuffle = True)



    model = makeModel()
    model.summary()


    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"]) #compile the model

    mcp_save = ModelCheckpoint('.galaxy_subcls_u.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(train_features_FR_u, train_targets_FR_u,
              validation_data=(val_features_FR_u, val_targets_FR_u),
              batch_size=50000,
              epochs=250,
              callbacks=[mcp_save]
                   )

    epochs = range(1, len(history.history["loss"])+1)
    plt.figure(1, figsize=(8,4))
    plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
    plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("CCE Loss");
    plt.legend()

    plt.show()

    mcp_save = ModelCheckpoint('.galaxy_subcls_b.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(train_features_FR_b, train_targets_FR_b,
              validation_data=(val_features_FR_b, val_targets_FR_b),
              batch_size=50000,
              epochs=250,
              callbacks=[mcp_save]
                   )

    epochs = range(1, len(history.history["loss"])+1)
    plt.figure(1, figsize=(8,4))
    plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
    plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("CCE Loss");
    plt.legend()

    plt.show()

    model.load_weights(filepath=".galaxy_subcls_u.hdf5") #load the weights of the best model

    y_FR_u = model.predict(test_features_FR_u) # make predictions


    probabilities_FR_u = y_FR_u.copy() #save the probabilities in a variable

    y_FR_u = oneHotEncode(y_FR_u)


    conf_mx_u = confusion_matrix(test_targets_FR_u.argmax(axis=1), y_FR_u.argmax(axis=1)) #compute the confusion matrix

    confMatrix(conf_mx_u, cmap=plt.cm.gray)

    errorConfMatrix(conf_mx_u,cmap=plt.cm.gray)


    print(classification_report(test_targets_FR_u,y_FR_u))


    print(classification_report(test_set_targets,oneHotEncode(model.predict(test_set_features))))


    model.load_weights(filepath=".galaxy_subcls_b.hdf5") #load the weights of the best model



    y_FR_b = model.predict(test_features_FR_b) # make predictions


    probabilities_FR_b = y_FR_b.copy() #save the probabilities in a variable


    y_FR_b = oneHotEncode(y_FR_b)



    conf_mx_b = confusion_matrix(test_targets_FR_b.argmax(axis=1), y_FR_b.argmax(axis=1)) #compute the confusion matrix


    confMatrix(conf_mx_b)


    errorConfMatrix(conf_mx_b)


    print(classification_report(test_targets_FR_b,y_FR_b)) #see how the classifier performed on test data



    print(classification_report(test_set_targets,oneHotEncode(model.predict(test_set_features))))


    df_test_features = pd.DataFrame(test_set_features,columns=['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    df_test_targets = pd.DataFrame(test_set_targets,columns = ['N','STARFORMING','STARBURST'])


    df_test = pd.concat([df_test_features,df_test_targets],axis=1,sort=False)



    df_test_1 = df_test[df_test['mag_r']<-1]
    df_test_2 = df_test[df_test['mag_r']>-1]
    df_test_2 = df_test_2[df_test_2['mag_r']<0]
    df_test_3 = df_test[df_test['mag_r']>0]
    df_test_3 = df_test_3[df_test_3['mag_r']<1]
    df_test_4 = df_test[df_test['mag_r']>1]

    test_1_features = extractFeatures(df_test_1,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    test_1_targets = df_test_1[['N','STARFORMING','STARBURST']].values
    test_2_features = extractFeatures(df_test_2,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    test_2_targets = df_test_2[['N','STARFORMING','STARBURST']].values
    test_3_features = extractFeatures(df_test_3,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    test_3_targets = df_test_3[['N','STARFORMING','STARBURST']].values
    test_4_features = extractFeatures(df_test_4,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    test_4_targets = df_test_4[['N','STARFORMING','STARBURST']].values

    print(classification_report(test_1_targets,oneHotEncode(model.predict(test_1_features))))

    print(classification_report(test_2_targets,oneHotEncode(model.predict(test_2_features))))

    print(classification_report(test_3_targets,oneHotEncode(model.predict(test_3_features))))

    print(classification_report(test_4_targets,oneHotEncode(model.predict(test_4_features))))


    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.65,0.8,0.54,0.5])
    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.16,0.17,0.03,0.01])
    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.09,0.14,0.01,0.00])
    plt.legend(['N','STARFORMING','STARBURST'])
    plt.ylabel('f1 score')
    plt.xlabel('r band Magnitude')
    plt.show()

    model.load_weights(filepath=".galaxy_subcls_u.hdf5") #load the weights of the best model

    print(classification_report(test_1_targets,oneHotEncode(model.predict(test_1_features))))


    print(classification_report(test_2_targets,oneHotEncode(model.predict(test_2_features))))


    print(classification_report(test_3_targets,oneHotEncode(model.predict(test_3_features))))


    print(classification_report(test_4_targets,oneHotEncode(model.predict(test_4_features))))


    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.90,0.92,0.95,0.92])
    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.23,0.35,0.13,0.08])
    plt.plot(['m < \u03BC - \u03C3','\u03BC - \u03C3 < m < \u03BC','\u03BC < m < \u03BC + \u03C3','\u03BC + \u03C3 < m'],[0.24,0.55,0.20,0.03])
    plt.legend(['N','STARFORMING','STARBURST'])
    plt.ylabel('f1 score')
    plt.xlabel('r band Magnitude')
    plt.show()





