import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import statistics
import datashader as ds
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pickle
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.python.keras import metrics #1.9.0
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.models import Sequential
import statistics
import pickle
import umap #0.3.9
import datashader as ds #0.8.0

plt.style.use('ggplot')

def pl(lbls,key,data):
    dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1], 'Column3': lbls})
    dataset['Column3'] = dataset['Column3'].astype('category')
    cvs = ds.Canvas(plot_width = 900, plot_height = 500)
    aggs = cvs.points(dataset,'Column1','Column2',ds.count_cat('Column3'))
    img = ds.transfer_functions.shade(aggs, color_key=key)
    return img


def readFile(name):
    df = pd.read_csv(name)
    df = df.fillna('N')
    df['subcls'] = df['subcls'].replace({'None': 'N'})
    for col in ['mag_u','mag_g','mag_r','mag_i','mag_z']:
        df=df.where(df[col] >-1000)
    df = df.dropna()
    for col in ['w1','w2','w3','w4']:
        df=df.where(df[col] <1000)
    df = df.dropna()
    if name == 'MyTable_Galaxy_Dato1998.csv':
        df=df.where(df['cls'] == 'GALAXY')
        df = df.dropna()
    return df

def unpickle(name):
    infile = open(name,'rb')
    file= pickle.load(infile)
    infile.close()
    return file


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
    model.add(Dense(3, activation = 'sigmoid'))
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



def extractFeatures(df,ls):
    """Returns numpy matrix

    Takes in a dataframe and a list of feature columns
    applies the StandardScaler to the features 
    Returns the scaled features as an array of numpy arrays
    """
    scaler = StandardScaler()

    features = scaler.fit_transform(df[ls])
    return features



def massGroups():
    """Returns numpy array

    Separates mass into 3 different groups cut by std
    Returns the array 
    """
    mean = statistics.mean(Y_M)
    sd = statistics.stdev(Y_M)

    x = Y_M
    x = list(x)
    for i in range(len(x)):
        if x[i] > (mean - 1*sd) and x[i] < (mean + 1*sd):
            x[i] = 'a'
        elif x[i] < (mean - 1*sd):
            x[i] = 'b'
        elif x[i] > (mean + 1*sd):
            x[i] = 'c'    
    return x



def imgPlotter(sub,ckey,data,name,agg = ds.count_cat):
    """Saves image

    Takes in groups, colour key, data from umap and the name of the file
    Saves the created image to a pickle file  
    """
    img = pl(sub,ckey,data,agg)
    file = open(name, 'wb')
    pickle.dump(img, file)
    file.close()

if __name__ == "__main__":
    df= unpickle('SDSS_allphoto_111M.csv_classified.pkl')


    X = df[['psf_u_corr','psf_g_corr','psf_r_corr','psf_i_corr','psf_z_corr','w1','w2','w3','w4']].to_numpy()
    #Y = df['class_pred'].to_numpy()


    regr = unpickle('RF_mass_model.sav')


    reducer = unpickle('Umap_class_model.sav')


    model = makeModel()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"]) #compile the model
    model.load_weights(filepath=".galaxy_subcls_u.hdf5") #load the weights of the best model


    umap = unpickle('Umap_galaxy_model.sav')


    X_STD = extractFeatures(df,['psf_u_corr','psf_g_corr','psf_r_corr','psf_i_corr','psf_z_corr','w1','w2','w3','w4'])



    Y_FR = model.predict(X_STD)
    Y_FR = oneHotEncode(Y_FR)
    Y_M = regr.predict(X_STD)



    mass_groups = massGroups()


    ckey_mass= dict(a='yellow', b='red',c='royalblue')
    ckey_FR= dict(N='yellow', STARFORMING='royalblue',STARBURST='red')



    X_chunk = np.array_split(X,6)



    data =[]
    for i in range(len(X_chunk)):
        data_chunks = umap.transform(X_chunk[i].reshape(-1, 9))
        data.append(data_chunks)



    data = np.concatenate([data[0],data[1],data[2],data[3],data[4],data[5]])


    FR_group = []
    for i in range(len(Y_FR)):
        if np.argmax(Y_FR) == 0:
            FR_group.append('N')
        elif np.argmax(Y_FR) == 1:
            FR_group.append('STARFORMING')
        else:
            FR_group.append('STARBURST')
    dfr = pd.Series(FR_Group)
    dfr = dfr.astype(category)
    FR = dfr.values


    imgPlotter(mass_groups,ckey_mass,data,'Umap_mass.pkl')
    imgPlotter(FR_group,ckey_FR,data,'Umap_FR.pkl')

