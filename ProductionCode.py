#Davit Endeladze and Leo Oliver, 12/11/2019
#Code to train models for mass and formation rate, apply them to photometric dataset and produce 2D representation of the result
#Python 3.6.9
import pandas as pd #version 0.25.1
import numpy as np #version 1.17.2
import matplotlib.pyplot as plt #version 3.1.1
import seaborn as sns #version 0.9.0
from sklearn.preprocessing import StandardScaler #0.21.3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, classification_report,confusion_matrix
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


plt.style.use('ggplot') #makes matplotlib prettier


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



def formRatePlotter(df):
    """Displays plot

    Takes in dataframe
    Groups by sublcass mean 
    Plots mean formation rate for each sublcass
    Displays obtained histograms
    """
    df = df.groupby(['subcls']).mean()
    names = df.FormRate.index.values
    for i in names:
        plt.hist(df['FormRate'][i],alpha = 0.5)
    names = np.where(names=='N', 'None', names) #change sublass of N back to None
    plt.legend(names)
    plt.xlabel('Formation Rate/log(per Gigayears)')
    
    plt.show()

def dataPlotter(df,low,high,*args,label='Magnitude'):
    """Displays plot

    Takes in dataframe, minimum and maximum x limit to be displayed,label for the x-axis, column names
    For each column makes a histogram 
    Displays obtained histograms with a given label within the limits given
    """
    for col in args:
         sns.distplot(df_gal[col])
    plt.xlim(low,high)        
    plt.xlabel(label)
    plt.ylabel('Normalised Number of Sources')
    plt.legend(args)
    plt.show()

def scorer(y,y_pred):
    """Returns float

    Takes in a real vlaue and a predicted value
    Calculates a deviation from 0 of the differences of 2 inputs (perfect model would be a spike above 0)
    Returns the average deviation from 0
    """
    x = y - y_pred
    std = np.sqrt(np.mean(abs(x)**2))
    return -std #note that its negative as GridSearchCV maximasises the scorer given, whereas we want to minimise std

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

def trainRF(params,x,y,scoring,cv):
    """Returns a sklearn model

    Takes in parameters of the model to be looped over, scoring function, training features and targets and number of cross-validations to be performed
    GridSearchCV is applied to a RandomForestRegressor, with given parameters,to maximise a given scoring function
    The best model is calculated by using cross-validation on the given features and targets and saved
    Returns the trained best sklearn model
    """
    grid = GridSearchCV(RandomForestRegressor(), param_grid=params,
                     scoring=scoring, cv = cv, return_train_score = True)
    grid.fit(x, y) #fit over the predifined train features and targets
    filename = 'RF_mass_model.sav'
    pickle.dump(grid.best_estimator_, open(filename, 'wb')) #saves the best model for future use
    best_grid = grid.best_estimator_
    return best_grid

def unpickle(name):
    """Returns an unpickled file

    Takes in the name of a pickled file as a string
    Returns the file unpickled
    """
    infile = open(name,'rb')
    file= pickle.load(infile)
    infile.close()
    return file

def differenceHist(x,label='Diffirance in Mass/log(Solar Masses)'):
    """Displays a plot

    Takes in a difference between actual and predicted values
    Plots the histogram with given x-label
    """
    diff = x
    sns.kdeplot(diff, shade=True)
    plt.xlabel(label)
    plt.ylabel('Normalised Number of Sources')
    plt.show()

def compHist(x,y,label='mass/log(Solar Masses)'):
    """Displays a plot

    Takes in a actual and predicted values
    Plots the histogram with given x-label
    """
    sns.kdeplot(y, shade=True, label = "Predicted")
    sns.kdeplot(x, shade=True, label = "Actual")
    plt.ylabel('Normalised Number of Sources')
    plt.xlabel(label)
    plt.show()


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
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
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
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def pl(lbls,key,data,agg = ds.count_cat):
    """Returns image

    Takes in labels to colour in by, colour key and data from Umap
    Produces the image from data coloured according to the colour key
    Returns the image
    """
    dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1], 'Column3': lbls})
    if agg==ds.count_cat:
        dataset['Column3'] = dataset['Column3'].astype('category')
    cvs = ds.Canvas(plot_width = 900, plot_height = 500)
    aggs = cvs.points(dataset,'Column1','Column2',agg('Column3'))
    if agg==ds.count_cat:
        img = ds.transfer_functions.shade(aggs, color_key=key)
    else:
        img = ds.transfer_functions.shade(aggs, cmap = plt.cm.magma_r)
    return img

def massGroups(df):
    """Returns numpy array

    Separates mass into 3 different groups cut by std
    Returns the array 
    """
    mean = statistics.mean(df)
    sd = statistics.stdev(df)

    x = df.to_numpy()
    x = list(x)
    for i in range(len(x)):
        if x[i] > (mean - 1*sd) and x[i] < (mean + 1*sd):
            x[i] = 'a'
        elif x[i] < (mean - 1*sd):
            x[i] = 'b'
        elif x[i] > (mean + 1*sd):
            x[i] = 'c'    
    return x


def zGroups():
    """Returns numpy array

    Separates z into 3 different groups
    Returns the array 
    """
    mean_z = statistics.mean(df_gal['z'])
    sd_z = statistics.stdev(df_gal['z'])

    x = df_gal['z'].to_numpy()
    sub_z = list(x)
    for i in range(len(sub_z)):
        if sub_z[i] > 0.05 and sub_z[i] < 0.4 :
            sub_z[i] = 'a'
        elif sub_z[i] < 0.05:
            sub_z[i] = 'b'
        elif sub_z[i] > 0.4:
            sub_z[i] = 'c'   
    return sub_z


def imgPlotter(sub,ckey,data,name,agg = ds.count_cat):
    """Saves image

    Takes in groups, colour key, data from umap and the name of the file
    Saves the created image to a pickle file  
    """
    img = pl(sub,ckey,data,agg)
    file = open(name, 'wb')
    pickle.dump(img, file)
    file.close()

def normalise(ls):
    """Normalises an array

    Takes an array
    returned normalised array 
    """
    for i in range(len(ls)): #normalise the probabilities given by the sigmoid function
        total = sum(ls[i])
        for j in range(len(ls[i])):
            ls[i][j]/=total
    return ls


def extractMax(ls):
    """Returs a max entry in a list in a list of lists

    Takes an list of lists
    Returns a list with highest values in each list
    """
    ls2 = []
    for i in range(len(ls)):
        ls2.append(max(ls[i]))
    return ls2


def makeProb(ls):
    """Returs a probability dist for each subclass

    Takes an list of lists
    Returs a probability dist for each subclass
    """
    Y_1 = []
    Y_2 = []
    Y_3 = []
    for i in range(len(ls)):
        if np.argmax(ls[i])==0:
            Y_1.append(max(ls[i]))
        elif np.argmax(ls[i])==1:
            Y_2.append(max(ls[i]))
        else:
            Y_3.append(max(ls[i]))
    return Y_1,Y_2,Y_3

if __name__ == "__main__":

    df_gal = readFile('MyTable_Galaxy_Dato1998.csv') #read in the galaxy data
    df = readFile('MyTable_All_Dato1998.csv') #read in spectroscopic data

    dataPlotter(df_gal,15,27,'mag_u','mag_g','mag_r','mag_i','mag_z') #display visible magnitude bands

    dataPlotter(df_gal,4.5,18,'w1','w2','w3','w4') #display infrared magnitude bands

    dataPlotter(df_gal,7,13,'mass',label='mass/log(Solar Masses)') #display mass


    formRatePlotter(df_gal) #plot average formation rates for each subclass


    #Training the RF to predict mass

    scoring_func = make_scorer(scorer) #turn the predefined scorer into a scorer that can be used by a random forest
    train = False #set this flag to True if a model needs to be trained

    #extract features and targets for training and testing
    features_mass = extractFeatures(df_gal,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    targets_mass = df_gal['mass'].values


    train_features_mass, test_features_mass, train_targets_mass, test_targets_mass = train_test_split(
                                                                                  features_mass,targets_mass,test_size=0.2,
                                                                                  random_state=42, shuffle = True)

    if train: #check if training is required
        params = {
            'n_estimators': [100,200,300],
            'min_samples_leaf':[3,4,5],
            'max_features':[3,4,5]
        }
        regr = trainRF(params,train_features_mass,train_targets_mass,scoring_func,5)
        error = grid.cv_results_['std_train_score'][-7]
    else:
        regr = unpickle('RF_mass_model.sav') #unpickle the trained regressor
        error = 0.00027032704437985147 #precalculated by using 5 fold cross-validation

    print(regr)

    y_mass = regr.predict(test_features_mass) #predict mass using the model
    print(scorer(test_targets_mass,y_mass))
    m_score = scorer(test_targets_mass,y_mass)
    f_error = error/m_score
    print(regr.score(test_features_mass,test_targets_mass))

    y_mass_error = y_mass*f_error #error assigned to each mass measurement

    differenceHist(test_targets_mass - y_mass) #plot the difference hist

    compHist(test_targets_mass,y_mass) #plot the comparasion hist


    #DNN for subclass recognition

    features_FR = extractFeatures(df,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])
    targets_FR = extractCatTargets(df,'subcls',['N','STARBURST','STARFORMING'])


    #split data by train, validation and test
    train_features_FR, test_features_FR, train_targets_FR, test_targets_FR = train_test_split(
        features_FR,targets_FR, test_size=0.2, random_state=42, shuffle = True)
    train_features_FR, val_features_FR, train_targets_FR, val_targets_FR = train_test_split(
        train_features_FR, train_targets_FR, test_size=0.2, random_state=42, shuffle = True)

    train_2 = False #set flag to True if training is needed

    model = makeModel()

    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"]) #compile the model

    if train_2: #train if required
        mcp_save = ModelCheckpoint('.galaxy_subcls.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        history = model.fit(train_features_FR, train_targets_FR,
                  validation_data=(val_features_FR, val_targets_FR),
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


    model.load_weights(filepath=".galaxy_subcls.hdf5") #load the weights of the best model

    y_FR = model.predict(test_features_FR) # make predictions

    probabilities_FR = y_FR.copy() #save the probabilities in a variable


    y_FR = oneHotEncode(y_FR)
    conf_mx = confusion_matrix(test_targets_FR.argmax(axis=1), y_FR.argmax(axis=1)) #compute the confusion matrix

    confMatrix(conf_mx,cmap=plt.cm.gray)

    errorConfMatrix(conf_mx,cmap=plt.cm.gray)

    print(classification_report(test_targets_FR,y_FR)) #see how the classifier performed on test data
    #0 = None, 1 = Starforming, 2 = Starburst


    #Dimentionality reduction with Umap

    train_3 = False #set to True if trining is needed

    mass_groups = massGroups(df_gal['mass']) #split the mass into 3 groups
    z_groups = zGroups() #split the z into 3 groups
    FR_groups = df_gal['subcls'].to_numpy()

    X = df_gal[['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4']].to_numpy() #extract unstandardised features

    if train_3: #train the model is required and save it
        reducer = umap.UMAP()
        reducer = reducer.fit(X)
        filename = 'Umap_galaxy_model.sav'
        pickle.dump(reducer, open(filename, 'wb'))

    else: #if not load up a saved model
        reducer = unpickle('Umap_galaxy_model.sav')

    data = reducer.transform(X) #reduce 9 dimensions to 2, using the 800000 dataset

    features_prob = extractFeatures(df_gal,['mag_u','mag_g','mag_r','mag_i','mag_z','w1','w2','w3','w4'])

    probabilities_FR = model.predict(features_prob) # make predictions

    probabilities_FR = normalise(probabilities_FR)

    max_probabilities_FR = extractMax(probabilities_FR)#extract the highest probability


    ckey_mass= dict(a='yellow', b='red',c='royalblue')    #create a colour keys
    ckey_z= dict(a='yellow', b='red',c='royalblue')
    ckey_FR= dict(N='yellow', STARFORMING='royalblue',STARBURST='red')


    imgPlotter(mass_groups,ckey_mass,data,'Umap_unsupervised_mass.pkl')
    imgPlotter(z_groups,ckey_z,data,'Umap_unsupervised_z.pkl')
    imgPlotter(FR_groups,ckey_FR,data,'Umap_unsupervised_FR.pkl') #create the images and pickle it
    imgPlotter(max_probabilities_FR,ckey_FR,data,'Umap_unsupervised_FR.pkl',agg=ds.mean)


    sns.scatterplot(df_gal['z'],df_gal['mass'])
    plt.ylabel('mass/log(Solar Masses)')
    plt.show() #Display the selection bias curve


    plt.hist(max_probabilities_FR,bins=200)#total confidence of RF
    plt.xlabel('Confidence level Total')
    plt.ylabel('Count')
    plt.show()

    prob1,prob2,prob3 = makeProb(probabilities_FR)


    plt.hist(prob1,bins=200) #None
    plt.xlabel('Confidence level None')
    plt.ylabel('Count')
    plt.show()

    plt.hist(prob2,bins=200) #Starburst
    plt.xlabel('Confidence level Starburst')
    plt.ylabel('Count')
    plt.show()

    plt.hist(prob3,bins=200) #Starforming
    plt.xlabel('Confidence level Starfroming')
    plt.ylabel('Count')
    plt.show()



