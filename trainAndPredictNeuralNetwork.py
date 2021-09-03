import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def preprocess_predictors(X_df):
    # preprocess_predictors function takes a dataframe as an input and splits its features into predefined continuous and 
    # categorical features. Then, a transformation step is applied to represent all features in the range of [0, 1] 
    continuous_data = ['Bobin Tonaj', 'Kaide Tonaj', 'Kalinlik', 'Genislik']
    categorical_data1 = ['ProgNo']
    categorical_data2 = ['Kaide Sira No']

    # min-max scaling of each continuous feature to the range [0, 1]
    scaler = MinMaxScaler()
    continuousTransform = scaler.fit_transform(X_df[continuous_data])

    # one-hot encode 'ProgNo' categorical data in the range of [0, 1]
    progNoBinarizer1 = LabelBinarizer().fit(X_df[categorical_data1])
    categoricalTransform1 = progNoBinarizer1.transform(X_df[categorical_data1])

    # one-hot encode 'Kaide Sira No' categorical data in the range of [0, 1]
    progNoBinarizer2 = LabelBinarizer().fit(X_df[categorical_data2])
    categoricalTransform2 = progNoBinarizer2.transform(X_df[categorical_data2])

    # Concatenating continuous feateures with categorical feature(s)
    transformed_X_df = pd.DataFrame(np.hstack([continuousTransform, categoricalTransform1, categoricalTransform2]))

    # Sorted list of unique ProgNo key values
    uniqueProgNo = sorted(X_df["ProgNo"].value_counts().keys().tolist())

    # Sorted list of unique Kaide Sira No key values
    uniqueKaideSiraNo = sorted(X_df["Kaide Sira No"].value_counts().keys().tolist())

    transformed_X_df.columns = [['Bobin Tonaj', 'Kaide Tonaj', 'Kalinlik', 'Genislik'] + uniqueProgNo + uniqueKaideSiraNo]

    return (transformed_X_df)

def preprocess_targets(y_df):
    # preprocess_targets function takes an output vector as an input; then, applies a transformation step to represent  
    # all targets in the range of [0, 1]
    maxTime =  y_df['Tav Suresi'].max()
    scaled_y_df = y_df / maxTime
    return (scaled_y_df, maxTime)

def get_time_in_hrs(y_df):
    # get_time_in_hrs function parses 'Tav Suresi' output from 'h:mm' format into hrs. ,
    if sum(y_df['Tav Suresi'].str.contains(':')) > 0:
        temp_y_df = pd.DataFrame()
        temp_y_df_2 = pd.DataFrame()
        temp_y_df[['hrs', 'min']] = y_df['Tav Suresi'].str.split(':', expand = True)
        temp_y_df = temp_y_df.apply(pd.to_numeric)

        temp_y_df_2 = ((temp_y_df['hrs'] * 60 + temp_y_df['min']) / 60).to_frame(name='Tav Suresi')
        return temp_y_df_2
    else:
        return y_df


def trainNeuralNetwork(directory, sheetname):
    # Loading training data
    df = pd.read_excel(directory, sheetname)
    X_df = df.drop(columns=['Tav Suresi'])
    X_df.set_index(['Bobin No'], inplace = True)

    y_df = df[['Tav Suresi']]
    y_df.set_index(X_df.index, inplace = True)
    y_df = get_time_in_hrs(y_df)  

    # Preprocessing data
    X_df_transformed = preprocess_predictors(X_df)  
    global maxTavTime
    y_df_transformed, maxTavTime = preprocess_targets(y_df)
  
    # Finding the number of nodes at the input layer
    n_cols = X_df_transformed.shape[1]  

    # Instantiate the model
    layer_number = 7
    node_number = 50
    input_layer = n_cols
    global model
    model = Sequential()

    # Create an input with 4 nodes, 6 hidden layers with 25 nodes and an output layer with 1 node. 
    model.add(Dense(node_number, input_shape = (n_cols, ), activation = 'relu', kernel_initializer = 'he_normal'))
    for i in range(layer_number-2):
        model.add(Dense(node_number, activation = 'relu'))
    
    model.add(Dense(1, activation = 'linear'))

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Setting an early stopping criteria
    early_stop = EarlyStopping(monitor = 'loss', patience = 25)

    # Train your model for 60 epochs, using X_test and y_test as validation data
    model.fit(X_df_transformed, y_df_transformed, epochs = 15, validation_split = 0.2, callbacks = [early_stop])

    #model parametrelerini json formatında döndür
    weights = []

    for i in range(layer_number):
        weights.append(model.layers[i].get_weights()[0]) 

    param_out = {'NN': weights, 
                 'input_layer': input_layer,
                 'layer_number': layer_number,
                 'node_number': node_number}

    #model.save('model.h5')
    return json.dumps(param_out, cls = NumpyEncoder)

def predictNeuralNetwork(Input2dArray): 
    #class object oluşturulacak, input 2d array olarak gelecek
    #BobinNo, BobinTonaj, KaideTonaj,Kalinlik, Genislik, PrgNo, KaideSiraNo
    # BobinNo: Bobine ait unique identity'yi gösterir
    # BobinTonaj: Tekil bobine ait tonaj bilgisini içerir
    # KaideTonaj: Kaide içerisine kombinlenen bobinlerin tekil bobin ağırlıklarının toplamını gösterir
    # Kalinlik: Bobine ait kalınlık verisini içerir
    # Genislik: Bobine ait genişlik verisini içerir
    # PrgNo: Seçilen tavlama program no'sunu gösterir, bir kaide için array içindeki tüm program numaralarının aynı olması beklenir
    # KaideSiraNo: Oluşturulan bobin kombinasyonuna göre kaide içerisine atanan bobinlerin, 1. pozisyon en altta olacak şekilde
    #              bulunduğu pozisyonları içerir.            
    X_df = pd.DataFrame(Input2dArray, 
                      columns = ['Bobin No', 'Bobin Tonaj', 'Kaide Tonaj', 'Kalinlik', 'Genislik', 'ProgNo', 'Kaide Sira No'])

    X_df.set_index(['Bobin No'], inplace = True)

    # Preprocessing data
    X_df_transformed = preprocess_predictors(X_df)

    # Loading the model
    if model is None:
        model = keras.models.load_model('model.h5')
    
    # Predictions
    y_pred = model.predict(X_df_transformed) * maxTavTime   

    return y_pred