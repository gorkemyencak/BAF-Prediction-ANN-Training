import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import json

# Training a Neural Network 
# Our Neural Network will have 3 layers; an input, a hidden and an output layer. It has 6 input features namely; 
#   * Bobin Tonajı
#   * Kaide Tonajı 
#   * Kalınlık
#   * Genişlik 
#   * Program No 
#   * Kaide Sıra No

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




# *** Functions ***
def plot_loss(loss, val_loss):
  # plot_loss function takes loss and validation loss values and helps us to draw a linegraph w.r.t. # of epochs
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  #plt.yscale('log')
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

 
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
  transformed_X_df = np.hstack([continuousTransform, categoricalTransform1, categoricalTransform2])

  # create a list of minmaxscaler tupple
  global minmaxlist
  minmaxlist = list(zip(X_df[continuous_data].min(axis=0), X_df[continuous_data].max(axis=0)))

  return (transformed_X_df)


def preprocess_test_predictors(X_df):
  continuous_data = ['Bobin Tonaj', 'Kaide Tonaj', 'Kalinlik', 'Genislik']
  continuousNumpyArray = X_df[continuous_data].to_numpy()

  categorical_data = ['ProgNo', 'Kaide Sira No']
  progNoBinarizer = LabelBinarizer().fit(X_df[categorical_data])
  categoricalTransform = progNoBinarizer.transform(X_df[categorical_data])

  # Concatenating continuous feateures with categorical feature(s)
  transformed_X_df = np.hstack([continuousNumpyArray, categoricalTransform])
  return (transformed_X_df)


def preprocess_targets(y_df):
  # preprocess_targets function takes an output vector as an input; then, applies a transformation step to represent all targets 
  # in the range of [0, 1]
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


  '''
  if ':' in y_df[['Tav Suresi']]:
    hrs, min = y_df[['Tav Suresi']].split(':')
    print(hrs)
    print(min)
    return (int(hrs) * 60 + int(min)) / 60
  else:
    return y_df[['Tav Suresi']]
    '''


# Loading training data
df = pd.read_excel(r'C:\Users\goyencak\Documents\GitHub\BAF_model\BAF_training_data.xlsx', sheet_name='Sheet2')
X_df = df.drop(columns=['Tav Suresi'])
X_df.set_index(['Bobin No'], inplace = True)

y_df = df[['Tav Suresi']]
y_df.set_index(X_df.index, inplace = True)
y_df = get_time_in_hrs(y_df)

#print(y_df)

# Sorted list of unique ProgNo key values
uniqueProgNo = sorted(X_df["ProgNo"].value_counts().keys().tolist())
print(uniqueProgNo)

# Sorted list of unique Kaide Sira No key values
uniqueKaideSiraNo = sorted(X_df["Kaide Sira No"].value_counts().keys().tolist())
print(uniqueKaideSiraNo)

# *************************
print(X_df.head(10))
print(y_df.head(10))


# Preprocessing data
X_df_transformed_npArray = preprocess_predictors(X_df)
print(minmaxlist)
X_df_transformed = pd.DataFrame(X_df_transformed_npArray)

uniqueProgNo = list(map(lambda i: 'ProgNo_' + str(i), uniqueProgNo))
uniqueKaideSiraNo = list(map(lambda i: 'KaideSiraNo_' + str(i), uniqueKaideSiraNo))

#print(uniqueProgNo)
#print(uniqueKaideSiraNo)

X_df_transformed.columns = [['BobinTonaj', 'KaideTonaj', 'Kalinlik', 'Genislik'] + uniqueProgNo + uniqueKaideSiraNo]

'''
X_df_transformed.columns = X_df_transformed.columns.map(lambda i : 'ProgNo_' + i  if i != 'Bobin Tonaj' and
                                                                                     i != 'Kaide Tonaj' and
                                                                                     i != 'Kalinlik' and
                                                                                     i != 'Genislik' else i)
'''

#print(X_df_transformed[0:15])


y_df_transformed, maxTavTime = preprocess_targets(y_df)
print(maxTavTime)
print('\n')
print(y_df_transformed[0:3])

# Finding the number of nodes at the input layer
n_cols = X_df_transformed.shape[1]


#print(X_df_transformed.head())


# Model Parameters  # Best nb_epoch = 80, patience = 30
layer_number = 5   # Best => 5
node_number = 250  # Best => 250
input_layer = n_cols

# Instantiate the model
model = Sequential()

# Create an input with 4 nodes, 6 hidden layers with 25 nodes and an output layer with 1 node. 
model.add(Dense(node_number, input_shape = (n_cols, ), activation = 'relu', kernel_initializer = 'he_normal', use_bias = False))

for i in range(layer_number-2):
  model.add(Dense(node_number, activation = 'relu', use_bias = False))

model.add(Dense(1, activation = 'linear', use_bias = False))

model.summary()

# Compile the model
#   we specify the optimizer and loss function when compiling the model (see tf.keras.optimizers)
#   Adam is a stochastic gradient descent based algorithm where we update a the weights by taking into account a subset (batch) of 
#   training examples (Epoch).
#   We can compute loss as 'mean_squared_error' for linear regression models while 'categorical_crossentropy' for classification
#   models. 
#opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')#, metrics = ['accuracy']) #'adam' , 'rmsprop'

# Verify that the model contains information from compiling
print('Loss function: ', model.loss)

# Test if your model is well assembled by predicting before training
#print(model.predict(X_df))

# Obtaining initial weights
#initial_weights = model.get_weights()

# Setting an early stopping criteria
early_stop = EarlyStopping(monitor = 'loss', patience = 30)

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_df_transformed, y_df_transformed, epochs = 80, validation_split = 0.2, callbacks = [early_stop])  #, verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])


# Get results

weights = []
#print(len(model.layers))
for i in range(layer_number):
  #print(model.layers[i].get_weights()[0])
  weights.append(model.layers[i].get_weights()[0])  

def flatten(t):
    return [item for sublist in t for item in sublist]



param_out = {'NN': weights, 
             'input_layer': input_layer,
             'layer_number': layer_number,
             'node_number': node_number,
             'columns': flatten(X_df_transformed.columns.values.tolist()),
             'scaler': minmaxlist,
             'maxtavtime': maxTavTime}
 
print(json.dumps(param_out, cls = NumpyEncoder))

with open('NNParams.txt', 'w') as f:
  f.write(json.dumps(param_out, cls = NumpyEncoder)) 
  f.close()


# Prediction based of test samples
transformed_X_df = preprocess_predictors(pd.DataFrame(X_df))

"""
#test
model = keras.models.load_model('model.h5')
print('test123123')
"""

y_pred = model.predict(transformed_X_df) * maxTavTime


print('\n\n Transformed Predictor Matrix')
print(transformed_X_df[0:10])
print('\n Transformed Output Vector')
print(y_df[0:10])
print('\n Predictions')
print(y_pred[0:10])
print('\n Tav Time vs Pred: Differences')
print((y_df-y_pred)[0:50])


# Error of NN model
error = mean_squared_error(y_df, y_pred)
print('\n MSE: %.3f' % error)
error_abs_perc = mean_absolute_percentage_error(y_df, y_pred)
print('\n MAPE: %.3f' % error_abs_perc)

#print('\n Error_Abs_Perc: ', (abs(y_df-y_pred)/y_df)[0:50])
print('\n Max error: \n', (y_df-y_pred).max())
print('\n Max observation: ', y_df.max())
print('\n Max prediction: ', y_pred.max())

model.save('model.h5')



