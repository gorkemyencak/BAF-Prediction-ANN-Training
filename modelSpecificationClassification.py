import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import load_model

data = pd.read_csv('basketbol_shot_log.csv')
predictors = data.drop(['shot_result'], axis = 1).as_matrix()

# to_categorical is a utility function to convert the data from one column to multiple columns
target = to_categorical(data.shot_result)

n_cols = predictors.shape[1]

# Output layer has a separate node for each possible outcome and uses 'softmax' activation that ensures the prediction sum to 1 
# so they can be interpreted as probabilities. 'softmax' activation is generally used for multi-class classification problems.
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols, )))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# We add metrics = ['accuracy'] that enables us to print out the accuracy score at the end of each Epoch. We can use 
# 'binary_crossentropy' as loss function when we use 'sigmoid' activation function for our output node. 
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

model.fit(predictors, target)


# Saving, reloading and using your Model to make predictions
model.save('model_file.h5')
my_model = load_model('my_model.h5')
predictions = my_model.predict(predictors_new)
probability_true = predictions[:, 1]
