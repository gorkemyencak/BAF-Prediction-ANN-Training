import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential

#reading data
predictors = np.loadtext('predictors_data.csv', delimiter = ',')

#finding the number of nodes at the input layer
n_cols = predictors.shape[1]

#building keras model with Sequential method
#   Sequential model requires that each layer has weights and connections only to the one layer coming directly after it in the 
#   network diagram. We start adding layers using add method of the model. Dense is a standard layer type which specifies that
#   all of the nodes in the previous layer connect to all of the nodes in the current layer. In each layer, we specify the number 
#   of nodes as the first argument, and the activation function we want to use in that layer using the keyword 'activation'. In the
#   first layer, we need to specify input shapes that says the input will have n_cols columns, and there is nothing after the comma 
#   meaning that it can have any number of rows (training examples). This model has 2 hidden layers and 1 output layer.
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols, )))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))

model.summary()

#compiling model
#   we specify the optimizer and loss function when compiling the model (see tf.keras.optimizers)
#   Adam is a stochastic gradient descent based algorithm where we update a the weights by taking into account a subset (batch) of 
#   training examples (Epoch).
#   We can compute loss as 'mean_squared_error' for linear regression models while 'categorical_crossentropy' for classification
#   models.   
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#model.compile(optimizer = 'adam', loss = 'mse')

#verify that the model contains information from compiling
print('Loss function: ', model.loss)

#fitting the model (training the model after compiling it)
#   Fitting a model requires applying backpropagation to update the weights. Scaling data before fitting can ease optimization 
model.fit(predictors, target)

#predict on new data
preds = model.predict(X_test)
print(preds)

#evaluating results
print('Final loss value:', model.evaluate(X_test, y_test))






