import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# Remark!! We can make experimentation with different architectures such as:
#   * More layers
#   * Fewer layers
#   * Layers with more nodes
#   * Layers with fewer nodes

# Remark2!! Model capacity represents neural network complexity. Start with a simple neural network and then gradually increase:
#   * Number of layers 
#   * Number of nodes within an existing layer
# While trying those parameters iteratively, you should observe which model gives the best validation score. Try to avoid overfitting
# by decreasing the model capacity.


def get_new_model(input_shape = input_shape):
    # Creates a model with 2 hidden layers of 100 nodes and 1 output layer with 2 output nodes including the bias weights
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape = input_shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    model.summary()
    return model

#learning rates to test
lr_to_test = [.000001, 0.01, 1]

#loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr)
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # In deep learning, commonly used validation is split rather than cross-validation due to the size of data. Below, 30% of data 
    # will be used for validation. Our goal is to have best validation score possible, so we should keep training while validation
    # scores are improving, and then stop training when validation score is not improving. We do this with 'early stopping'. We then
    # create an 'early_stopping_monitor' before fitting the model. That monitor takes an argument, patience, which shows how many 
    # Epochs the model can go without improving before we stop training. By default, keras trains for 10 epochs; however, we can 
    # specify with 'nb_epoch' argument otherwise. 
    early_stopping_monitor = EarlyStopping(patience = 3)

    model.fit(predictors, target, 
              validation_split = 0.3, nb_epoch = 20,
              callbacks = [early_stopping_monitor])


#creating a new model with 3 hidden layers
model_2 = Sequential()
model_2.add(Dense(10, activation = 'relu', input_shape = input_shape))
model_2.add(Dense(10, activation = 'relu'))
model_2.add(Dense(10, activation = 'relu'))
model_2.add(Dense(2, activation = 'softmax'))

#compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#compile model_1
model_1 = model
model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#fit model_1
early_stopping_monitor = EarlyStopping(patience = 2) #EarlyStopping can also take 'monitor' passed as an argument 
model_1_training = model_1.fit(predictors, target, 
                               validation_split = 0.2, nb_epoch = 15, 
                               callbacks = [early_stopping_monitor], verbose = False)

#fit model_2
model_2_training = model_2.fit(predictors, target, 
                               validation_split = 0.2, nb_epoch = 15, 
                               callbacks = [early_stopping_monitor], verbose = False)

#create the plot
# By accessing history attribute, we can check the saved metrics of the model at each epoch during training as an array of numbers
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
