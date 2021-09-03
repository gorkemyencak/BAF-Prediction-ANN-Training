import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# *** Functions ***
def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


# Instantiate the model
model = Sequential()
model.add(Dense(16, input_shape = (64,), activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs = 60, validation_data = (X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Checking whether the model you build benefits from more data
initial_weights = model.get_weights()
training_sizes = np.array([ 125,  502,  879, 1255])
early_stop = EarlyStopping(monitor = 'loss', patience = 3)
train_accs = []
test_accs = []

for train_size in training_sizes:
    # Split a fraction according to training size
    X_train_frac, X_test, y_train_frac, y_test = train_test_split(X, y, train_size = train_size, random_state = 42)
    # Set model initial weights
    model.set_weights(initial_weights)
    # Fit model on the training set fraction
    model.fit(X_train_frac, y_train_frac, epochs = 100, validation_data = (X_test, y_test)), callbacks = [early_stop])
    # Get accuracy for training set fraction
    train_accs.append(model.evaluate(X_train_frac, y_train_frac, verbose = 0)[1])
    # Get accuracy for test set fraction
    test_accs.append(model.evaluate(X_test, y_test, verbose = 0)[1])

    print("Done with size: ", train_size)

# Plot train vs test accuracies
plot_results(train_accs, test_accs)