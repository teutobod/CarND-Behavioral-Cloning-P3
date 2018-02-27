from keras.models import Model
import matplotlib.pyplot as plt
import pickle

history = pickle.load(open("train_history.p", "rb" ))

### print the keys contained in the history object
print(history.keys())

### plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
