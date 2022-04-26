import matplotlib.pyplot as plt
import numpy as np

from ModelUtils import *
from DataPoint import *
from CountNet import *
import pickle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
	

class CountNet:

	def __init__(self, file_to_save):
		self.OutputPath = file_to_save
		self.fig_path = mkDir(os.path.join(self.OutputPath,'RESULTS/FIGURES'))
		self.result_path = mkDir(os.path.join(self.OutputPath,'RESULTS/TestData'))
		print(self.fig_path, self.result_path)
		segnet_model = Sequential()

		pretrained_model= tf.keras.applications.ResNet50(include_top=False,
		                   input_shape=(128,128,3),
		                   pooling='avg',
		                   weights='imagenet')

		for layer in pretrained_model.layers[:-1]:
		    layer.trainable=True

		pretrained_model = Model(pretrained_model.input, pretrained_model.layers[-2].output)
		segnet_model.add(pretrained_model)
		segnet_model.add(Conv1D(filters=18, kernel_size=1 ,strides=1,kernel_initializer= 'uniform', activation= 'relu'))
		segnet_model.add(BatchNormalization())
		segnet_model.add(Dense(128, activation='relu'))
		segnet_model.add(Conv1D(6, kernel_size=1, activation="relu"))
		segnet_model.add(BatchNormalization())
		segnet_model.add(Dense(128, activation='relu'))
		segnet_model.add(Flatten())
		segnet_model.add(Dense(36, activation='relu'))
		segnet_model.add(Dense(1, activation='linear'))
		segnet_model.summary()

		self.network = segnet_model

	def setData(self, path_to_data, split = [.7,.15,.15], batch_size = 64):
		
		data = getData(path_to_data)
		images, lables = splitData(data)

		full_dataset = tf.data.Dataset.from_tensor_slices((images, lables))
		DATASET_SIZE = len(data)
		BATCH_SIZE = batch_size

		train_size = int(split[0] * DATASET_SIZE)
		val_size = int(split[1] * DATASET_SIZE)
		test_size = int(split[2] * DATASET_SIZE)

		full_dataset = full_dataset.shuffle(DATASET_SIZE)
		train_dataset = full_dataset.take(train_size)
		train_dataset = train_dataset.batch(BATCH_SIZE)
		test_dataset = full_dataset.skip(train_size)
		val_dataset = test_dataset.skip(val_size)
		val_dataset = val_dataset.batch(BATCH_SIZE)
		test_dataset = test_dataset.take(test_size)
		test_dataset = test_dataset.batch(BATCH_SIZE)

		self.train_ds = train_dataset
		self.val_ds = val_dataset
		self.test_ds = test_dataset

	def train(self, lr = 0.001, mntm=0.9, num_epochs=15, save_name = 'SavedModels/DeepModel'):
		save_name = self.OutputPath + '/' + save_name
		print(save_name)
		self.network.compile(optimizer=SGD(learning_rate=lr, momentum = mntm),loss=tf.keras.losses.MeanSquaredError(),metrics=['mse'])
		self.history = self.network.fit(self.train_ds, validation_data=self.val_ds, epochs=num_epochs)

		fig1 = plt.gcf()
		plt.plot(self.history.history['mse'])
		plt.plot(self.history.history['val_mse'])
		plt.axis(ymin=0,ymax=.5)
		plt.grid()
		plt.title('Model Error: Learning Rate ' + str(lr))
		plt.ylabel('MSE')
		plt.xlabel('Epochs')
		plt.legend(['train', 'validation'])
		plt.savefig(os.path.join(self.fig_path,'TrainingLoss.png'))
		
		tf.keras.models.save_model(self.network,save_name)

	def evaluate(self):

		self.network.evaluate(self.test_ds)
		
		numpy_labels = []
		numpy_images = []
		prediction = []
		TestResults = []
		for images, labels in self.test_ds.unbatch().take(-1): 
		    numpy_images.append(images.numpy().astype("uint8"))
		    numpy_labels.append(labels.numpy())
		    p = self.network.predict(np.expand_dims(images.numpy().astype("uint8"), axis=0)).round()[0,0]
		    prediction.append(p)
		    TestResults.append(TestPoint(images.numpy().astype("uint8"), labels.numpy(), p))
	    
		print("___________________________________________________")
		print("Preforming Corrected MSE Calculation for Test Set")
		print("___________________________________________________")
		self.error = 0
		self.accuracy = 0
		for result in TestResults:
			if result.ground_truth - result.prediction == 0 :
				self.accuracy += 1
			else:
				self.error += (result.ground_truth - result.prediction)**2	

		self.error = self.error / len(TestResults)
		self.accuracy = self.accuracy / len(TestResults)

		print("Corrected MSE       : " + str(self.error))
		print("Prediction Accuracy : " + str(self.accuracy))

		saveData(os.path.join(self.result_path,'TestResults.obj'),TestResults)


	    


