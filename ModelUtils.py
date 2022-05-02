import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pickle
import os

from sklearn.metrics import confusion_matrix
import seaborn as sn

from DataPoint import *


def getData(file):
	filehandler = open(file, 'rb') 
	data = pickle.load(filehandler)
	filehandler.close()
	return data
def saveData(file, data):
	file_to_save = open(file, 'wb') 
	pickle.dump(data, file_to_save)
	file_to_save.close()
	
def splitData(data):
	images = []
	labels = []
	for d in data:
		if type(d) is DataPoint:
			images.append(tf.constant(d.image))
			labels.append(tf.constant(d.num_of_cubes))
		else :
			images.append(tf.constant(d.image))
			labels.append(tf.constant(d.ground_truth))
	return images, labels

def mkDir(name):
	files = name.split("/")
	path = None
	for f in files:
		if path == None:
			path = os.path.join(f)
			try: 
				os.mkdir(path)
			except OSError as error:
				# print(error) 
				pass
		else:
			path = os.path.join(path,f)
			try: 
				os.mkdir(path)
			except OSError as error:
				# print(error) 
				pass
	
	return path

def makeDataHist(data):
	x = []

	for p in data:
		x.append(p.num_of_cubes)

	counts = np.bincount(x)

	print(counts[1:])
	fig, ax = plt.subplots()
	ax.bar(range(1,len(counts)), counts[1:], width = 1, align='center', ec='black')
	ax.set(xticks=range(len(counts)))
	plt.ylabel("Number of Frames")
	plt.xlabel("Number of Cubes")
	return fig

def makeResultDataHist(data):
	x = []

	for p in data:
		x.append(p.ground_truth)

	counts = np.bincount(x)

	print(counts[1:])
	fig, ax = plt.subplots()
	ax.bar(range(1,len(counts)), counts[1:], width = 1, align='center', ec='black')
	ax.set(xticks=range(len(counts)))
	plt.ylabel("Number of Samples")
	plt.xlabel("Number of Cubes")
	return fig 

def evalOnDataset(data, model):
	error = 0
	accuracy = 0
	misses = []
	out_data = []
	for result in data:
			
		if type(result) is DataPoint :
			p = model.predict(np.expand_dims(result.image.astype("uint8"), axis=0)).round()[0,0]
			result = TestPoint(result.image, result.num_of_cubes, p)
			
		if result.ground_truth - result.prediction == 0 :
			accuracy += 1
		
		else:
			error += (result.ground_truth - result.prediction)**2
			misses.append(result)

		out_data.append(result) 

	error = error / len(data)
	accuracy = accuracy / len(data)

	print("Corrected MSE       : " + str(error))
	print("Prediction Accuracy : " + str(accuracy))


	return error, accuracy, misses, out_data

def missBoxPlots(data):
	counts = {
		"1" : [],
		"2" : [],
		"3" : [],
		"4" : [],
		"5" : [],
		"6" : [],
		"7" : [],
		"8" : [],
		"9" : [],
		"10" : [],
		"11" : [],
		"12" : [],
		"13" : []
	}

	for d in data:
		if d.ground_truth == 1:
			counts["1"].append(d.prediction)
		elif d.ground_truth == 2:
			counts["2"].append(d.prediction)
		elif d.ground_truth == 3:
			counts["3"].append(d.prediction)
		elif d.ground_truth == 4:
			counts["4"].append(d.prediction)
		elif d.ground_truth == 5:
			counts["5"].append(d.prediction)
		elif d.ground_truth == 6:
			counts["6"].append(d.prediction)
		elif d.ground_truth == 7:
			counts["7"].append(d.prediction)
		elif d.ground_truth == 8:
			counts["8"].append(d.prediction)
		elif d.ground_truth == 9:
			counts["9"].append(d.prediction)
		elif d.ground_truth == 10:
			counts["10"].append(d.prediction)
		elif d.ground_truth == 11:
			counts["11"].append(d.prediction)
		elif d.ground_truth == 12:
			counts["12"].append(d.prediction)

	fig, ax = plt.subplots()
	ax.boxplot(counts.values())
	ax.set_xticklabels(counts.keys())
	return fig

def getConMat(data):
	gt = []
	pred = []
	for d in data:
	    gt.append(d.ground_truth)
	    pred.append(d.prediction)
	cm = confusion_matrix(gt, pred)
	df_cm = pd.DataFrame(cm, range(len(cm[0])), range(len(cm[0])))
	x_axis_labels = list(range(1, len(cm[0])+1))
	y_axis_labels = list(range(1, len(cm[0])+1))
	fig = plt.figure(figsize=(10,7))
	sn.set(font_scale=1.4) # for label size
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='d',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size

	# plt.show()
	return fig
