import numpy as np
from DataPoint import *
import pickle
import DataPoint

def saveDataSet(data, labels, num):
	for i in range(len(data)):
		data[i].num_of_cubes = labels[i]
	file_to_save = open('FinalDataset'+str(num)+'.obj', 'wb') 
	pickle.dump(data, file_to_save)
	file_to_save.close()
	return data
