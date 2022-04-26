from CountNet import *
import os



if __name__=="__main__":
	
	Model = CountNet("Test")
	Model.setData(os.path.join('Datasets','FullDataset.obj'))
	Model.train(lr = 0.01, mntm=0.9, num_epochs=5, save_name = 'SavedModels/DeepModel')
	Model.evaluate()

	