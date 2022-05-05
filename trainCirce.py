from CountNet import *
import os

if __name__=="__main__":
	Model = CountNet("Model_EP15_LR001_BS32")
	Model.setData(os.path.join('Datasets','FullDataset.obj'), batch_size = 32, split = [.8,.13,.07])
	Model.train(lr = 0.001, mntm=0.9, num_epochs=15, save_name = 'SavedModels/DeepModel')
	Model.evaluate()

	