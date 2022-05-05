# CountNet
This Network is used to count object in video frames. Was developed for a USF Course CIS 6930.  

# Pull Repo
  1. open a terminal 
  2. cd into desired working directory
  3. Run:
  ```console
      git clone git@github.com:ChanceHamilton59/CountNet.git 
  ```
# Datasets and Pretrained Model
This repo has two datasets. They both consists of custom python class objects found in the file "DataPoint.py". These Datasets are found in "Datasets/" folder:
  1. FullDataset.obj consists of 45280 samples, evenly distributied between 1-6 object counts
  2. unseenData.obj consists of 139 samples , they consists of frames with 1-6 object counts. These were generated seperatly from the FullDataset and are used to test how well the model generalizes. 

To get the datasets and pretrained model you will need to download them from the [shared link](https://drive.google.com/drive/folders/1eTfM-uAG4hYV5Y3Px67QohSmAxTjV8Id?usp=sharing) or by running the following line in a terminal:

  ```console
       wget --no-check-certificate "https://drive.google.com/drive/folders/1eTfM-uAG4hYV5Y3Px67QohSmAxTjV8Id?usp=sharing"
  ```
 Once you download the dataset and pretrained model you will need to move them into the **CountNet/** directory. Take note of their local path as you will need them to either train the model or load it for your own testing. The code provided works assuming you have the following file structure:
 
 ```
CountNet   
└───Datasets
│   │   FullDatasets.obj
│   │   unseen_data.obj
│   
└───Model_EP15_LR001_BS32
|   └───RESULTS
│       │   ...
|   └───SavedModels
│       │   ...
│   ...
│   ...
│   ...
```

# To Train Locally
If you wish to run the model locally on your machine from the file open a terminal session
  1. Run:
  ```console
       cd CountNet/
       jupyter lab TrainLocal.ipynb
  ```
  4. Run all the cells. Make changes to cell 2 line 3 to reflect the **path to the dataset** and the **name you wish to save the trained model**.
  5. The last cells are used to test the model on the test set and an unseen dataset that the model has never been exposed to
## Note
This code will create files and images as well as save the trained model and the testing results on the testset.


# Run on usf Circe
After you secure copy this repo to your working directory on circe.rc.usf.edu:
  1. Run:
  ```console
       cd CountNet/
       sbatch run.sh
  ``` 
## Needed Changes
1. You need to make changes to the run.sh file (line 10) to reflect your working directory on Circe.
2. If you wish to alter the parameters of the model you will need to make changes the trainCirce.py file to reflect the changes similar to what is seen in **TrainLocal.ipynb**

# Evaluate Locally
Assuming you set up your file stucture as outlined in the above sections, You should be able to launch the Jupyter notebook called **EvalPretrainedModel.ipynb** by running the following line in a terminal from the **CountNet** directory
 ```console
       jupyter lab EvalPretrainedModel.ipynb
  ```
  ## Note
  This assumes that your have set up your file directly exactly as outlined above. If not you will need to make changes Cell 2 Line 1 to reflect the local path to the pretrained model and in Cell 3 line 1 to reflect the path to the test results saved by the evaluations process.
  
# Coppellia Sim
We have provided tools for labeling and viewing the frames. These python scripts can be seen in DataTools. We have also included the Coppellia Sim environment file used to recorde the videos. There are also the files needed to run trails of a robotic hand pouring the cubes.

## Note
You will need to fix the Coppellia API to match your machine. 
