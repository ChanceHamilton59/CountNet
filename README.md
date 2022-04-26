# CountNet
This Network is used to count object in video frames. Was developed for a USF Course CIS 6930.  

# Pull Repo
  1. open a terminal 
  2. cd into desired working directory
  3. Run:
  ```console
      git clone git@github.com:ChanceHamilton59/CountNet.git 
  ```
# Datasets
This repo has two datasets. They both consists of custom python class objects found in the file "DataPoint.py". These Datasets are found in "Datasets/" folder:
  1. FullDataset.obj consists of 45254 samples, evenly distributied between 1-6 object counts
  2. unseenData.obj consists of 118 samples , they consists of frames with 7-13 object counts. These were never used for training and are used to test how the model generalizes to unseen object counts. 

To get the data sets you will need to download them from the shared link provided by onedrive. You can get the data by clicking [here](https://drive.google.com/drive/folders/1eTfM-uAG4hYV5Y3Px67QohSmAxTjV8Id?usp=sharing) or by running the following line in a terminal:

  ```console
       wget --no-check-certificate "https://drive.google.com/drive/folders/1eTfM-uAG4hYV5Y3Px67QohSmAxTjV8Id?usp=sharing"
  ```

# To Run Locally
If you wish to run the model locally on your machine from the file open a terminal session
  1. Run:
  ```console
       cd CountNet/
       jupyter lab TrainLocal.ipynb
  ```
  4. Run all the cells. Make changes to cell 2 line 3 to reflect the path to the dataset and the name you wish to save the trained model under the desired perameters. 
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
2. If you wish to alter the parameters of the model you will need to make changes the trainCirce.py file to reflect the changes
