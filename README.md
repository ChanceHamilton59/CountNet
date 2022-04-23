# CountNet
This Network is used to count object in video frames. Was developed for a USF Course CIS 6930.  

# Pull Repo
  1. open a terminal 
  2. cd into desired working directory
  3. run $ git clone (link to this directory) 

# To Run Locally
If you wish to run the model locally on your machine from the file open a terminal session
  1. $ cd CountNet/
  2. $ jupyter lab TrainLocal.ipynb
  3. Run all the cells. Make changes to cell 2 line 3 to reflect the path to the dataset and the name you wish to save the          trained model under the desired perameters. 
  4. The last cells are used to test the model on the test set and an unseen dataset that the model has never been exposed to

# Run on usf Circe
After you secure copy this repo to your working directory on circe.rc.usf.edu:
  1. $ cd into repo
  2. $ sbatch run.sh
## Changes
1. You need to make changes to the run.sh file (line 10) to reflect your working directory on Circe.
2. If you wish to alter the parameters of the model you will need to make changes the trainCirce.py file to reflect the changes
