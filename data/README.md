## Data Readme File

The data folder contains the following:
  - ```BERPublicSearch``` folder is where downloaded dataset is to be saved to as a text file
  - ```training``` folder is where the balanced dataset is stored and saved to git
  - ```build_balanced_dataset.ipynb``` is the notebook to be used to build the balanced dataset from the downloaded dataset. The downloaed dataset must first be split into a number of smaller files before running this notebook. The new balanced dataset csv file will be saved to the ```training``` folder

### Updating the dataset
  - download the new dataset from https://ndber.seai.ie/BERResearchTool/ber/search.aspx and save to the ```BERPublicSearch``` folder
  - follow the instructions in the ```build_balanced_dataset.ipynb``` notebook for splitting the file
  - run the ```build_balanced_dataset.ipynb``` notebook
  - add and commit the updated ```ber-rating-dataset-final.csv``` file in git  