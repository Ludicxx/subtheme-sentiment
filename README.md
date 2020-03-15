# Sub Theme - Sentimental Analysis

## Problem Statement
The task is to develop an approach that, given a review, will identify the sub themes along with their respective sentiments.


## Code Structure
This repository has 4 folders -
- model
- code
- dataset
- trash-code

### model
This folder contains the pickle files for trained model, tokenizer, label set and a script for running the trained model on input data.

### code
This folder contains the script for data cleaning, pre-processing, model training, saving the pickle files in model folder and prediction on test set.

### dataset
This folder contains the dataset used for this problem.

### trash-code
This folder contains all the other models and algorithms tried in the process which were not used in the final model.

### Dependencies
Following is the list of python libraries that need to be installed before running the code - pandas, numpy, sklearn, tensorflow, keras, pickle