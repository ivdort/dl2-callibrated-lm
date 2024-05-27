# Setup

## Environment setup
Load a conda environment from environment.yml
This can be done by running the following command in the terminal, from the directory where the file is located:

conda env create -f environment.yml


## Dataset generation and preparation
Seeing as we have created two small datasets, these can either be generated from scratch using generation.py (for the math dataset) and 5W_generation.py (for the 5W dataset), both found in the dataset folder, or the full datasets can be found in the same folder, named 'math_dataset_1_to_10_no_div.csv' and '5W_dataset.csv' respectively.

For the abstract-citations model, the dataset can be found here: https://www.kaggle.com/datasets/Cornell-University/arxiv 
It should be downloaded and placed where you want. You can specify the location in the FILE_PATH parameter in the BERT/citations/citations.ipynb notebook.

## Model downloads
All trained models can be found in the following drive: https://drive.google.com/drive/folders/1c5Ij8N2njsnsR-T5stIP-6xJc9iUAHd3?usp=sharing


## Math model
After creating the math dataset by running the generation.py script, change the 'data' path in dl2-bert-base-for-math.ipynb. You can then train a new model using our BERT configuration or download the checkpoint of our trained model and set the correct path to the trained model in the 'Model Evaluation', as well as the following chapters of the notebook. Each of the experiments can be run sequentially and has it's own markdown heading within the notebook.

## 5W model

## Abstract-title model
The code for this model can be found in BERT/citations/citations.ipynb 
Running the code should be straightforward based on the notebook, but training the model locally will take quite some time. If you want to do this, it is recommended to run via Kaggle or some other GPU. Regardless of whether you are training or loading the model, always run the first code cell.


### Dataset preparation
As stated above, the dataset can be found here: https://www.kaggle.com/datasets/Cornell-University/arxiv 
It should be downloaded and placed where you want. You can specify the location in the FILE_PATH parameter in the BERT/citations/citations.ipynb notebook.

### Training
To train the model yourself, run the code cell under 'Training'. The parameters set by default are the ones we used.

### Loading
To load the trained model, specify the MODEL_LOCATION parameter under the 'Loading model' markdown. Then load the data in the following cell.

### Evaluating
To evaluate, run all cells starting from the 'Check Model Calibration' markdown cell.

