# About messier42
Stretching is an important part of processing RAW images, especially in the field of Astrophotography, where signals from distant Deep space objects can be extremely faint. Therefore, we need to "stretch" the signal. In simple terms, stretching involves changing the luminance values of pixels differently depending on what the source of the signal is. For example, the signal from a cloudy part of a nebula must be made brighter, but the signal from light glow (light pollution) must not be stretched to the same amount (or even be made fainter). This project aims to use an ANN model to perform the stretching process.

The main idea behind the project is that the AI model could learn the stretching patterns of an astrophotographer and mimic their photo editing style, given an input map .csv file (generated from a collection of images by said photographer) 

The project involves scraping data from the internet, which in this case is done by scraping a .csv file uploaded to this repository under the `data/scrapeFromHere` folder. Then, the data is cleaned and an OLS and ANN model are trained and tested using an 80:20 train:test split. Lastly, two docker images, one for the training and one for applying the solution to a given set of data points are created and docker compose files are used to orchestrate the training, testing and application of the ANN solution. 

## How to run the project:
Detailed instructions on how to run the project are available in the documentation/instructions folder. The instructions to run a demo of the project are available here:

### Requirements:
1. Docker Desktop/Docker engine
2. `Python3` and `matplotlib` if you want to visualise the test/training losses for the ANN model

### Steps:
1. Download one of the scenario folders from the repository
2. Make sure Docker Engine is running, if not start it by running the Docker desktop app
3. Open the terminal in the downloaded scenario folder
4. Enter the following command 
```
docker compose up
```        
5. The results will be saved in a newly created ai_system folder

## Ownership
This project is released under the AGPL 3.0 license and was built as a part of the course 'M. Grum: Advanced AI-based Application systems' (University of Potsdam) by Haani Ansari and Dipta Roy Karmakar.

## Structure
This section gives an overview of the code and folder structure of the project.
The project has the following folder structure:
├───code
│   ├───(NotInUse)ann_Tensorflow
│   ├───ann_PyBrain
│   ├───ols
│   └───tools
│       └───visualisers
├───data
│   ├───backup
│   ├───cleanedData
│   ├───extras
│   ├───photos
│   │   ├───input
│   │   ├───output
│   │   └───target
│   ├───scrapedData
│   └───scrapeFromHere
├───documentation
│   ├───instructions
│   └───visualisations
│       └───source
│           ├───ann_Pybrain
│           └───ols
├───images
│   ├───activationBase_messier42
│   ├───codeBase_messier42
│   ├───knowledgeBase_messier42
│   └───learningBase_messier42
└───scenarios
    ├───apply_ann_solution_for_image_stretching
    ├───apply_ols_solution
    ├───create_ann_for_messier42
    └───create_models_and_apply_solution
### 1) Code
The code folder contains all the python scripts that were used to realise the project. 

#### 1.1) ANN_Pybrain/Tensorflow
These 2 folders contain the python code (model.py) required to the train the model. They also store a copy of the trained model(s).
A FNN model is created when running the `hybridModel.py` which uses the `train_data.csv` and `test_data.csv` for the training and validation of the model. The model has 3 input variables (RGB subpixel values) and 2 output variables (the multiplication factor for Red & Blue channel and the multiplication factor for the green channel)

The model configuration is as follows: 
1. Hidden Layers = 100 with Sigmoid activation function
2. 100 rounds of training
The Tensorflow folder currently only contains a placeholder script as the current model was made using Pybrain.

#### 1.2) OLS
This subfolder contains the python code and to realise an OLS model and a copy of the OLS models(s) for the same dataset used to create the ANN model

#### 1.3) Tools
This subfolder contains all the individual python scripts that are a part of the data preperation step. It contains the following scripts:
1. `ingest(16bit).py` - This script is used to convert images to csv data. 
2. `scraper.py` - This script is used to scrape a csv file from a given hyperlink.
3. `cleaner.py` - This script is used to clean the data, after it has been scraped.
4. `editor(16bit).py` - This script is used to edit an image using the ANN model.
5. `visualiser.py` - This script is used to create the required visualisations.
6. `csvSplitter.py` - This script is used to create smaller csv files from a given input csv file.

##### 1.3.1) Visualisers
This subfolder contains all the scripts for visualisations found in the documentation/visualisations folder. The scripts are:
1. `dataVisualiser.py` - This script is used to create visualisations from the scraped and cleaned data.
2. `performanceVisualiser.py` - This script is used to create scatter plots that visualise the performance of the ANN.
3. `visualiser.py` - This script is used to create visualisations not mentioned above.

Note: This folder does not contain the code for all visualisations found in the visualisations folder, namely the diagnostic plots for the OLS model, which were made when the model was created and are present in the `OLS_model.py` script.

### 2) Data
This folder contains the CSV and images used to train and test the ANN model.

#### 2.1) Photos
This folder contains the images (usually in .TIF format) that the model was trained on.

#### 2.2) ScrapeFromHere
This folder contains the csv files generated by the images in the `Images` folder, and this is where the data was scraped from for the project.

#### 2.3) ScrapedData
This folder contains the CSV files scraped from the `ScrapeFromHere` folder on the remote repository.

#### 2.4) CleanedData
This folder contains all the CSV files that have been cleaned, using the `cleaner.py` script. It also contains the test and train data split CSVs.

### 3) Images
This folder contains the activation and knowledge base docker images for the AI application. The images are in their respectively named subfolders

1. activationBase_messier42: This contains the dockerfile and the files loaded to said docker file for building the activation base image availalbe at haaniansari/activationbase_messer42
2. codeBase_messier42: same as above and available at haaniansari/codebase_messier42
3. knowledgeBase_messier42: same as above and availalbe at haaniansari/knowledgebase_messier42
4. learningBase_messier42: same as above and availalbe at haaniansari/learningbase_messier42




