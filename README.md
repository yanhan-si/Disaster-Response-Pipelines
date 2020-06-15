# Disaster Response Pipeline Project

### Overview:

When a disaster happens, there will be millions of communications either direct or via social media. At this time, disaster response organizations need to filter and then pull out the messages which are the most important and relevant.

The way that disasters are typically responded to is that different organizations will take care of different parts of the problem. One organization might care about medical supplies, another would care about electricity, etc.

These categories are what we need to pull out for each message in the dataset. In this project, I built a data pipeline to prepare the message data from major natural disasters around the world and a machine learning pipeline to categorize emergency text messages based on the need communicated by the sender. Finally, this disaster response pipeline was deployed through a Flask web app.

-------------------------------

## I. ETL Pipeline
In this step, the raw unstructured data are extracted from the .csv files. Then they are transformed by categorical feature engineering, one-hot encoding, concatenation, etc. In the end, the structured data are loaded into the SQL database.

## II. ML Pipeline
In the machine learning pipeline, the data are loaded from the SQL database and then go through CountVectorizer, TfidfTransformer and MultiOutputClassifier.

## III. Web APP deployment
Here is what the web page looks like. You can enter a message and get the output immediately.
Let's take this sentence *"I'm badly injured. I need help"* as an example and see what the output looks like.
Our classifier correctly classifies this message into *"Related", "Request", "Aid Related" and "Medical Help"* categories.

![Prediction](https://github.com/yanhan-si/Disaster-Response-Pipelines/blob/master/Prediction.png)

Summary of the training data

![Training](https://github.com/yanhan-si/Disaster-Response-Pipelines/blob/master/TrainingOverview.png)

-------------------------------

####  ***List of python libraries used***
*pandas
*numpy
*nltk
*sklearn
*flask
*plotly
*sqlalchemy
*re

-------------------------------

#### ***Files in the repository***

*data/process_data.py -- a data ETL pipeline

*models/train_classifier.py -- a machine learning pipeline

*app/run.py -- a Flask web app

-------------------------------

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
