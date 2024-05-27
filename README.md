# Disaster Relief Response: Message Classification

<font size='4'>

<center>

<p align="center">
    <img src="images/eyalgever.png" width="500" height="300">
    <br>
    <font size='2'><em>Digital Sculputre: Eyal Gever</em></font>
</p>

</center>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Cleaning](#data-cleaning)  
3. [Machine Learning](#machine-learning)
3. [Web Application](#Web-Application)
4. [File Descriptions](#file-descriptions)
5. [Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)


# Project Overiew
This project addresses a problem that occurs during a disaster relief response.   There are too many messages coming in from the public, and the response agencies are unable to filter through the messages to find those that are important.

The goal of this project is to build a machine learning model that can categorize the messages and dispatch them to the appropriate agency, with a good level of accuracy.

The data for this project comes from [Appen](https://www.appen.com/), and includes approximately 25,000 messages that were sent during a disaster.  The data was cleaned and preprocessed, and then used to train a machine learning model.

The project includes a web application that will allow the user to input a message, and the model will categorize the message and dispatch it to the appropriate agency.  

# Data Cleaning
An ETL script was written to clean and preprocess the data which will be used to train a machine learning model.
The script has 3 parameters, the first two are the paths to the csv files containing the messages and labels, and the third is the path to the database where the cleaned data will be stored.

 To run the ETL script, navigate to the data folder and run the following command:
```bash
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db
```
The script will extract the required data, transform into a usable format, and load it into a SQLite database.

# Machine Learning
A machine learning pipeline was created to train a model to classify the messages.  The pipeline includes transformers to preprocess the text data, and a classifier to categorize the messages.  The script has two parameters, the first is the path to the database containing the cleaned data, and the second is the path to the pickle file where the model will be saved.

To run the machine learning pipeline, use the following command:
```bash
python train_classifier.py DisasterResponse.db classifier.pkl
```

# Web Application
A web application was created to allow the user to input a message, and the model will categorize the message and dispatch it to the appropriate agency.

To run the web application, navigate to the app folder and run the following command:
```bash
python run.py
```
Type http://127.0.0.1:3000 in the web browser to access the web application from the internal webserver.

</font>