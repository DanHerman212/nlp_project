# Disaster Relief Response: Message Classification

<font size='4'>

<center>

<p align="center">
    <img src="images/eyalgever.png">
    <br>
    <font size='2'><em>Digital Sculputre: Eyal Gever</em></font>
</p>

</center>

# Table of Contents
1. [Project Overview](#project-overview)
2. [Data Analysis](#data-analysis)
3. [Data Cleaning](#data-cleaning)
4. [Machine Learning](#machine-learning)
5. [Web Application](#web-application)

# Project Overiew
The goal for this project is to create a web application to assist with a natural disaster response.<br>

<br>
Durring a natural disaster there are many agencies involved in responding to specific incidents that are part of the larger disaster.  Response agencies get floaded with messages from people in need of help.  Realistically, only a small percent of messages are real emergencies that require response.  This creates a challenge for responders, as they don't have the time and resources to sift through all the messages to find the ones that are real emergencies.<br>
<br>
To respond to this challenge, we've build a web application, powered by machine learning, that has been trained on 20,000 disaster response messages, from 36 different categories.  The application can take a message as input, analyze the words and identify the appropriate agencies that are required for response.<br>
<center>

<p align="center">
    <img src="images/response_dashboard1.png">
    <br>
    <font size='2'><em>WebApp Dashboard</em></font>
</p>

</center>
<br>
Included in this project is an ETL pipeline to clean and load the data, a machine learning pipeline to train the model, and a web application to display the results.<br>
<br>
The next sections will review the data cleaning process and machine learning workflows, followed by instructions to run the web application.

# Data Analysis
There were approximately 25,000 messages in the dataset.  These messages were assigned to 36 different categories.  These categories represent agencies that respond to messages.  It's possible for a message to be assigned to multiple categories.<br>
<br>

<center>
<p align="center">
    <img src="images/categories.png">
    <br>
    <font size='2'><em>Top Response Categories from Messages</em></font>
</p>
</center>



# Data Cleaning
An ETL script was created to perform the data cleaning process.  The script will extract the data from two CSV files, the first containing messages and the second containing the disaster response categories.  The script will then transform the data into a usable format, and load it into a SQLite database.<br>

 To run the ETL script, navigate to the data folder and run the following command:
```bash
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db
```
The script will extract the required data, transform into a usable format, and load it into a SQLite database.

# Machine Learning
We created a machine learning pipeline to process the incoming messages and classify them to the appropriate agency for response.  The pipeline uses two techniques for machine learning, the first is Natural Language Processing (NLP), we assign values to words based on frequency.  We also strip out words that don't provide meaning.  The second technique is a multi-output classifier, which assigns the message to one or more categories.<br>

The NLP process is listed as follows:
1. Tokenize the messages - split the messages into individual words
2. Remove stop words - remove common words that don't add value to the analysis, e.g. 'the', 'and', 'is'
3. Lemmatize the words - reduce the words to their root form, e.g. 'running' to 'run'
4. Vectorize the words - convert the words into a numerical format that can be used by the model
5. Apply TFIDF - Term Frequency Inverse Document Frequency, a method to determine the importance of a word in a document relative to a collection of documents



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