import sys
import pandas as pd

# create function to load data, take filepath as input from text data and labels
def load_data(messages_filepath, categories_filepath):
    # load messages
    messages = pd.read_csv(messages_filepath)
    # load categories
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')

    return df

# create function to clean the data, prepared for machine learning
def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df
    

# create function to save the data, take the cleaned dataset and save it to an sqlite database
def save_data(df, database_filename):
    # save the clean dataset into an sqlite database
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + 'disaster_response.db')
    
    return df.to_sql('features', engine, index=False, if_exists='replace')

# create main function to run the ETL pipeline
def main():
    # check for correct number of arguments
    if len(sys.argv) == 4:
        # assign arguments to variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # save data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        # print message to confirm data saved
        print('Cleaned data saved to database!')
    
    else: # print message to provide correct arguments
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# 
if __name__ == '__main__':
    main()