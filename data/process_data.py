# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath="./data/disaster_messages.csv",
              categories_filepath="./data/disaster_categories.csv"):
    '''Load csv files into dataframes and return'''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    return messages_df, categories_df


def clean_data(messages_df, categories_df):
    '''
    INPUT
    messages_df - pandas dataframe
    categories_df - pandas dataframe

    OUTPUT
    df - a cleaned and concatenated dataframes from the two inputs

    This function cleans df using the following steps :
    1. Inner join messages_df and categories_df on the id
    2. Split categories into separate category columns
    3. Rename the categories columns
    4. One hot encoding the categories
    5. Drop the origianl categories column from df
    6. Concatenate messages_df and categories_df
    7. Remove duplicates
    '''
    # merge datasets
    df = messages_df.merge(categories_df, how='inner', on=['id'])

    # create a dataframe of the 36 individual category columns
    categories_df = df.categories.str.split(pat=';', expand=True)

    # use the first row to extract a list of new column names for categories
    category_colnames = categories_df.iloc[0].apply(
        func=lambda x: x[:-2]
    ).to_list()
    categories_df.columns = category_colnames

    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories_df[column] = categories_df[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    '''Save data into database'''
    engine = create_engine('sqlite:///'+database_filepath)
    filename = database_filepath.split("/")
    df.to_sql(filename[1][:-3], engine, index=False)


def main():
    '''This main function performs the following steps :
    1. Load in data from .csv files
    2. Clean data
    3. Save the cleaned data into database
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages_df, categories_df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
