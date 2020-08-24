# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pickle
import re


def load_data(database_filepath):
    '''Load data into dataframe from the database'''
    engine = create_engine('sqlite:///./'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns.values
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT
    text - string

    OUTPUT
    clean_tokens - a list of words

    This function processes the input using the following steps :
    1. Remove punctuation characters
    2. Tokenize text into list
    3. Lemmatize, Normalize and Strip each word
    4. Remove stop words
    '''
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Reduce words to their root form and normalize
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            # Remove stop words
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Build a machine learning pipeline'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(), n_jobs=-1))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    return model.score(X_test, Y_test)


def save_model(model, model_filepath):
    # Export your model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''This main function performs the following steps :
    1. Load in data from database
    2. Split data into train and text sets
    3. Build a model using machine learning pipeline
    4. Fit the model using the train dataset
    5. Evaluate on the test dataset
    6. Save the model
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        print(model.get_params().keys())
        parameters_random = {

            'clf__estimator__min_child_weight': (0, 0.1, 1, 10, 100, 500),
            'clf__estimator__learning_rate': (0.01, 0.1, 1),
            'clf__estimator__min_split_loss': (0, 0.01, 0.1, 1),
            'clf__estimator__max_depth': (2, 6, 8, 10, 12),
            'clf__estimator__reg_lambda': (0.1, 1, 2, 5, 10)
        }

        random_model = RandomizedSearchCV(model, parameters_random,cv=5)

        print('Training model...')
        random_model.fit(X_train, Y_train)
        print("Best Score: ", random_model.best_score_)  # Best Score:  0.2802774609053288
        print("Best Params: ", random_model.best_params_) # {'clf__estimator__reg_lambda': 10,
        # 'clf__estimator__min_split_loss': 0, 'clf__estimator__min_child_weight': 1, 'clf__estimator__max_depth':
        # 10, 'clf__estimator__learning_rate': 1}

        # parameters_grid = {
        #
        #     'clf__estimator__min_child_weight': (0, 0.1, 1, 5),
        #     'clf__estimator__learning_rate': (0.1, 1, 3),
        #     'clf__estimator__min_split_loss': (0, 0.01),
        #     'clf__estimator__max_depth': (8, 10, 12),
        #     'clf__estimator__reg_lambda': (5, 10, 15)
        # }
        # grid_model = GridSearchCV(model, parameters_grid, cv=5)

        # print('Training model...')
        # grid_model.fit(X_train, Y_train)
        # print("Best Score: ", grid_model.best_score_)
        # print("Best Params: ", grid_model.best_params_)

        print('Evaluating model...')
        print(evaluate_model(random_model, X_test, Y_test, category_names)) # 0.2925247902364607
        # print(evaluate_model(grid_model, X_test, Y_test, category_names))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(random_model, model_filepath)
        # save_model(grid_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()