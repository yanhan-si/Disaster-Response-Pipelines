import json
import re
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_proportion = df.iloc[:, 4:].sum(axis=0) / df.shape[0]
    category_proportion = category_proportion.sort_values(ascending=False)
    category_names = list(category_proportion.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True,
                    'tickangle': -40
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()