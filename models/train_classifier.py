# import libraries
import re, sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, f1_score
from sqlalchemy import create_engine
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('features', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words



def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline



def evaluate_model(model, X_test, y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test.columns):
        print(f'label:',col)
        print(classification_report(y_test[col], y_pred[:, i]))
        print('Accuracy: {:.2f}'.format(accuracy_score(y_test[col], y_pred[:, i])))
        print('F1 Score: {:.2f}'.format(f1_score(y_test[col], y_pred[:, i], average='weighted')))
        print()
        print('Confusion Matrix:\n ', multilabel_confusion_matrix(y_test[col], y_pred[:, i]))
        print('------------------------------------------------------')


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath, compress=7)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()