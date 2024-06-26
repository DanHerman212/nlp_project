# import libraries
import re, sys  # regular expression and system libraries
import numpy as np  # numerical python
import pandas as pd  # data manipulation
from nltk.corpus import stopwords  # natural language toolkit
from nltk.tokenize import word_tokenize  # natural language toolkit
from nltk.stem import WordNetLemmatizer  # natural language toolkit
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)  # scikit-learn, text processing, feature extraction
from sklearn.model_selection import (
    train_test_split,
)  # scikit-learn, model selection, splitting data
from sklearn.multioutput import (
    MultiOutputClassifier,
)  # scikit-learn, multioutput classifier,
from sklearn.ensemble import (
    RandomForestClassifier,
)  # scikit-learn, ensemble, random forest classifier
from sklearn.pipeline import (
    Pipeline,
)  # scikit-learn, pipeline object for chaining multiple estimators
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    multilabel_confusion_matrix,
    f1_score,
)  # model evaluation objects
from sqlalchemy import create_engine  # SQL toolkit and Object Relational Mapper
import joblib  # joblib is a set of tools to provide lightweight pipelining in Python
import warnings  # warning control
from sklearn.model_selection import (
    GridSearchCV,
)  # scikit-learn, model selection, grid search

warnings.filterwarnings("ignore")  # ignore warnings


# create function to load data from SQLite database
# setup variables for machine learning
def load_data(database_filepath):
    """
    This function loads data from SQLite database and assigns variables for machine learning.

    Args:
    database_filepath: string. Filepath for SQLite database containing cleaned data.

    Returns:
    X: pandas Series. Series containing messages.
    Y: pandas DataFrame. DataFrame containing categories.
    category_names: list. List of category names.
    """
    # create object of SQLite engine
    engine = create_engine("sqlite:///" + database_filepath)
    # read data from SQLite database assign to df
    df = pd.read_sql_table("features", engine)
    # assign X and Y values
    X = df["message"]
    Y = df.iloc[:, 4:]
    # assign category names, will serve as column names for transformation
    category_names = Y.columns

    return X, Y, category_names


# create function to process the text
def tokenize(text):
    """
    This function processes the text data by normalizing, tokenizing, removing stop words, and lemmatizing.

    Args:
    text: string. Text data to be processed.

    Returns:
    words: list. List of processed words.
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words - words without meaning
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization - reduce words to their base or root form
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words


# create function to build the model and pipeline steps
def build_model():
    """
    This function builds a machine learning pipeline with a grid search object.

    Returns:
    cv: GridSearchCV object. Grid search object with pipeline and parameters.
    """
    pipeline = Pipeline([  
            ( "vect", CountVectorizer(tokenizer=tokenize, min_df=5, max_df=0.75)),  # Convert a collection of text 
            ( "tfidf", TfidfTransformer()),  # Transform a count matrix to a normalized tf or tf-idf representation
            ("clf", MultiOutputClassifier(RandomForestClassifier())),  # Multi target classification
        ]
    )

    # specify parameters for grid search
    parameters = {
        "vect__ngram_range": ((1, 1),(1, 2)),  
        "clf__estimator__n_estimators": [50, 100],  
        "clf__estimator__min_samples_split": [2,4],  # The minimum number of samples required to split an internal node
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=1)

    return cv


# create function to evaluate the model results
def evaluate_model(model, X_test, y_test, category_names):
    """
    This function evaluates the model by predicting on test data and printing classification report for each category.

    Args:
    model: GridSearchCV object. Trained model.
    X_test: pandas Series. Series containing messages for testing.
    y_test: pandas DataFrame. DataFrame containing categories for testing.
    category_names: list. List of category names.

    """
    # predict on test data
    y_pred = model.predict(X_test)
    # loop through each column and print classification report
    for i, col in enumerate(y_test.columns):
        # print classification report
        print(f"label:", col)
        print(classification_report(y_test[col], y_pred[:, i]))
        print("Accuracy: {:.2f}".format(accuracy_score(y_test[col], y_pred[:, i])))
        print(
            "F1 Score: {:.2f}".format(
                f1_score(y_test[col], y_pred[:, i], average="weighted")
            )
        )
        print()
        print(
            "Confusion Matrix:\n ",
            multilabel_confusion_matrix(y_test[col], y_pred[:, i]),
        )
        print("------------------------------------------------------")


# create function to save the model
def save_model(model, model_filepath):
    """
    This function saves the model to a pickle file.

    Args:
    model: GridSearchCV object. Trained model.
    model_filepath: string. Filepath to save the model to.

    """
    joblib.dump(model, model_filepath, compress=7)


# create main function to run the script
def main():
    """
    This function runs the machine learning pipeline to load, train, evaluate, and save the model.

    """
    # check for correct number of arguments
    if len(sys.argv) == 3:
        # assign arguments to variables
        database_filepath, model_filepath = sys.argv[1:]
        # load data
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # build model
        print("Building model...")
        model = build_model()

        # train model
        print("Training model...")
        model.fit(X_train, Y_train)

        # evaluate model
        print("Evaluating model...")
        evaluate_model(model, X_test, y_test, category_names)

        # save model
        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        # print message to confirm model saved
        print("Trained model saved!")

    else:  # print message to provide correct arguments
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


# run main function
if __name__ == "__main__":
    main()
