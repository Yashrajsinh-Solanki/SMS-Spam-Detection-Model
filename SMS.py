import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df.dropna(how="any", inplace=True, axis=1)
df.columns = ['label', 'message']

# Map labels to numerical values
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Define a function for text processing
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

# Apply text processing to the messages
df['clean_msg'] = df.message.apply(text_process)

# Split the dataset into training and testing sets
X = df.clean_msg
y = df.label_num
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Define the pipeline with Logistic Regression model
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression(solver='liblinear'))])

# Train the model
pipe.fit(X_train, y_train)

# Save the trained model and vectorizer using joblib
joblib.dump(pipe, 'logreg_model.pkl')
