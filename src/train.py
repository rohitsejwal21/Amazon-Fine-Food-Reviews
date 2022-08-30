import config
import train_vectorize
import train_tune 
import preprocess

from datetime import datetime 
import re 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd 
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

def run(fold, algo):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    # Preprocess Date
    df.loc[:, 'TimeDate'] = df['Time'].apply(preprocess.get_time_data) 
    df = preprocess.preprocess_date(df)

    # Preprocess Text
    df['Summary'] = preprocess.clean_text(df['Summary'])
    df.loc[:, 'Summary'] = df['Summary'].apply(preprocess.remove_stopwords)

    df['Text'] = preprocess.clean_text(df['Text'])
    df.loc[:, 'Text'] = df['Text'].apply(preprocess.remove_stopwords)

    #print(df.columns)

    features_label = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', \
                    'DayOfYear', 'WeekOfYear', 'LeapYear', 'Quarter']

    for col in features_label:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

        lbl = LabelEncoder()
        lbl.fit(df[col])

        df.loc[:, col] = lbl.transform(df[col])
        #print(col, df[col].dtype)

    numerical_features = ['HelpfulnessNumerator', 'HelpfulnessDenominator','Year', 'Month', 'Day', \
    'DayOfWeek', 'IsWeekend', 'DayOfYear', 'WeekOfYear', 'LeapYear', 'Quarter']

    train_df = df[df['kfold'] != fold]
    cv_df = df[df['kfold'] == fold]

    # Get the vectorized forms of Summary and Text Column
    # technique = 'BoW' & 'Tfidf' Supported
    train_summary_vector, train_text_vector, cv_summary_vector, cv_text_vector = train_vectorize.vectorize(train_df, cv_df, technique='BoW')

    # Training Data
    X_train = scipy.sparse.hstack((
        #train_df[numerical_features],
        train_summary_vector,
        train_text_vector
    )).tocsr()

    # Cross Validation Data
    X_cv = scipy.sparse.hstack((
        #cv_df[numerical_features],
        cv_summary_vector,
        cv_text_vector
    )).tocsr()

    # Train and CV Labels
    y_train = train_df['Score']
    y_cv = cv_df['Score']

    # Tune & fit the best model: Returns the best model with fitted data    
    model = train_tune.training_tuning_model(X_train, X_cv, y_train, y_cv, algo)

    # Get predictions
    preds_cv = model.predict(X_cv)
    
    # Evaluation
    accuracy = accuracy_score(y_cv, preds_cv)
    f1 = f1_score(y_cv, preds_cv)

    print(model)
    print(f'Accuracy: {accuracy}, F1 Score: {f1}')

if __name__ == '__main__':

    for f in range(0, 1):
        print(f'Fold {f}:')
        run(fold=f, algo='lr')