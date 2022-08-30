import config
from datetime import datetime 
import re 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd 
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

def get_time_data(x):
    return datetime.fromtimestamp(x)

def remove_stopwords(s):

    stop_words = list(stopwords.words('english'))
    word_list = s.split()

    #word_list = [x for x in word_list if len(x)>2]
    word_list = [x for x in word_list if x not in stop_words or len(x)>2]

    return ' '.join(word_list)

def preprocess_date(df):
    
    df.loc[:, 'Year'] = df['TimeDate'].dt.year
    df.loc[:, 'Month'] = df['TimeDate'].dt.month
    df.loc[:, 'Day'] = df['TimeDate'].dt.day
    df.loc[:, 'DayOfWeek'] = df['TimeDate'].dt.dayofweek
    df.loc[:, 'IsWeekend'] = df['DayOfWeek'].apply(lambda x: True if x >= 4 else False)
    df.loc[:, 'DayOfYear'] = df['TimeDate'].dt.dayofyear
    df.loc[:, 'WeekOfYear'] = df['TimeDate'].dt.weekofyear
    df.loc[:, 'LeapYear'] = df['Year'].apply(lambda x: True if x%4 == 0 else False)
    df.loc[:, 'Quarter'] = df['TimeDate'].dt.quarter
    #df.loc[:, 'Weekday'] = df['TimeDate'].dt.day_name()

    return df 

def clean_text(df):

    df = df.str.lower() 
    df = df.apply(lambda x: re.sub(r'http\S+', '', str(x)))
    df = df.apply(lambda x: re.sub(r'http', '', str(x)))
    df = df.apply(lambda x: re.sub(r'www', '', str(x)))
    df = df.apply(lambda x: re.sub(r'<\S+', '', str(x)))
    df = df.apply(lambda x: re.sub(r'[`]', '\'', str(x)))
    df = df.apply(lambda x: re.sub(r'won\'t', 'will not', str(x)))
    df = df.apply(lambda x: re.sub(r'wouldn\'t', 'would not', str(x)))
    df = df.apply(lambda x: re.sub(r'shouldn\'t', 'should not', str(x)))
    df = df.apply(lambda x: re.sub(r'can\'t', 'can not', str(x)))
    df = df.apply(lambda x: re.sub(r'couldn\'t', 'could not', str(x)))
    df = df.apply(lambda x: re.sub(r'doesn\'t', 'does not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'m', ' am', str(x)))
    df = df.apply(lambda x: re.sub(r'\'re', ' are', str(x)))
    df = df.apply(lambda x: re.sub(r'\'s', ' is', str(x)))
    df = df.apply(lambda x: re.sub(r'\'t', ' not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'ll', ' will', str(x)))
    df = df.apply(lambda x: re.sub(r'\'d', ' would', str(x)))
    df = df.apply(lambda x: re.sub(r'\'ve', ' have', str(x)))
    df = df.apply(lambda x: re.sub(r'[!|@|#|$|%|^|?|<|>|\'|"]', '', str(x)))
    df = df.apply(lambda x: re.sub(r'[0-9]', '', str(x)))
    df = df.apply(lambda x: re.sub(r'[-|.|,|:|;|/|~|(|)|[|]|{|}]', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r' +', ' ', str(x)))
    #df = df.apply(lambda x: re.sub(r' i ', ' ', str(x)))
    #df = df.apply(lambda x: re.sub(r' a ', ' ', str(x)))

    return df

def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    # Preprocess Date
    df.loc[:, 'TimeDate'] = df['Time'].apply(get_time_data) 
    df = preprocess_date(df)

    # Preprocess Text
    df['Summary'] = clean_text(df['Summary'])
    df.loc[:, 'Summary'] = df['Summary'].apply(remove_stopwords)

    df['Text'] = clean_text(df['Text'])
    df.loc[:, 'Text'] = df['Text'].apply(remove_stopwords)

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

    # Summary Column Vectorization
    sum_cvt = CountVectorizer()
    sum_cvt.fit(train_df['Summary'])
    train_summary_vector = sum_cvt.transform(train_df['Summary'])
    cv_summary_vector = sum_cvt.transform(cv_df['Summary'])

    # Text Column Vectorization
    txt_cvt = CountVectorizer()
    txt_cvt.fit(train_df['Text'])
    train_text_vector = txt_cvt.transform(train_df['Text'])
    cv_text_vector = txt_cvt.transform(cv_df['Text'])

    # Training Data
    X_train = scipy.sparse.hstack((
        #train_df[numerical_features],
        train_summary_vector,
        train_text_vector
    )).tocsr()

    y_train = train_df['Score']
    
    '''
    X_train = pd.concat(
        [train_df[numerical_features], train_summary_vector, train_text_vector],
        axis=1
    )
    '''

    # Cross Validation Data
    X_cv = scipy.sparse.hstack((
        #cv_df[numerical_features],
        cv_summary_vector,
        cv_text_vector
    )).tocsr()

    y_cv = cv_df['Score']

    '''
    X_cv = pd.concat(
        [cv_df[numerical_features], cv_summary_vector, cv_text_vector],
        axis=1
    )
    y_cv = cv_df['Score']
    '''
    
    best_c = 0
    best_score = 0

    for c in [.1, 1, 10, 100, 1000]:
        model = LogisticRegression(C=c)
        model.fit(X_train, y_train)

        preds_cv = model.predict(X_cv)
        f1 = f1_score(y_cv, preds_cv)
        if f1 > best_score:
            best_score = f1
            best_c = c
        #print(f1)

    # Init Model
    model = LogisticRegression(C=best_c)
    
    # Fit Data to the model
    model.fit(X_train, y_train)

    # Get predictions
    preds_cv = model.predict(X_cv)
    
    # Evaluation
    accuracy = accuracy_score(y_cv, preds_cv)
    f1 = f1_score(y_cv, preds_cv)

    print(model)
    print(f'Accuracy: {accuracy}, F1 Score: {f1}')

if __name__ == '__main__':

    for f in range(0, 2):
        print(f'Fold {f}:')
        run(fold=f, model='lr')