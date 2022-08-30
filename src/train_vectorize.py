import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(train_df, cv_df, technique):

    if technique == 'BoW':
        sum_cvt = CountVectorizer()
        txt_cvt = CountVectorizer()
    elif technique == 'Tfidf':
        sum_cvt = TfidfVectorizer()
        txt_cvt = TfidfVectorizer()

    # Summary Column Vectorization
    sum_cvt.fit(train_df['Summary'])
    train_summary_vector = sum_cvt.transform(train_df['Summary'])
    cv_summary_vector = sum_cvt.transform(cv_df['Summary'])

    # Text Column Vectorization
    txt_cvt.fit(train_df['Text'])
    train_text_vector = txt_cvt.transform(train_df['Text'])
    cv_text_vector = txt_cvt.transform(cv_df['Text'])
    
    return train_summary_vector, train_text_vector, cv_summary_vector, cv_text_vector 