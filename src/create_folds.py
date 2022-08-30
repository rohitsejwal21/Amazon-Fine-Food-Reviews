import config 

import sqlite3
import pandas as pd
from sklearn.model_selection import StratifiedKFold 

def score_map(s):
    if s > 3:
        return 1
    else:
        return 0

if __name__ == '__main__':

    conn = sqlite3.connect(config.DATABASE_FILE)
    df = pd.read_sql_query('''
    SELECT * FROM Reviews LIMIT 500000
    ''', 
    conn)

    df.loc[:, 'Score'] = df['Score'].map(score_map)

    df = df.sample(frac=0.15).reset_index(drop=True)
    y = df['Score']

    skf = StratifiedKFold(n_splits=5)

    for fold, (train_, cv_) in enumerate(skf.split(X=df, y=y)):
        df.loc[cv_, 'kfold'] = fold 

    df.to_csv('../input/train_folds.csv', index=False)