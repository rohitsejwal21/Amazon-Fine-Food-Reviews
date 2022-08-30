from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def training_tuning_model(X_train, X_cv, y_train, y_cv, algo):

    if algo == 'lr':    
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
        
        # Init Model
        model = LogisticRegression(C=best_c)
        
        # Fit Data to the model
        model.fit(X_train, y_train)

        return model
    elif algo =='rf':

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        return model