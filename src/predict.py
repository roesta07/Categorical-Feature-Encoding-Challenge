import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing,linear_model
from sklearn import metrics
import joblib
import numpy as np
import dispatcher


BASE_DIR=os.path.dirname(os.path.realpath(__name__))
TEST_DATA=os.environ.get("TEST_DATA")
MODEL=os.environ.get("MODEL")
# TRAINING_DATA="input/train_folds.csv"
print(MODEL)

FOLD=0

if __name__=="__main__":
    X_test_full=pd.read_csv(TEST_DATA,dtype={'id':np.int32})
    cols=[cols for cols in joblib.load(os.path.join('models',f'{MODEL}_{FOLD}_label_encoder.pkl')).keys()]
    X_test=X_test_full[cols]
    lbl=preprocessing.LabelEncoder()
    for col in X_test.columns:
        X_test[col] =lbl.fit_transform(X_test[col])
## now predicting 
    clf = joblib.load(os.path.join('models',f'{MODEL}_{FOLD}.pkl'))
    preds=clf.predict_proba(X_test)[:,1]
    sample_df=pd.DataFrame({'id':X_test_full['id'].values,'target':preds})
    sample_df.to_csv('sample.csv',index=None)
