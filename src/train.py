import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing,linear_model
from sklearn import metrics
import joblib
import numpy as np
import dispatcher


TRAINING_DATA=os.environ.get("TRAINING_DATA")
FOLD=int(os.environ.get("FOLD"))
MODEL=os.environ.get("MODEL")
TEST_DATA=("input/test.csv")
# TRAINING_DATA="input/train_folds.csv"

FOLD_MAPPING={
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}
print('fold is',FOLD)



if __name__=="__main__":
    df=pd.read_csv(TRAINING_DATA)
    X_test_full=pd.read_csv(TEST_DATA,dtype={'id': np.float64})
    preds_list=[]
    ## filters kfold where it is equal to 0 which maps to [1,2,3,4]

    X_train=df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]

    ## filters valid data for valid where fold = 0
    X_valid=df.loc[df.kfold==FOLD]

    y_train=X_train.target.values
    y_valid=X_valid.target.values

    ## Deleting Rest of the columns
    X_train=X_train.drop(["id","target","kfold"],axis=1)
    X_valid=X_valid.drop(["id","target","kfold"],axis=1)
    
    ## test data
    ## we need id for test data

    ## this code is a precaution to align the columns of train and valid data to be same
    X_valid=X_valid[X_train.columns]
    X_test=X_test_full[X_train.columns]
    label_encoders={}
    lbl=preprocessing.LabelEncoder()
    
    for col in X_train.columns:
        ## old way
        # lbl.fit(X_train[c].values.tolist()+X_valid[c].values.tolist())
        # X_train.loc[:,c]=lbl.transform(X_train[c].values.tolist())
        # X_valid.loc[:,c]=lbl.transform(X_valid[c].values.tolist())
        # new way it can be acheived ihn two lines
        X_train[col]=lbl.fit_transform(X_train[col])
        X_valid[col]=lbl.fit_transform(X_valid[col])

        X_test[col] =lbl.fit_transform(X_test[col])       
        label_encoders.update({col:lbl})
    print(label_encoders)
## data is ready to train
    ## get the required model from the dispatcher.py
    clf=dispatcher.MODELS[MODEL]
    clf.fit(X_train,y_train)
    """
    few things about predict_prob
    predict_proba returns a probability of observing both 0 and 1 
    whereas predict returns only zero and 1
    if you want only prob of getting 1 using predict_proba use [:,1] at the end
    """

    preds=clf.predict_proba(X_valid)[:,1]
    # print(preds)
    
    # preds_list.append(metrics.roc_auc_score(y_valid,preds))
    print(metrics.roc_auc_score(y_valid,preds))
    # use joblib to save pkl file
    joblib.dump(label_encoders,f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}_{FOLD}.pkl")
    # print(preds_list)
    # print('mean score',np.array(preds_list).mean())






