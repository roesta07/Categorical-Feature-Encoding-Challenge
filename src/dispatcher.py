from sklearn import ensemble,linear_model

MODELS={
    "randomforest":ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1,verbose=2),
    "log":linear_model.LogisticRegression(random_state=0,max_iter=300)
}