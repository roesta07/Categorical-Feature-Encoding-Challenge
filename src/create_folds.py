import pandas as pd
from sklearn import model_selection



if __name__=="__main__":
    df=pd.read_csv("input/train.csv")
    ## creating new column 
    df["kfold"]=0
    ## as frac is equal to 1 this is just going to suffle the data
    ## reset_index to turn it into dataframe and drop allows us to drop the older index
    df=df.sample(frac=1).reset_index(drop=True)
    ## stratifier objects
    kf= model_selection.StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    ## each stratified split gives only the splitted index 
    for fold, (train_idx,val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(val_idx))
        ## assigning fold number to each index
        df.loc[val_idx,"kfold"]=fold
        # df.to_csv("input/train_folds.csv",index=False)