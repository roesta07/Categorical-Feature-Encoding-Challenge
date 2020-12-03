import pandas as pd
from sklearn import model_selection



if __name__=="__main__":
    df=pd.read_csv("input/train.csv")
    ## creating new column 
    df["kfold"]=-1 #can be any number
    ## as frac is equal to 1 this is just going to suffle the data
    ## reset_index to turn it into dataframe and drop allows us to drop the older index
    df=df.sample(frac=1).reset_index(drop=True)
    ## stratifier objects
    kf= model_selection.StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    ## each stratified split gives only the splitted index 
    ## by default eigthy percent of the dataframe will be x and 20 percent will be y
    ## i.e eight percent will be train_idx and rest will be assigned to val_idx
    ## which further will be divided into five folds
    for fold, (train_idx,valid_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(valid_idx))
        ## here we use stratifiedKfold little differently
        ## we ignore train_index and 
        ## assigning fold number to each index only from valid_idx covering all dataset
        ## i.e 20% repeated to 5 folds make 100 percent
        df.loc[valid_idx,"kfold"]=fold
        df.to_csv("input/train_folds.csv",index=False)