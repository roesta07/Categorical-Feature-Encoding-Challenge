from sklearn import model_selection
import pandas as pd
import numpy as np
from django.views.generic import View


"""

-- multi_class_classification [0,1,0,1]
-- multi_class_classification []
-- single_column_regression
-- multi_column_regression
-- multi_label_classification
Tasks--
Write simple tasks


"""

class CrossValidations():
    known_classification_problems={
        'binary_classification': 'classification_fold',
        'multi_class_classification':'classification_fold',
        'single_column_regression':'classification_fold',
        'multi_column_regression':'classification_fold',
        'holdout':'classification_fold',
        'multilayer_label_encoder':'classification_fold',
    }

    def __init__(self,df,target_cols,problem_type):
        self.df=df
        self.target_cols=target_cols
        self.problem_type=problem_type

    def fold(self):
        if self.problem_type in ['single_column_regression','multi_column_regression']:
            ## validate parameters
            if len(self.target_cols) != 1 and self.problem_type =='single_column_regression':
                raise Exception("More than two columns in target_cols")
            if len(self.target_cols) <2 and self.problem_type =='multi_column_regression':
                raise Exception("More than one target solumn required")
            ## map functions
            return self.regression_folds()
    
    def regression_folds(self,num_folds=5):
        target=self.target_cols[0]
        num_bins=10
        min_price=self.df['SalePrice'].min()
        max_price=self.df['SalePrice'].max()
        diff=(max_price-min_price)/num_bins
        bin_dt=[]
        for i in range(num_bins+1):
            if len(bin_dt)==0:
                bin_dt.append(min_price)
            else:
                bin_dt.append(bin_dt[i-1]+(diff))
        self.df['SalesBin']=pd.cut(self.df['SalePrice'],bins=bin_dt,include_lowest=True,labels=[0,1,2,3,4,5,6,7,8,9])
        kf=model_selection.StratifiedKFold(n_splits=num_folds,shuffle=True)
        for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.df,y=self.df['SalesBin'].values)):
            self.df.loc[valid_idx,'kfold']=fold
        return self.df

    def make_folds(self,num_folds=5):
        if self.problem_type not in ['binary_classification','multi_class_classification','single_column_regression','multi_column_regression','holdout_0.1','multilayer_label_encoder']:
            raise Exception("Problem type not specified")
        elif self.problem_type in ['binary_classification','multi_class_classification']:
            if len(self.target_cols) != 1:
                raise Exception("More than two columns in target_cols")
            target=self.target_cols[0]
            kf=model_selection.StratifiedKFold(n_splits=num_folds,shuffle=True)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.df,y=self.df[target].values)):
                print(len(valid_idx))
                self.df.loc[valid_idx,'kfold']=fold

        elif self.problem_type in ['single_column_regression','multi_column_regression']:
            if len(self.target_cols) != 1 and self.problem_type =='single_column_regression':
                raise Exception("More than two columns in target_cols")
            if len(self.target_cols) <2 and self.problem_type =='multi_column_regression':
                raise Exception("More than one target solumn required")
            target=self.target_cols[0]
            num_bins=10
            min_price=self.df['SalePrice'].min()
            max_price=self.df['SalePrice'].max()
            diff=(max_price-min_price)/num_bins
            bin_dt=[]
            for i in range(num_bins+1):
                if len(bin_dt)==0:
                    bin_dt.append(min_price)
                else:
                    print('when i is',i)
                    bin_dt.append(bin_dt[i-1]+(diff))
            self.df['SalesBin']=pd.cut(self.df['SalePrice'],bins=bin_dt,include_lowest=True,labels=[0,1,2,3,4,5,6,7,8,9,10])
            kf=model_selection.StratifiedKFold(n_splits=num_folds,shuffle=True)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.df,y=self.df[target].values)):
                print(len(valid_idx))
                self.df.loc[valid_idx,'kfold']=fold

        elif self.problem_type =='multi_label_classification':
            target=self.target_cols[0]
            cols=df[target].split(',')
            kf=model_selection.StratifiedKFold(n_splits=num_folds,shuffle=True)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.df,y=self.df[cols].values)):
                print(len(valid_idx))
                self.df.loc[valid_idx,'kfold']=fold
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage=float(self.problem_type.split('_')[1])
            num_holdout=int(len(self.df)*holdout_percentage)
            self.df.loc[:num_holdout,'kfold']=0
            self.df.loc[num_holdout:'kfold']=1
        elif self.problem_type=='multilayer_label_encoder':
           ## basically changing the form into single number
            targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.delimeter)))
            
            kf=model_selection.StratifiedKFold(n_splits=num_folds,shuffle=True)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.df,y=targets)):
                self.df.loc[valid_idx,'kfold']=fold
        return self.df

if __name__=="__main__":
    df=pd.read_csv('input/train_house.csv')

    # df=pd.read_csv('../input/train_house.csv')
    cv=CrossValidations(df=df,target_cols=["SalePrice"],problem_type='single_column_regression')
    df=cv.fold()
    print(df.head())





