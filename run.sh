export TRAINING_DATA=input/train_folds.csv
export MODEL=$1
export TEST_DATA=input/test.csv

# FOLD=2 python src/train.py
# FOLD=3 python src/train.py
# FOLD=4 python src/train.py
python src/predict.py 