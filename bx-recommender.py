import zipfile
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate
import pandas as pd

with open('./BX-Datasets-dump/abc.csv') as f:
    all_lines = f.readlines()

reader = Reader(line_format='user item rating', sep=';')
data = Dataset.load_from_file('./BX-Datasets-dump/abc.csv', reader=reader)

data.split(n_folds=5)
algo = SVD()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

userid = str(276737)
itemid = str("0600570967")
print(algo.predict(userid, itemid))
