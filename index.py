import zipfile
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate

# Unzip ml-100k.zip
zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
zipfile.extractall()
zipfile.close()

with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

data.split(n_folds=5)
algo = SVD()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

userid = str(49)
itemid = str(258)
print(algo.predict(userid, itemid, 2))
