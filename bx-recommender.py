import zipfile
from surprise import Reader, Dataset, SVD, evaluate, NMF
from surprise.model_selection import cross_validate
import pandas as pd

# Read data into an array of strings
# with open('./BX-Datasets-dump/BX-Book-Ratings.csv') as f:
#     all_lines = f.readlines()

df = pd.read_csv("./BX-Datasets-dump/abc.csv", sep=';')
# df = pd.read_csv("./BX-Datasets-dump/BX-Book-Ratings.csv", sep=';', encoding = "ISO-8859-1")
# Prepare the data to be used in Surprise
reader = Reader(line_format='user item rating', sep=';', rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# Split the dataset into 5 folds and choose the algorithm
data.split(n_folds=5)
algo = SVD()

# Train and test reporting the RMSE and MAE scores
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict a certain item
userid = str("6516")
itemid = str("0515090506")
print(algo.predict(userid, itemid))
