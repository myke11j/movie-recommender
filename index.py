import zipfile
from surprise import Reader, Dataset, SVD, evaluate, NMF
from surprise.model_selection import cross_validate

# Unzip ml-100k.zip
zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
zipfile.extractall()
zipfile.close()

# Read data into an array of strings
with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

# Prepare the data to be used in Surprise
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# Split the dataset into 5 folds and choose the algorithm
data.split(n_folds=5)
algo = NMF()

# Train and test reporting the RMSE and MAE scores
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict a certain item
userid = str(196)
itemid = str(302)
actual_rating = 4
print(algo.predict(userid, itemid))
