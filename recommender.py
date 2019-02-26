import pandas as pd 
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Read datasets from CSVs
df = pd.read_csv('datasets/ml-latest-small/ratings.csv', names=['user_id','item_id','rating','titmestamp'])
movieTitles = pd.read_csv('datasets/ml-latest-small/movies.csv')

# Drop row at index 0 as it has columns, found out check manually after getting ValueError
df.drop(df.index[[0]], inplace=True)

# Convert Object to numeric types for processing
df['item_id'] = df['item_id'].astype(str).astype(int)
df['rating'] = df['rating'].astype(str).astype(float)

# Renamed movieId column to item_id to merge titles into df
movieTitles.rename(columns={'movieId':'item_id'}, inplace=True)

df = pd.merge(df, movieTitles, on='item_id')


# We will create a new dataframe which will have rating for title and number of ratings it received.
averageRatingsDf = pd.DataFrame(df.groupby('title')['rating'].mean())
averageRatingsDf['numberOfRatings'] = df.groupby('title')['rating'].count()

averageRatingsDf = averageRatingsDf.sort_values(['numberOfRatings', 'rating'], ascending=False)

# Some visualization
sns.jointplot(x='rating', y='numberOfRatings', data=averageRatingsDf)

# Data engineering
movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
movie_matrix.head()

# We will pick two movies to give recommendation
forest_gump_user_rating = movie_matrix['Forrest Gump (1994)']
braveheart_user_rating = movie_matrix['Braveheart (1995)']

# Find correlation between watched movies and all movies
similar_to_forest_gump = movie_matrix.corrwith(forest_gump_user_rating)
similar_to_braveheart = movie_matrix.corrwith(braveheart_user_rating)

corr_forest_gump = pd.DataFrame(similar_to_forest_gump, columns=['Correlation'])
corr_forest_gump.dropna(inplace=True)
corr_forest_gump.head()

corr_braveheart = pd.DataFrame(similar_to_braveheart, columns=['correlation'])
corr_braveheart.dropna(inplace=True)
corr_braveheart.head()

corr_forest_gump = corr_forest_gump.join(averageRatingsDf['numberOfRatings'])
corr_braveheart = corr_braveheart.join(averageRatingsDf['numberOfRatings'])

#forest_gump_recommended_movies = corr_forest_gump[corr_forest_gump['numberOfRatings'] > 100 ].sort_values(by='Correlation', ascending=False).head(10)

#braveheart_recommended_movies = corr_braveheart[corr_braveheart['numberOfRatings'] > 100 ].sort_values(by='Correlation', ascending=False).head(10)