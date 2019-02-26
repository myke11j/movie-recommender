
# coding: utf-8

# In[24]:


# Import necessary libraries

import pandas as pd 
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print('Pandas version', pd.__version__)
print('Numpy version', np.__version__)
print('Seaborn version', sns.__version__)


# In[25]:


# Read datasets from CSVs
df = pd.read_csv('datasets/ml-latest-small/ratings.csv')
movieTitles = pd.read_csv('datasets/ml-latest-small/movies.csv')


# In[26]:


df.head(10)


# In[27]:


df.info()


# In[28]:


movieTitles.head(10)


# In[29]:


movieTitles.info()


# In[30]:


df = pd.merge(df, movieTitles, on='movieId')
df.head(10)


# We will create a new dataframe which will have rating for title and number of ratings it received.b

# In[57]:


averageRatingsDf = pd.DataFrame(df.groupby('title')['rating'].mean())
averageRatingsDf['numberOfRatings'] = df.groupby('title')['rating'].count()
averageRatingsDf.head(10)


# In[58]:


averageRatingsDf = averageRatingsDf.sort_values(['numberOfRatings', 'rating'], ascending=False)
averageRatingsDf.head(15)


# In[59]:


sns.jointplot(x='rating', y='numberOfRatings', data=averageRatingsDf)


# The above scatter plot clarly shows, that avergae rating of movies goes up with more number of ratings.

# In[60]:


# Data engineering
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()


# Now, we will choose two movies, for which we will recommend movies.
# I've choosen
# 
# - Forrest Gump
# - Braveheart

# In[61]:


forest_gump_user_rating = movie_matrix['Forrest Gump (1994)']
braveheart_user_rating = movie_matrix['Braveheart (1995)']


# In[62]:


forest_gump_user_rating.head()


# In[63]:


braveheart_user_rating.head()


# In[64]:


# Find correlation between watched movies and all movies
similar_to_forest_gump = movie_matrix.corrwith(forest_gump_user_rating)
similar_to_braveheart = movie_matrix.corrwith(braveheart_user_rating)


# In[65]:


similar_to_forest_gump.head()


# In[66]:


similar_to_braveheart.head()


# In[67]:


corr_forest_gump = pd.DataFrame(similar_to_forest_gump, columns=['Correlation'])
corr_forest_gump.dropna(inplace=True)
corr_forest_gump.head()


# In[68]:


corr_braveheart = pd.DataFrame(similar_to_braveheart, columns=['correlation'])
corr_braveheart.dropna(inplace=True)
corr_braveheart.head()


# In[69]:


corr_forest_gump = corr_forest_gump.join(averageRatingsDf['numberOfRatings'])
corr_braveheart = corr_braveheart.join(averageRatingsDf['numberOfRatings'])


# In[72]:


rec1 = corr_braveheart[corr_braveheart['numberOfRatings'] > 50 ].sort_values(by='correlation', ascending=False).head(10)


# In[74]:


rec2 = corr_forest_gump[corr_forest_gump['numberOfRatings'] > 50 ].sort_values(by='Correlation', ascending=False).head(10)

print(rec1);
print(rec2);