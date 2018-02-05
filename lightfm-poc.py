import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import sys

dataset = fetch_movielens(min_rating=4.0)

#create & train model
model = LightFM(loss='bpr') # logistic, warp, bpr, warp-kos
model.fit(dataset['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:

        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        # print(np.argsort(-scores))
        top_items = data['item_labels'][np.argsort(-scores)]
        # print(top_items)
        # print("User ", user_id)
        print("Movies user %s liked:" % user_id)

        for x in known_positives[:3]:
            print("\t%s" % x)

        print("Movies user %s might like:" % user_id)

        for x in top_items[:3]:
            print("\t%s" % x)
            
users = []
for id in sys.argv:
    if sys.argv.index(id) != 0:
        users.insert(sys.argv.index(id) - 1, id)

sample_recommendation(model, dataset, users) # 219, 118, 36