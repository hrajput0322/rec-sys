#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
import pprint
import tempfile
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import time
from tensorflow.keras import regularizers
masterdf = pd.read_csv('Final.csv')
### standardize item data types, especially string, float, and integer

masterdf[['user_id',      
          'product_id',  
         ]] = masterdf[['user_id','product_id']].astype(str)

# we will play around with the data type of the quantity, 
# which you shall see later it affects the accuracy of the prediction.

masterdf['quantity'] = masterdf['quantity'].astype(float)
### define interactions data and user data

### interactions 
### here we create a reference table of the user , item, and quantity purchased
interactions_dict = masterdf.groupby(['user_id', 'product_id', 'timestamp'])[ 'quantity'].sum().reset_index()

## we tansform the table inta a dictionary , which then we feed into tensor slices
# this step is crucial as this will be the type of data fed into the embedding layers
interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

## we do similar step for item, where this is the reference table for items to be recommended
items_dict = masterdf[['product_id']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

## map the features in interactions and items to an identifier that we will use throught the embedding layers
## do it for all the items in interaction and item table
## you may often get itemtype error, so that is why here i am casting the quantity type as float to ensure consistency
interactions = interactions.map(lambda x: {
    'user_id' : x['user_id'], 
    'product_id' : x['product_id'], 
    'quantity' : float(x['quantity']),
        "timestamp": x["timestamp"]
})

items = items.map(lambda x: x['product_id'])
## Basic housekeeping to prepare feature vocabularies

## timestamp is an exmaple of continuous features, which needs to be rescaled, or otherwise it will be 
## too large for the model.
## there are other methods to reduce the size of the timestamp, ,such as standardization and normalization
## here we use discretization, which puts them into buckets of categorical features, 

timestamps = np.concatenate(list(interactions.map(lambda x: x["timestamp"]).batch(100)))
max_timestamp = timestamps.max()
min_timestamp = timestamps.min()
timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,)

item_titles = interactions.batch(10_000).map(lambda x: x["product_id"])
user_ids = interactions.batch(10_000).map(lambda x: x["user_id"])

unique_item_titles = np.unique(np.concatenate(list(item_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(60_000)
test = shuffled.skip(60_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()
class Model(tfrs.models.Model):

    def __init__(self,
                 rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 64

        # item models.
        self.item_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_item_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
        ])
            
        ## user model    
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # A small model to take in user and item embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        #,kernel_regularizer=regularizers.l2(0.01)
        
        ## this is Relu-Based DNN
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # rating and retrieval task.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
            
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items.batch(128).map(self.item_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        
        # And pick out the item features and pass them into the item model.
        item_embeddings = self.item_model(features["product_id"])

        return (
            user_embeddings,
            item_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and item embeddings.
            self.rating_model(
                tf.concat([user_embeddings, item_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ## ratings go here as a method to compute loss
        ratings = features.pop("quantity")

        user_embeddings, item_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)
model = Model(rating_weight=1, retrieval_weight=1)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
def predict_movie(user, top_n=3):
    # Create a model that takes in raw query features
    index = tfrs.layers.factorized_top_k.BruteForce(model.item_model)

    # Generate embeddings for the unique item titles
    item_embeddings = model.item_model(unique_item_titles)

    # Index the embeddings with the item titles as candidates
    index.index(item_embeddings)

    # Get recommendations.
    _, indices = index(tf.constant([str(user)]))
    titles = unique_item_titles[indices[0, :top_n].numpy()]

    print('Top {} recommendations for user {}:\n'.format(top_n, user))
    for i, title in enumerate(titles):
        print('{}. {}'.format(i+1, title.decode("utf-8")))

# Example usage
predict_movie(user="d97b3cfb22b0d6b25ac9ed4e9c2d481b", top_n=10)

