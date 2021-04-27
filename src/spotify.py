# Standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Packages for clustering and cleaning
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

"""
Some features/functions inspired by: https://thecleverprogrammer.com/2021/03/03/spotify-recommendation-system-with-machine-learning/
"""


class SpotifyRecommender:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_recommendations(self, songs, amount=1):

        # normalize the data
        self.normalize_data()

        distance = []
        song = (
            self.dataset[(self.dataset.name.str.lower() == songs.lower())]
            .head(1)
            .values[0]
        )
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]

        for songs in tqdm(rec.values):
            d = 0
            for column in np.arange(len(rec.columns)):
                if not column in [0, 1, 5, 6, 7]:
                    d = d + np.absolute(float(song[column]) - float(songs[column]))

            distance.append(d)

        rec["distance"] = distance
        rec = rec.sort_values("distance")
        columns = ["artists", "name"]

        return rec[columns][:amount]

    def extract_features(self, dataset):
        features = dataset.drop(
            columns=["id", "name", "artists", "id_artists", "release_date"]
        )
        return features

    def get_feature_correlations(self, features):
        return features.corr()

    def normalize_data(self):
        # Normalize data types in the dataset
        datatypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
        normarization = self.dataset.select_dtypes(include=datatypes)
        for col in normarization.columns:
            MinMaxScaler(col)

        kmeans = KMeans(n_clusters=10)
        features = kmeans.fit_predict(normarization)
        self.dataset["features"] = features
        MinMaxScaler(self.dataset["features"])
