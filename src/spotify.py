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

    def get_recommendations(self, track_title, amount=1):

        # Notify the user of thre process
        print("Getting recommended songs based on " + track_title + "...")

        # normalize the data
        self.normalize_data()

        distance = []
        song = (
            self.dataset[(self.dataset.name.str.lower() == track_title.lower())]
            .head(1)
            .values[0]
        )
        rec = self.dataset[self.dataset.name.str.lower() != track_title.lower()]

        for songs in tqdm(rec.values):
            d = 0
            for column in np.arange(len(rec.columns)):
                if not column in [
                    0,
                    1,
                    3,
                    4,
                    5,
                    6,
                    7,
                    13,
                    19,
                ]:  # Filter out unwanted columns
                    d = d + np.absolute(float(song[column]) - float(songs[column]))

            distance.append(d)

        rec["distance"] = distance
        rec = rec.sort_values("distance")
        columns = ["artists", "name"]

        return rec[columns][:amount]

    def get_recommendations_byId(self, track_id, amount=1):

        # Notify the user of thre process
        print("Getting recommended songs based on Track Id: " + track_id + "...")

        # normalize the data
        self.normalize_data()

        distance = []
        song = (
            self.dataset[(self.dataset.id.str.lower() == track_id.lower())]
            .head(1)
            .values[0]
        )
        rec = self.dataset[self.dataset.id.str.lower() != track_id.lower()]

        for songs in tqdm(rec.values):
            d = 0
            for column in np.arange(len(rec.columns)):
                if not column in [
                    0,
                    1,
                    3,
                    4,
                    5,
                    6,
                    7,
                    13,
                    19,
                ]:  # Filter out unwanted columns
                    d = d + np.absolute(float(song[column]) - float(songs[column]))

            distance.append(d)

        rec["distance"] = distance
        rec = rec.sort_values("distance")
        columns = ["artists", "name", "id"]

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
        norms = self.dataset.select_dtypes(include=datatypes)
        for col in norms.columns:
            MinMaxScaler(col)

        kmeans = KMeans(n_clusters=10)
        features = kmeans.fit_predict(norms)
        self.dataset["features"] = features
        MinMaxScaler(self.dataset["features"])
