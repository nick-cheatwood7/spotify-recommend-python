# Import needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom module
import spotify as sp


def main():
    # Init seaborn
    sns.set()

    # Read in data
    data = pd.read_csv("./data/tracks.csv")

    # Get metrics on the data
    # data.info()

    # Test for empty values
    data.isnull().sum()

    # Clean the data
    data["name"].fillna("Unknown Title", inplace=True)

    # Test for empty values again
    data.isnull().sum()

    # Normalize the data
    recommender = sp.SpotifyRecommender(data)
    print(recommender.get_recommendations("Re: Stacks", 20))


if __name__ == "__main__":
    main()
