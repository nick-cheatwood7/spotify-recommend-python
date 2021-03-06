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

    # Init the recommender instance
    recommender = sp.SpotifyRecommender(data)

    # Get recommendation by Track title
    print(recommender.get_recommendations("Re: Stacks", 20))

    # Get recommendation by Track Id
    print(recommender.get_recommendations_byId("2LthqyP0MLhGUBICwR1535", 20))


if __name__ == "__main__":
    main()
