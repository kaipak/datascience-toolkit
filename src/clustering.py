# General tools for clustering including K_mean, KNN, etc.
# Also some tools for PCA
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans


def eval_KMeans(df: pd.DataFrame, columns: list=None, clust_min: int=1,
                clust_max: int=4):
    """ Take a dataframe and show diags on finding right number of clusters
    that minimizes distortion.

    Args:
        df: Input dataframe
        columns: Which columns to use for clustering. If none given, then 
            default to using all columns.
        clust_min: 
    Returns:
        void? 

    """
    if columns is None:
        columns = df.columns

    num_clusters = [int(n) for n in range(clust_min, clust_max + 1)]
    mean_distances = []

    for n in num_clusters:
        cluster = KMeans(n_clusters=n)
        cluster.fit(df[columns])
        mean_distances.append(np.mean(get_distances(df, cluster)))

    sns.lineplot(num_clusters, mean_distances)
    return(mean_distances)


def get_distances(df: pd.DataFrame, cluster: KMeans,
                  metric: str='Euclidean') -> pd.DataFrame:
    """Compute centers to clusters for each observation

    Args:
        df: Input dataframe or matrix to generate clusters
        cluster: sklearn KMeans object
        metric: Which norm to use when computing distance

    """
    centers = cluster.cluster_centers_
    labels = cluster.labels_

    distances = [
        np.linalg.norm(centers[label] - df.loc[i, :])
        for i, label in enumerate(labels)
    ]
    return distances


def compute_distortion(df: pd.DataFrame):
    """
    :param df_in:
    :return:
    """
    distortion = np.mean(df['dist_to_cc'])
    return distortion

