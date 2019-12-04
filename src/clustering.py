# General tools for clustering including K_mean, KNN, etc.
# Also some tools for PCA
#

import numpy as np
import pandas as pd


def get_distances(df: pd.DataFrame, labels: np.ndarray, centers: np.ndarray,
                  metric: str='Euclidean') -> pd.DataFrame:
    """Compute centers to clusters for each observation

    Args:
        df_in: Input matrix to generate clusters
        labels: Returned matrix of labels from clustering algorithm.
            This is typically generated as cluster.cluster_centers_
        centers: array of center coordinates that K_means generated
        metric: Which norm to use when computing distance

    """
    df_ret = df.copy()
    clusters = centers.shape[1]

    df_ret['dist_to_cc'] = [
        np.linalg.norm(centers[label] - df.loc[i, :])
        for i, label in enumerate(labels)
    ]
    return df_ret


def compute_distortion(df: pd.DataFrame):
    """
    :param df_in:
    :return:
    """
    distortion = np.mean(df['dist_to_cc'])
    return distortion

