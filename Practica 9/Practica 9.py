import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode
from sklearn.cluster import KMeans


def drop_column(df, column_names) -> pd.DataFrame:
    df = df.drop(column_names, axis = 1)
    return df


def transform_percent(str_percent:str) -> float:
    return float(str(str_percent).strip('%'))/100


def get_df() -> pd.DataFrame:
    df = pd.read_csv(".\Practica 1\Valve_Player_Data.csv")

    df = drop_column(df, ["Month_Year"])
    df["Date"] = pd.to_datetime(df["Date"], format = '%Y-%m-%d')
    df['Percent_Gain'] = df['Percent_Gain'].apply(transform_percent)

    return df


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f'{label_column} == "{label}"')
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend(prop={'size': 5})
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_means(points: List[np.array], k: int):
    DIM = len(points[0])
    N = len(points)
    num_cluster = k
    iterations = 15

    x = np.array(points)
    y = np.random.randint(0, num_cluster, N)

    mean = np.zeros((num_cluster, DIM))
    for t in range(iterations):
        for k in range(num_cluster):
            mean[k] = np.mean(x[y == k], axis=0)
        for i in range(N):
            dist = np.sum((mean - x[i]) ** 2, axis=1)
            pred = np.argmin(dist)
            y[i] = pred

    for kl in range(num_cluster):
        xp = x[y == kl, 0]
        yp = x[y == kl, 1]
        plt.scatter(xp, yp)
    plt.savefig("practica 9/plots/k_means.png")
    plt.close()
    return mean


df = get_df()
df = df[["Gain", "Peak_Players","Game_Name"]].head(1000)
points = list(df[["Gain", "Peak_Players"]].to_records(index=False))
labels = df["Game_Name"].tolist()
scatter_group_by("practica 9/plots/clustering.png", df, "Gain", "Peak_Players", "Game_Name")

list_t = [
    (np.array(tuples[0:2]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]
# np.random.seed(0)
kn = k_means(
    points,
    3,
)
print(kn)