from cProfile import label
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode

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
    df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f'{label_column} == "{label}"')
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend(prop={'size': 5})
    ax.ticklabel_format(style='plain')
    plt.savefig(f"practica 8/plots/knn.png")
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]


df = get_df()
df = df[["Gain", "Peak_Players","Game_Name"]].head(1000)
points = list(df[["Gain", "Peak_Players"]].to_records(index=False))
labels = df["Game_Name"].tolist()
scatter_group_by(df, "Gain", "Peak_Players", "Game_Name")

list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([900, 450]), np.array([1000, 1000]), np.array([40404, 30003]), np.array([80000, 403500])],
    5,
)
print(kn)