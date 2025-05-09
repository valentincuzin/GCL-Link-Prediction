import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


def res_load(csv_name: str):
    data = pd.read_csv("processed_csv/" + csv_name, sep=";").T
    data.columns = data.loc["metrics"]
    data.drop(index="metrics", inplace=True)
    return data


def extract_mean(value):
    if isinstance(value, str):
        return float(value.split("$\\pm$")[0])
    else:
        return value


def jaccard_sim(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    # data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    return jaccard_score([x > 0.5 for x in data1], [x > 0.5 for x in data2])


def jaccard_heatmap(data):
    data = data[~data.index.str.contains("NCN")]
    res = np.zeros((len(data), len(data)))
    for idx in range(len(data)):
        for idx2 in range(len(data)):
            if idx < idx2:
                res[idx][idx2] = jaccard_sim(data.iloc[idx], data.iloc[idx2])
    res += np.transpose(res) + np.identity(len(data))
    heatmap(
        pd.DataFrame(res, index=data.index, columns=data.index),
        "Corelation Overlap",
        False,
    )


def heatmap(
    data: pd.DataFrame,
    Title: str,
    size: tuple[int, int] = (12, 12),
    variance=False,
    normalized=True,
):
    data = data.copy()
    data: pd.DataFrame = data.T
    if "pretrain_time" in data.index:
        data = data.drop("pretrain_time")
    if "pretrain_time" in data.columns:
        data = data.drop("pretrain_time", axis=1)
    if "Hits@10" in data.index:
        data = data.drop("Hits@10")
    if "Hits@10" in data.columns:
        data = data.drop("Hits@10", axis=1)
    data_var = data.copy()
    for column in data.columns:
        data[column] = data[column].apply(extract_mean).apply(np.mean)
    if normalized:
        normalized_data = (data - data.min(axis=1).values[:, np.newaxis]) / (
            data.max(axis=1).values[:, np.newaxis]
            - data.min(axis=1).values[:, np.newaxis]
        )
    else:
        normalized_data = data
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(normalized_data)

    plt.xticks(rotation=90)

    ax.set_yticks(
        range(len(data.index)),
        labels=data.index,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xticks(range(len(data.columns)), labels=data.columns)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            text_color = (
                "white"
                if normalized_data.iloc[i, j] < np.mean(normalized_data)
                else "black"
            )
            text = None
            if variance:
                text = ax.text(
                    j,
                    i,
                    f"{data_var.iloc[i, j]}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{data.iloc[i, j]}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    ax.set_title(Title)
    fig.tight_layout()
    plt.show()


def average_metrics(data):
    data: pd.DataFrame = data.copy()
    time = None
    if "pretrain_time" in data.index:
        time = data["pretrain_time"]
        data = data.drop("pretrain_time")
    for column in data.columns:
        data[column] = data[column].apply(extract_mean).apply(np.mean)
    data["avg"] = data.mean(axis=1)
    if time is not None:
        data["pretrain_time"] = time
    return data


def average_methods(data):
    data = data.copy()
    data = data.T
    if "pretrain_time" in data.index:
        data = data.drop("pretrain_time")
    for column in data.columns:
        data[column] = data[column].apply(extract_mean).apply(np.mean)
    return data.mean(axis=1)


def split_methods(data, sep: str):
    data_1 = data[~data.index.str.contains(sep, case=False)]
    data_2 = data[data.index.str.contains(sep, case=False)]
    return data_1, data_2
