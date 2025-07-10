import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from cdlib.evaluation.internal.statistical_ranking import friedman_test,bonferroni_dunn_test


def res_load(csv_name: str):
    data = pd.read_csv(csv_name, sep=";").T
    data.columns = data.loc["metrics"]
    data.drop(index="metrics", inplace=True)
    if 'Hits@10' in data.columns:
        data.drop(columns="Hits@10", inplace=True)
    
    for idx in data.index:
        if data.loc[idx].dtype == 'object' and '_mean' not in idx:
            try:
                data.loc[idx] = data.loc[idx].map(lambda x: np.array(list(map(float, x.strip('[]').split()))))
            except ValueError as e:
                print(e)
    data = data.fillna(0)
    return data

def plot_epochs(data, title):

    x = np.array([0, 10, 30])
    for method in data.keys():
        plt.plot(x, data[method], label=method)

    plt.title(title)
    plt.xlabel('CT_EPOCHS')
    plt.ylabel('ROCAUC')
    plt.legend()

    # Affichage du plot
    plt.show()


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
    fig, ax = plt.subplots(figsize=size, dpi=400)
    im = ax.imshow(normalized_data)

    plt.xticks(rotation=90, fontsize=14)

    ax.set_yticks(
        range(len(data.index)),
        labels=data.index,
        fontsize=14,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xticks(range(len(data.columns)), labels=data.columns,rotation=45,
        ha="right",
        rotation_mode="anchor")

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
                    fontsize=11
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{data.iloc[i, j]}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11
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


def compute_significantly_top_methods(df, metric: str = "ROCAUC"):

    unique_methods = df.columns.unique()
    all_scores_by_algorithm=[]

    # print("-------df",df.columns,df)
    to_remove = []
    for algo in unique_methods:
        if isinstance(df.loc[metric][algo], np.ndarray):
            all_scores_by_algorithm.append([score for score in df.loc[metric][algo]])
        else:
            to_remove.append(algo)
    unique_methods = unique_methods.drop(to_remove, errors='ignore')
    #print("size data",len(df),len(all_scores_by_algorithm[0]))
    #print("check",len(df),sorted_methods,all_scores_by_algorithm)
    # print("scores",all_scores_by_algorithm)
    try:
        f_value,p_value, rankings, pivots = friedman_test(*all_scores_by_algorithm)
    except:
        print("problem with the samples")
        print("methods",unique_methods)
        print("all_scores_by_algorithm",[len(a) for a in all_scores_by_algorithm])
        raise 
    labeled_ranks = {algo:rank for algo,rank in zip(unique_methods,rankings)}    
    
    sorted_methods = [x for _,x in sorted(zip(rankings,unique_methods), reverse=True)]
    best = sorted_methods[0]
    worst = sorted_methods[-1]
    #print("best",best,"worst",worst)    
 
    comparisons, z_values, p_values, adj_p_values = bonferroni_dunn_test(labeled_ranks,best)
    compared_to_list = [pair.split(" vs ")[1] for pair in comparisons]
    tops=[best]
    tops+=[compared_to_list[i] for i in range(len(comparisons)) if p_values[i]>0.1]    
    
    comparisons, z_values, p_values, adj_p_values = bonferroni_dunn_test(labeled_ranks,worst)
    #print("worst",comparisons,p_values)
    worsts=[worst]
    compared_to_list = [pair.split(" vs ")[1] for pair in comparisons]
    worsts+=[compared_to_list[i] for i in range(len(comparisons)) if p_values[i]>0.1]
    return tops,worsts,sorted_methods

def compare_methods(names: list, metric):
    res_met = {}
    for name in names:
        data = res_load(name)
        for method in data.index:
            if method not in res_met.keys():
                res_met[method] = {}
            for metric in data.columns:
                if metric not in res_met[method].keys():
                    res_met[method][metric] = []
                res_met[method][metric].append(extract_mean(data[metric].loc[method]))
    all_data = pd.DataFrame(res_met)
    tops,worsts,sorted_methods = compute_significantly_top_methods(all_data)
    return all_data, tops, worsts, sorted_methods

def tex_table(means, data= None):
    # rename means results
    for name in means.columns:
        if '_mean' in name:
            new_name = name.split('_mean')[0]
            means = means.rename(columns={name: new_name})
    for metric in means.index:
        if metric == 'pretrain_time':
            continue
        if data is not None:
            tops,worsts,sorted_methods = compute_significantly_top_methods(data, metric)
            for name in data.columns:
                if name in tops:
                    means.loc[metric, name] += " $\\bigstar$"
                elif name in worsts:
                    means.loc[metric, name] += ' X'
        tmp = means.loc[metric].apply(extract_mean)
        max_index = tmp.nlargest(3).index
        means.loc[metric, max_index[0]] = "\color{red}{"+means.loc[metric, max_index[0]]+"}"
        means.loc[metric, max_index[1]] = "\color{blue}{"+means.loc[metric, max_index[1]]+"}"
        means.loc[metric, max_index[2]] = "\color{violet}{"+means.loc[metric, max_index[2]]+"}"
        
    means.columns = [col.replace('_', ' ').replace('&', '+') for col in means.columns]
    latex = means.T.to_latex(
        index=True, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    print("\\resizebox{\\textwidth}{!}{")
    print(latex)
    print('}')
    return means

def mix_dataset(names, metric, filtre = None, filtre2 = None):
    dataset = []
    for_significant = []
    for name in names:
        data = res_load(name)[metric]
        data, mean = split_methods(data, '_mean')
        dataset.append(mean)
        for_significant.append(data)
    import pandas as pd
    all_data = pd.concat(dataset, keys=names, axis=1)
    all_res = pd.concat(for_significant, keys=names, axis=1)
    if filtre is not None:
        pattern = '|'.join(filtre)
        all_data = all_data[all_data.index.str.contains(pattern)]
        all_res = all_res[all_res.index.str.contains(pattern)]
    if filtre2 is not None:
        pattern = '|'.join(filtre2)
        all_data = all_data[~all_data.index.str.contains(pattern)]
        all_res = all_res[~all_res.index.str.contains(pattern)]
    return all_data.T, all_res.T