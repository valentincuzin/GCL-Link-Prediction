import pandas as pd
import numpy as np

def compute_table(res_dict: dict[str, list | float], name: str):
    # Compute the mean and std from a dict return tab and latex table
    new_tab = []
    for key, result in res_dict.items():
        print(key)
        if key == "test_pred":
            new_tab.append({"metrics": key, name: result})
        elif isinstance(result, list):
            result = np.array(result)
            unit = 100 if key != 'pretrain_time' else 1
            mean = round(unit * np.mean(result), 2)
            std = round(unit * np.std(result), 2)
            new_tab.append({"metrics": key, name: fr"{mean}$\pm${std}"})
    df = pd.DataFrame(data=new_tab)
    df.set_index('metrics')
    res_latex = df.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return df, res_latex

def full_output(full_res: list):
    # concat full result to one dataframe, then print with latex
    full_res = pd.concat(full_res, axis=1)
    full_res = full_res.loc[:, ~full_res.columns.duplicated()]
    full_res.set_index('metrics', inplace=True)
    print(full_res)
    full_latex = full_res.to_latex(
        index=True, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    print(full_latex)
    return full_res, full_latex
