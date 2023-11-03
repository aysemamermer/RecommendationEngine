
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#%%------------------------------Read data------------------------------

tqdm.pandas()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_df = pd.read_parquet('train_final.parquet')
print("Training Data Shape:", train_df.shape)

test_df = pd.read_parquet('test_final.parquet')
print("Test Data Shape:", test_df.shape)

ss = pd.read_parquet('submission_sample_final.parquet')

train_df["split"] = "train"
test_df["split"] = "test"

#%%------------------------------Defination of columns------------------

feat_cols = [col for col in train_df.columns if "feature_" in col]
sec_cols = [col for col in train_df.columns if "n_seconds" in col]
cat_cols = ['carrier', 'devicebrand']
num_cols = sec_cols + feat_cols
menu_cols = [f"menu{i}" for i in range(1,10)]




train_df["target"] = train_df["target"].str.replace("menu", "")\
        .str.split(",").apply(lambda x: [int(elm)-1 for elm in x])
def create_menu_multilabel_oh(target_list):
    out = np.zeros(9)
    for menuid in target_list:
        out[menuid] = 1
    return out.tolist()

test_df[menu_cols] = 0
train_df[menu_cols] = 0

train_df[menu_cols] = train_df["target"].apply(lambda x: create_menu_multilabel_oh(x)).to_list()


def create_menu_sec_cols(inp_arr):
    arr = np.zeros(9)
    for sec_info, one_indice in zip(
        [
            inp_arr["n_seconds_1"],
            inp_arr["n_seconds_2"],
            inp_arr["n_seconds_3"]
        ],
        inp_arr["target"]):
        arr[one_indice] = sec_info
    return arr

menu_sec_cols = [f"menu{i}_secs" for i in range(1,10)]
return_arr = train_df[["n_seconds_1", "n_seconds_2", "n_seconds_3", "target"]].apply(create_menu_sec_cols, 1)
train_df[menu_sec_cols] = np.stack(return_arr, axis=0)

print(train_df.head())
