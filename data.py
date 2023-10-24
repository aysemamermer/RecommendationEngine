import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None)

df = pd.read_parquet("train_final.parquet")

melted_df = df.melt(id_vars=['id', 'month', 'carrier',
                            'devicebrand', 'feature_0', 'feature_1', 'feature_2', 'feature_3',
                            'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',
                            'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13',
                            'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18',
                            'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23',
                            'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28',
                            'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33',
                            'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38',
                            'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43',
                            'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48',
                            'feature_49'], value_vars=['n_seconds_1', 'n_seconds_2', 'n_seconds_3'],
                    var_name='seconds', value_name='value')


target_df = df['target'].str.split(', ', expand=True)

result_df = melted_df.copy()

for i in range(1, 4):
    column_name = f'n_seconds_{i}'
    target_name = f'target_{i}'
    result_df[target_name] = target_df[i - 1]
    result_df.loc[result_df['seconds'] == column_name, 'target'] = result_df.loc[result_df['seconds'] == column_name, target_name]

result_df = result_df[result_df['seconds'] != 'seconds']

print(df.head(10))

result_df.drop(['target_1', 'target_2', 'target_3','seconds'], axis=1, inplace=True)

print(result_df.head(10))


non_numeric_columns = []
for column in result_df.columns:
    if result_df[column].dtype == 'object':  # Veri tipi 'object' ise (kategorik)
        non_numeric_columns.append(column)

label_encoder = LabelEncoder()

result_df = pd.get_dummies(result_df, columns=non_numeric_columns)

print("new column types")
print(result_df.dtypes)

print(result_df.head(10))



