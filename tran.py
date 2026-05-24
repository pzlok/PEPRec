import pandas as pd


def replace_elements(lst, replacements):
    return [replacements.get(item, item) for item in lst]


df_ex = pd.read_csv('./exercise.csv')
df_data = pd.read_csv('./merge_data.csv')

df_data.sample(frac=1)

# 韩英对照字典
ex_dict = df_ex[["KO_name", "EN_name"]].set_index("KO_name").to_dict(orient='dict')["EN_name"]

# 韩中对照字典
# ex_dict = df_ex[["KO_name", "ZH_name"]].set_index("KO_name").to_dict(orient='dict')["ZH_name"]

pr_list = df_data['exercise prescription'].to_list()

tran_list = []
for pr in pr_list:
    new_list = replace_elements(pr.split(','), ex_dict)
    tran_list.append(','.join(new_list))

df_data['exercise prescription'] = tran_list

# 保存翻译成英文的数据
df_data.to_csv('./tran_data_EN.csv', index=False, encoding='utf-8')

# 保存翻译成中文的数据
# df_data.to_csv('./tran_data_ZH.csv', index=False, encoding='utf-8')
