import pandas as pd
import numpy as np


df = pd.read_csv('./train_data/adult_select.csv')
df_add = df.copy(deep=True)

# 设置高斯噪声的均值和标准差
mean_noise = 0  # 均值
std_deviation_noise1 = 0.5  # 标准差
std_deviation_noise2 = 1.0  # 标准差
std_deviation_noise3 = 0.01  # 标准差

for i in range(2):
    # 添加高斯白噪声
    noise1 = np.random.normal(0, std_deviation_noise1, [len(df), 9])
    noise2 = np.random.normal(0, std_deviation_noise2, [len(df), 4])
    noise3 = np.random.normal(0, std_deviation_noise3, [len(df), 1])

    for ind, row in df.iterrows():
        if row['height_cm'] != 0:
            row['height_cm'] = np.around(row['height_cm'] + noise1[ind][0], decimals=1)
        if row['weight_kg'] != 0:
            row['weight_kg'] = np.around(row['weight_kg'] + noise1[ind][1], decimals=1)
        if row['body fat_%'] != 0:
            row['body fat_%'] = np.around(row['body fat_%'] + noise1[ind][2], decimals=1)
        if row['waistline_cm'] != 0:
            row['waistline_cm'] = np.around(row['waistline_cm'] + noise1[ind][3], decimals=1)
        if row['diastolic_mmHg'] != 0:
            row['diastolic_mmHg'] = np.around(row['diastolic_mmHg'] + noise2[ind][0], decimals=0)
        if row['systolic_mmHg'] != 0:
            row['systolic_mmHg'] = np.around(row['systolic_mmHg'] + noise2[ind][1], decimals=0)
        if row['left grip_kg'] != 0:
            row['left grip_kg'] = np.around(row['left grip_kg'] + noise1[ind][4], decimals=1)
        if row['right grip_kg'] != 0:
            row['right grip_kg'] = np.around(row['right grip_kg'] + noise1[ind][5], decimals=1)
        row['sit and bend forward_cm'] = np.around(row['sit and bend forward_cm'] + noise1[ind][6], decimals=1)
        row['BMI_kg/㎡'] = round(row['weight_kg'] / ((row['height_cm'] / 100) ** 2), 1)
        if row['cross sit-ups_times'] != 0:
            row['cross sit-ups_times'] = np.around(row['cross sit-ups_times'] + noise2[ind][2], decimals=0)
        if row['relative grip_%'] != 0:
            row['relative grip_%'] = np.around(row['relative grip_%'] + noise1[ind][7], decimals=1)
        if row['step test heart rate_bpm'] != 0:
            row['step test heart rate_bpm'] = np.around(row['step test heart rate_bpm'] + noise2[ind][3], decimals=0)
        if row['step test oxygen uptake_VO₂max'] != 0:
            row['step test oxygen uptake_VO₂max'] = np.around(row['step test oxygen uptake_VO₂max'] + noise1[ind][8], decimals=1)
        if row['reaction time_s'] != 0:
            row['reaction time_s'] = np.around(row['reaction time_s'] + noise3[ind][0], decimals=3)
        df_add.loc[len(df_add)] = row
        if ind % 10 == 0:
            print(ind)

print('-----------------------')
print(len(df_add))
# 数据去重
subset = df_add.columns[:-1]
df_add.drop_duplicates(subset=subset, keep='first', inplace=True)
print(len(df_add))
# 打乱数据
df_add = df_add.sample(frac=1)
df_add.to_csv('./train_data/adult_noise.csv', index=False, encoding='utf-8')
