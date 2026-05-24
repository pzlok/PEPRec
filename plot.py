import math

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots


# 画出男女等级直方图
def plot_grade_gender(df):
    # 男、女 A、B、C、D等级个数
    woman_a = len(df[(df['score class'] == 'A') & (df['gender'] == 0)])
    woman_b = len(df[(df['score class'] == 'B') & (df['gender'] == 0)])
    woman_c = len(df[(df['score class'] == 'C') & (df['gender'] == 0)])
    woman_d = len(df[(df['score class'] == 'D') & (df['gender'] == 0)])
    man_a = len(df[(df['score class'] == 'A') & (df['gender'] == 1)])
    man_b = len(df[(df['score class'] == 'B') & (df['gender'] == 1)])
    man_c = len(df[(df['score class'] == 'C') & (df['gender'] == 1)])
    man_d = len(df[(df['score class'] == 'D') & (df['gender'] == 1)])
    men = (man_a, man_b, man_c, man_d)
    women = (woman_a, woman_b, woman_c, woman_d)
    fig, ax = plt.subplots()
    index = np.arange(4)
    bar_width = 0.4
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, men, bar_width, alpha=opacity, color='b', error_kw=error_config, label='Male')
    rects2 = plt.bar(index + bar_width, women, bar_width, alpha=opacity, color='r', error_kw=error_config,
                     label='Female')
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.xlabel('score class')
    plt.ylabel('number')
    # plt.title('图表标题')
    plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D'))
    plt.ylim(0, 4000)  # 设置y轴的标尺
    plt.xlim(-0.5, 4)  # 设置x轴的标尺
    plt.legend()
    # plt.show()
    fig.savefig('./result/plot/男女等级直方图1.jpg', dpi=600)  # 保存图片并设置分辨率


# df = pd.read_csv('./train_data/adult_noise.csv')
df = pd.read_csv('./train_data/adult_select.csv')

# 画男女等级条形图
plot_grade_gender(df)

# # 画热力图
# # 设置画板尺寸
# plt.subplots(figsize=(13, 11))
# df1 = df.drop(['age', 'gender', 'score class', 'height_cm', 'weight_kg', 'relative grip_%', 'exercise prescription'], axis=1, inplace=False)
# d = df1.corr()
# cols = ['body_fat', 'waistline', 'diastolic', 'systolic', 'L_grip', 'R_grip', 'sit_bend', 'BMI', 'cross_sit-ups', 'heart_rate', 'oxygen_uptake', 'reaction_time']
# sns.heatmap(d,
#             cmap="YlGnBu",
#             cbar=True,
#             annot=True,  # 注入数字
#             square=True,  # 单元格为正方形
#             fmt='.2f',  # 字符串格式代码
#             annot_kws={'size': 12},  # 当annot为True时，ax.text的关键字参数，即注入数字的字体大小
#             yticklabels=cols,  # 列标签
#             xticklabels=cols  # 行标签
#             )
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.show()
# plt.savefig('./result/plot/相关性图.jpg', dpi=600)


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)

# 设置全局字体样式
# font = {'weight': 'bold', 'size': 12}
# matplotlib.rc('font', **font)
# # 定义参数
# mus = [0, 0, 0]
# sigmas = [0.5, 1, 2]
# # 生成x轴上的数值范围
# x = np.linspace(-6, 6, 200)
# pos = [107, 118, 145]
# # 绘制高斯分布曲线
# for mu, sigma, p in zip(mus, sigmas, pos):
#     y = gaussian_noise_kernel(x, mu, sigma)
#     plt.plot(x, y, linewidth=2)
#     label = f'sigma={sigma}'
#     plt.text(x[p], y[p], label, ha='left', va='center', fontweight='bold')
# # 添加标题和图例
# # plt.title('Gaussian Noise Curves')
# # plt.legend()
# plt.xlim(-6, 6)
# plt.ylim(0, plt.ylim()[1])
# # 去除横纵坐标值
# # plt.xticks([])  # 去除x轴标签
# plt.yticks([])  # 去除y轴标签
# # 设置图片框加粗
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(2)
# # 显示图形
# # plt.show()
# plt.savefig('./result/plot/高斯分布.jpg', dpi=600)

# x = np.linspace(-np.pi, np.pi, 200)
# y = np.sin(x)
# y_noise = np.random.normal(0, 0.15, [1, 200])
# y_with_noise = y + y_noise
# plt.plot(x, y, linewidth=2)
# # plt.scatter(x, y_with_noise, s=10, c='b')
# plt.ylim(-1.5, 1.5)
# # plt.show()
# plt.savefig('./result/plot/sin.jpg', dpi=600)
# # plt.savefig('./result/plot/sin_noise.jpg', dpi=600)


# plt.style.use('science')
#
# df_res = pd.read_csv('./result/res.csv')
# df_le = pd.read_csv('./result/LeNet.csv')
# df_alex = pd.read_csv('./result/AlexNet.csv')
# df_zf = pd.read_csv('./result/ZFNet.csv')
# df_vgg = pd.read_csv('./result/VGG.csv')

# loss_res = df_res['loss1'].values
# loss_le = df_le['loss1'].values
# loss_alex = df_alex['loss1'].values
# loss_zf = df_zf['loss1'].values
# loss_vgg = df_vgg['loss1'].values

# loss_res = df_res['y_recall1'].values
# loss_le = df_le['y_recall1'].values
# loss_alex = df_alex['y_recall1'].values
# loss_zf = df_zf['y_recall1'].values
# loss_vgg = df_vgg['y_recall1'].values
#
# res_max = [df_res['p6'].values.max(),
#            df_res['p7'].values.max(),
#            df_res['p8'].values.max(),
#            df_res['p9'].values.max(),
#            df_res['p10'].values.max(),
#            df_res['p11'].values.max(),
#            df_res['p12'].values.max()]
#
# le_max = [df_le['p6'].values.max(),
#           df_le['p7'].values.max(),
#           df_le['p8'].values.max(),
#           df_le['p9'].values.max(),
#           df_le['p10'].values.max(),
#           df_le['p11'].values.max(),
#           df_le['p12'].values.max()]
#
# alex_max = [df_alex['p6'].values.max(),
#             df_alex['p7'].values.max(),
#             df_alex['p8'].values.max(),
#             df_alex['p9'].values.max(),
#             df_alex['p10'].values.max(),
#             df_alex['p11'].values.max(),
#             df_alex['p12'].values.max()]
#
# zf_max = [df_zf['p6'].values.max(),
#           df_zf['p7'].values.max(),
#           df_zf['p8'].values.max(),
#           df_zf['p9'].values.max(),
#           df_zf['p10'].values.max(),
#           df_zf['p11'].values.max(),
#           df_zf['p12'].values.max()]
#
# vgg_max = [df_vgg['p6'].values.max(),
#            df_vgg['p7'].values.max(),
#            df_vgg['p8'].values.max(),
#            df_vgg['p9'].values.max(),
#            df_vgg['p10'].values.max(),
#            df_vgg['p11'].values.max(),
#            df_vgg['p12'].values.max()]
#
# x = np.arange(1, 8)
#
# with plt.style.context(['ieee']):
#     fig1, ax1 = plt.subplots()
#     ax1.plot(x, res_max, label="1D-ResNet")
#     ax1.plot(x, le_max, label="1D-LeNet")
#     ax1.plot(x, alex_max, label="1D-AlexNet")
#     ax1.plot(x, zf_max, label="1D-ZFNet")
#     ax1.plot(x, vgg_max, label="1D-VGG")
#     ax1.legend(edgecolor='k')
#     ax1.autoscale(tight=True)
#     ax1.set_ylim(ymin=0, ymax=1)
#     # ax1.set_title('Precision K=9')
#     fig1.show()
