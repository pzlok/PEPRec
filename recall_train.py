import pandas as pd
import numpy as np
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from model.ResNet import ResNet
from model.LeNet import LeNet
from model.VGG import VGG19
from model.AlexNet import AlexNet
from model.ZFNet import ZFNet

# 为了每次的实验结果一致
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 使用gpu
device = torch.device("cuda:0")


def one_hot(x):
    """
    :param x: 输入待编码的数据(列表或一维 ndarray 数组)
    :return: 编码后的数据，映射关系字典
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(x)
    integer_encoded = label_encoder.transform(x)  # 对原数据进行标签编码
    x_set = label_encoder.classes_
    integer_encoded_set = label_encoder.transform(x_set)    # 对原数据去重后进行标签编码

    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # 对原数据在标签编码的基础上进行独热编码
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    integer_encoded_set = integer_encoded_set.reshape(len(integer_encoded_set), 1)
    # 对去重数据，在标签编码的基础上进行独热编码
    one_hot_encoded_set = one_hot_encoder.fit_transform(integer_encoded_set)

    # 设计字典
    x_set_dict = dict(zip(x_set, one_hot_encoded_set))
    return one_hot_encoded, x_set_dict


def weight_encoding(df_ex, one_hot_dict):
    """
    :param df_ex: 运动处方
    :param one_hot_dict: one-hot编码字典
    :return: 运动处方权重编码
    """
    y = []
    for p in df_ex:
        n = np.zeros(len(list(one_hot_dict.values())[0]))
        p_l = p.split(',')
        for i, s in enumerate(p_l):
            n[np.where(one_hot_dict[s] == 1)[0]] = 1  # 存在的运动项全设为1
        y.append(n.tolist())
    return y


def train(model):
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    epochs = 5000
    x_tr = x_train.reshape(-1, 1, x_train.shape[1])
    x_te = x_test.reshape(-1, 1, x_test.shape[1])
    rec1 = 0
    rec2 = 0
    save_model = None  # 按训练集保存模型
    save_data = list()  # 保存test的准确率、召回率、F1率
    for epoch in range(epochs):
        # 训练模式
        model.train()

        for inputs, labels in train_loader:
            x_reshape = inputs.reshape(-1, 1, x_train.shape[1])
            y_tr = model(x_reshape.to(device))
            optimizer.zero_grad()
            loss = loss_function(y_tr, labels.to(device))
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))


        save = list()  # 保存test数据
        save.append(loss.item())  # 保存训练集损失

        # 测试模式
        model.eval()
        with torch.no_grad():
            y_tr = model(x_tr.to(device))
            y_te = model(x_te.to(device))
            # 保存测试集损失
            save.append(loss_function(y_te, y_test.to(device)).item())
            # 转换成cpu数据，并将tensor转换成numpy
            y_tr_np = y_tr.data.cpu().numpy()
            y_train_np = y_train.numpy()
            y_te_np = y_te.data.cpu().numpy()
            y_test_np = y_test.numpy()

            # 计算指标值
            tr_num = 0
            tr_sum = 0
            for tr_i, tr_j in zip(y_tr_np, y_train_np):
                tr_p = np.sum(tr_j > 0)
                tr_a = np.argpartition(tr_i, -tr_p)[-tr_p:]
                tr_b = np.argpartition(tr_j, -tr_p)[-tr_p:]
                tr_num += len(np.intersect1d(tr_a, tr_b, False))
                tr_sum += tr_p
            if epoch % 9 == 0:
                print('训练集ACC：' + str(tr_num / tr_sum))
            if rec1 < tr_num / tr_sum:
                rec1 = tr_num / tr_sum
            # 保存训练集ACC
            save.append(round(tr_num / tr_sum, 4))

            # 计算指标值
            te_num = 0
            te_sum = 0
            for te_i, te_j in zip(y_te_np, y_test_np):
                te_p = np.sum(te_j > 0)
                te_a = np.argpartition(te_i, -te_p)[-te_p:]
                te_b = np.argpartition(te_j, -te_p)[-te_p:]
                te_num += len(np.intersect1d(te_a, te_b, False))
                te_sum += te_p
            if rec2 < te_num / te_sum:
                save_model = copy.deepcopy(model)
                rec2 = te_num / te_sum
            # 保存测试集ACC
            save.append(round(te_num / te_sum, 4))

            # 准确率、召回率、F1 = (2 * 准确率 * 召回率) / (准确率 + 召回率)
            for te_k in range(1, 16):
                te_num = 0
                te_sum_p = 0
                te_sum_r = 0
                for te_i, te_j in zip(y_te_np, y_test_np):
                    te_p = np.sum(te_j > 0)
                    te_a = np.argpartition(te_i, -te_k)[-te_k:]
                    te_b = np.argpartition(te_j, -te_p)[-te_p:]
                    te_num += len(np.intersect1d(te_a, te_b, False))
                    te_sum_p += te_k
                    te_sum_r += te_p
                precision = te_num / te_sum_p
                recall = te_num / te_sum_r
                f1 = (2 * precision * recall) / (precision + recall)
                save.append(precision)
                save.append(recall)
                save.append(f1)

            save_data.append(save)

    df_save = pd.DataFrame(save_data,
                           columns=['loss1', 'loss2', 'recall1', 'recall2', 'p1', 'r1', 'f1', 'p2', 'r2', 'f2',
                                    'p3', 'r3', 'f3', 'p4', 'r4', 'f4', 'p5', 'r5', 'f5',
                                    'p6', 'r6', 'f6', 'p7', 'r7', 'f7', 'p8', 'r8', 'f8',
                                    'p9', 'r9', 'f9', 'p10', 'r10', 'f10', 'p11', 'r11', 'f11',
                                    'p12', 'r12', 'f12', 'p13', 'r13', 'f13', 'p14', 'r14', 'f14', 'p15', 'r15', 'f15',
                                    ], dtype='float64'
                           )
    df_save = df_save.round(4)

    df_save.to_csv('./result/res_recall.csv', index=False, encoding='utf-8')
    # df_save.to_csv('./result/LeNet_recall.csv', index=False, encoding='utf-8')
    # df_save.to_csv('./result/AlexNet_recall.csv', index=False, encoding='utf-8')
    # df_save.to_csv('./result/ZFNet_recall.csv', index=False, encoding='utf-8')
    # df_save.to_csv('./result/VGG_recall.csv', index=False, encoding='utf-8')

    save_model.cpu()
    torch.save(save_model, './result/ResNet_recall.pt')
    # torch.save(save_model, './result/LeNet_recall.pt')
    # torch.save(save_model, './result/AlexNet_recall.pt')
    # torch.save(save_model, './result/ZFNet_recall.pt')
    # torch.save(save_model, './result/VGG_recall.pt')


if __name__ == '__main__':
    df = pd.read_csv('./train_data/adult_noise.csv')
    df.drop(['score class'], axis=1, inplace=True)

    pr_list = df['exercise prescription'].to_list()
    pr_set = set(pr_list)
    print('运动处方种类数：' + str(len(pr_set)))
    ex_list = list()
    for p in pr_set:
        for s in p.split(','):
            ex_list.append(s)
    ex_set = set(ex_list)
    print('运动种类数：' + str(len(ex_set)))

    # one-hot编码
    data_one_hot, data_dict = one_hot(np.array(ex_list))
    np.save('./result/data_dict.npy', data_dict)  # 保存one-hot字典

    x_feature = df.iloc[:, :17]
    # 归一化(也就是所说的min-max标准化)通过调用sklearn库的标准化函数
    min_max_scaler = preprocessing.MinMaxScaler()
    x_feature = min_max_scaler.fit_transform(x_feature)
    np.save('./result/min_max_scaler.npy', min_max_scaler)

    y_label = weight_encoding(df['exercise prescription'], data_dict)
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_label, test_size=0.2, random_state=42,
                                                        stratify=y_label)
    print('训练集数：' + str(len(y_train)))
    print('测试集数：' + str(len(y_test)))

    # 将数据类型转换为tensor方便pytorch使用
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=144,
                              shuffle=True,
                              num_workers=0,
                              drop_last=False)

    # 模型分类个数
    features = x_train.shape[1]
    classes = y_train.shape[1]

    # 创建训练模型
    res_net = ResNet(in_channels=1, classes=classes).to(device)
    # le_net = LeNet(input_channels=1, input_sample_points=features, classes=classes).to(device)
    # alex_net = AlexNet(input_channels=1, input_sample_points=features, classes=classes).to(device)
    # zf_net = ZFNet(input_channels=1, input_sample_points=features, classes=classes).to(device)
    # vgg19 = VGG19(In_channel=1, classes=classes).to(device)

    # 训练
    train(res_net)
    # train(le_net)
    # train(alex_net)
    # train(zf_net)
    # train(vgg19)

