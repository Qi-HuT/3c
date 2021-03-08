from pathlib import Path
import torch
# from process_data import load_data
import torch.utils.data as Data
from model import LSTM
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from process_data import *
# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.2.224', port=22, stdoutToServer=True, stderrToServer=True)


def main(train_type=None):
    model_path = './model.pth'
    # dir_path = Path('/home/g19tka13/Downloads/data/3C')
    # data_path = dir_path / 'taskA/train.csv'
    train_data, weighted = strtolist()
    test_data = loadtestdata()
    preudo_list = []
    used_unlabeled_data = None
    unlabeled_data = None
    vocab = None
    if train_type == 'self_train':
        unlabeled_data = pd.read_csv('/home/g19tka13/taskA/aclgenerate.csv', sep=',')
        unlabeled_data = unlabeled_data.head(3000)
        vocab = load_word_vector(train_data, test_data, 'self_train', unlabeled_data)
    #     prelabeled_data = None
    #     vocab = load_word_vector(train_data, test_data, 'self_train', used_unlabeled_data)
    #
    #     if len(preudo_list) == 0:  # 判断是否第一次训练模型。
    #         train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
    #     else:
    #         train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1, prelabeled_data)  # 加入数据
    else:
        vocab = load_word_vector(train_data, test_data)
    #     train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
    # test_iter, unlabel_iter = assemble(test_data, vocab, 0)
    # return train_iter, val_iter, test_iter, vocab, weighted, label_word_id
    best_val_f1 = 0
    if train_type == 'self_train':
        prelabel_data = None
        vocab_size = vocab.vectors.size()
        print('Total num. of words: {}, word vector dimension: {}'.format(
            vocab_size[0],
            vocab_size[1]))
        model = LSTM(vocab_size[0], vocab_size[1], hidden_size=100, num_layers=2, batch=10)
        model.embedding.weight.data = vocab.vectors
        model.embedding.weight.requires_grad = False
        print(model)
        while len(preudo_list) < 2700:
            class_id = []
            delete_id = []
            if len(preudo_list) == 0:  # 判断是否第一次训练模型。
                train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
            else:
                train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1, prelabeled_data=prelabel_data)  # 加入数据

            test_iter, unlabel_iter = assemble(test_data, vocab, 0, unlabeled_data=unlabeled_data)
            weight = torch.tensor(weighted)
            train_iter = Data.DataLoader(train_iter, batch_size=10, shuffle=True)
            val_iter = Data.DataLoader(val_iter, batch_size=10, shuffle=True)
            test_iter = Data.DataLoader(test_iter, batch_size=10, shuffle=False)
            unlabel_iter = Data.DataLoader(unlabel_iter, batch_size=10, shuffle=False)
            # vocab_size = vocab.vectors.size()
            # print('Total num. of words: {}, word vector dimension: {}'.format(
            #     vocab_size[0],
            #     vocab_size[1]))
            # model = LSTM(vocab_size[0], vocab_size[1], hidden_size=100, num_layers=2, batch=10)
            # model.embedding.weight.data = vocab.vectors
            # model.embedding.weight.requires_grad = False # 使用已经训练好的词向量, 即保持词向量不更新（固定词向量） 则设置为false
            # print(model)
            # print(model.parameters())
            # for parameter in model.parameters():
            #     print(parameter)
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            n_epoch = 10
            # nn.CrossEntropyLoss you will give your weights only once while creating the module
            # loss_cs = nn.CrossEntropyLoss(weight=weight)
            # loss_fnc = nn.CosineEmbeddingLoss()
            # loss_mes = nn.MSELoss()
            y = torch.ones(1).long()
            for epoch in range(n_epoch):
                # model.train放在哪参考网址 https://blog.csdn.net/andyL_05/article/details/107004401
                model.train()
                for item_idx, item in enumerate(train_iter, 0):
                    label = item[2]
                    unique_num, count = torch.unique(label, return_counts=True)  # default sorted=True
                    unique_num = unique_num.tolist()
                    # print(unique_num, count)
                    real_weight = torch.ones(6, dtype=torch.float)
                    for i in range(6):
                        if i in unique_num:
                            idx = unique_num.index(i)
                            real_weight[i] = 1 / np.log(1.02 + count[idx] / 10)
                        else:
                            real_weight[i] = 1 / np.log(2.02)
                    optimizer.zero_grad()
                    out = model(item)
                    # label_pred = KMeans(n_clusters=6, init=label_out).fit_predict(out)
                    # fixed weight result=0.1716
                    # loss = F.cross_entropy(out, label.long(), weight=weight)
                    # real time weight calculation
                    loss = F.cross_entropy(out, label.long(), weight=real_weight)
                    # nn.CosineEmbeddingLoss() 损失函数需要是二维矩阵，而不是一维的。
                    # loss = loss_fnc(torch.unsqueeze(label_pred, dim=0), torch.unsqueeze(label.long(), dim=0), y)
                    # loss = Variable(loss, requires_grad=True)
                    # loss_MES = loss_mes(out,  label_vector)
                    # loss = loss_fnc(out, torch.Tensor(one_hot), y)
                    loss.backward()
                    # print(model.lstm.all_weights.shape)
                    # print(model.lstm.)
                    optimizer.step()
                    if (item_idx + 1) % 5 == 0:
                        train_value, train_y_pre = torch.max(out, 1)  # max函数有两个返回值(此处out是二维数组)第一个是最大值的list，第二个是值对应的位置
                        # print('train_value', train_value)
                        # acc = torch.mean((torch.tensor(train_y_pre == label.long(), dtype=torch.float)))
                        # print(train_y_pre, label.long())
                        f1 = f1_score(label.long(), train_y_pre, average='macro')
                        # print(train_y_pre, label)
                        print('epoch: %d \t item_idx: %d \t loss: %.4f \t f1: %.4f' % (epoch, item_idx, loss, f1))

                model.eval()  # 跑完一个epoch就评价一次模型
                val_pre_label = []
                val_y_label = []
                # if (epoch+1) % 5 == 0:
                with torch.no_grad():
                    # print(unlabel_iter)
                    # for item in unlabel_iter:  # prelabel
                    #     index = item[2]
                    #     out = model(item)
                    #     out = F.softmax(out, dim=1)
                    #     predict_value, predict_class = torch.max(out, 1)
                    #     print('predict_value', predict_value)
                    #     for i in range(len(predict_value)):
                    #         if predict_value[i] > 0.9:
                    #             delete_id.append(index[i])  # 为了获得数据索引，根据索引从原数据中删除。
                    #             class_id.append(predict_class[i])
                    for item in val_iter:
                        label = item[2]
                        out = model(item)
                        _, val_y_pre = torch.max(out, 1)
                        val_pre_label.extend(val_y_pre)
                        val_y_label.extend(label)
                #         f1 = f1_score(label.long(), val_y_pre, average='macro')
                #         val_f1.append(f1)
                # f1 = np.array(f1).mean()
                f1 = f1_score(torch.Tensor(val_y_label).long(), torch.Tensor(val_pre_label), average='macro')
                print(f1)
                if f1 > best_val_f1:
                    print('val acc: %.4f > %.4f saving model %.4f' % (f1, best_val_f1, len(preudo_list)))
                    torch.save(model.state_dict(), model_path)
                    best_val_f1 = f1
            model.eval()  # 一轮训练结束在创建pseudo-label
            with torch.no_grad():
                for item in unlabel_iter:  # prelabel
                    index = item[2]
                    out = model(item)
                    out = F.softmax(out, dim=1)
                    predict_value, predict_class = torch.max(out, 1)
                    # print('predict_value', predict_value)
                    # print('predict_class', predict_class)
                    for i in range(len(predict_value)):
                        if predict_value[i] > 0.9:
                            delete_id.append(index[i].item())  # 为了获得数据索引，根据索引从原数据中删除。
                            class_id.append(predict_class[i].item())
            preudo_list.extend(delete_id)
            if len(preudo_list) != 0:
                unlabeled_data, prelabel_data = split_unlabeled_data(unlabeled_data, delete_id, class_id, prelabel_data)
    else:
        train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
        test_iter, unlabel_iter = assemble(test_data, vocab, 0)
        print('test_iter', test_iter)
        # train_iter, val_iter, test_iter, vocab, weight, label_word_id = load_data()
        weight = torch.tensor(weighted)
        train_iter = Data.DataLoader(train_iter, batch_size=10, shuffle=True)
        val_iter = Data.DataLoader(val_iter, batch_size=10, shuffle=True)
        test_iter = Data.DataLoader(test_iter, batch_size=10, shuffle=False)
        vocab_size = vocab.vectors.size()
        print('Total num. of words: {}, word vector dimension: {}'.format(
            vocab_size[0],
            vocab_size[1]))
        model = LSTM(vocab_size[0], vocab_size[1], hidden_size=100, num_layers=2, batch=10)
        model.embedding.weight.data = vocab.vectors
        model.embedding.weight.requires_grad = False
        print(model)
        # print(model.parameters())
        # for parameter in model.parameters():
        #     print(parameter)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        n_epoch = 1
        best_val_f1 = 0
        # nn.CrossEntropyLoss you will give your weights only once while creating the module
        # loss_cs = nn.CrossEntropyLoss(weight=weight)
        # loss_fnc = nn.CosineEmbeddingLoss()
        # loss_mes = nn.MSELoss()
        y = torch.ones(1).long()
        for epoch in range(n_epoch):
            # model.train放在哪参考网址 https://blog.csdn.net/andyL_05/article/details/107004401
            model.train()
            for item_idx, item in enumerate(train_iter, 0):
                label = item[2]
                unique_num, count = torch.unique(label, return_counts=True)  # default sorted=True
                unique_num = unique_num.tolist()
                # print(unique_num, count)
                real_weight = torch.ones(6, dtype=torch.float)
                for i in range(6):
                    if i in unique_num:
                        idx = unique_num.index(i)
                        real_weight[i] = 1 / np.log(1.02 + count[idx] / 10)
                    else:
                        real_weight[i] = 1 / np.log(2.02)
                optimizer.zero_grad()
                out = model(item)
                # label_pred = KMeans(n_clusters=6, init=label_out).fit_predict(out)
                # fixed weight result=0.1716
                # loss = F.cross_entropy(out, label.long(), weight=weight)
                # real time weight calculation
                loss = F.cross_entropy(out, label.long(), weight=real_weight)
                # nn.CosineEmbeddingLoss() 损失函数需要是二维矩阵，而不是一维的。
                # loss = loss_fnc(torch.unsqueeze(label_pred, dim=0), torch.unsqueeze(label.long(), dim=0), y)
                # loss = Variable(loss, requires_grad=True)
                # loss_MES = loss_mes(out,  label_vector)
                # loss = loss_fnc(out, torch.Tensor(one_hot), y)
                loss.backward()
                # print(model.lstm.all_weights.shape)
                # print(model.lstm.)
                optimizer.step()
                if (item_idx + 1) % 5 == 0:
                    _, train_y_pre = torch.max(out, 1)  # max函数有两个返回值(此处out是二维数组)第一个是最大值的list，第二个是值对应的位置

                    # acc = torch.mean((torch.tensor(train_y_pre == label.long(), dtype=torch.float)))
                    # print(train_y_pre, label.long())
                    f1 = f1_score(label.long(), train_y_pre, average='macro')
                    # print(train_y_pre, label)
                    print('epoch: %d \t item_idx: %d \t loss: %.4f \t f1: %.4f' % (epoch, item_idx, loss, f1))
            # finish each epoch val a time
            val_pre_label = []
            val_y_label = []
            # if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for item in val_iter:
                    label = item[2]
                    out = model(item)
                    _, val_y_pre = torch.max(out, 1)
                    val_pre_label.extend(val_y_pre)
                    val_y_label.extend(label)
                    # acc = torch.mean((torch.tensor(val_y_pre == label, dtype=torch.float)))
            #         f1 = f1_score(label.long(), val_y_pre, average='macro')
            #         val_f1.append(f1)
            # f1 = np.array(f1).mean()
            f1 = f1_score(torch.Tensor(val_y_label).long(), torch.Tensor(val_pre_label), average='macro')
            print(f1)
            if f1 > best_val_f1:
                print('val acc: %.4f > %.4f saving model' % (f1, best_val_f1))
                torch.save(model.state_dict(), model_path)
                best_val_f1 = f1
    test_f1 = []
    test_pre_label = []
    test_y_label = []
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        for item_idx, item in enumerate(test_iter, 0):
            label = item[2]
            out = model(item)
            _, test_y_pre = torch.max(out, 1)
            test_pre_label.extend(val_y_pre)
            test_y_label.extend(label)
            print('test_true_label={} test_pre_label={}'.format(label, test_y_pre))
            # f1 = f1_score(label.long(), test_y_pre, average='macro')
            # test_f1.append(f1)
    final_f1 = f1_score(torch.Tensor(test_y_label).long(), torch.Tensor(test_pre_label), average='macro')
    # final_f1 = np.array(test_f1).mean()
    print('test f1 : %.4f' % final_f1)
    count = 0
    for i in range(len(test_pre_label)):
        if test_y_label[i] == test_pre_label[i]:
            count += 1
    print(count)


if __name__ == '__main__':
    main('self_train')
