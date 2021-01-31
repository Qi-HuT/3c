from pathlib import Path
import torch
from process_data import load_data
import torch.utils.data as Data
from model import LSTM
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score


def main():
    model_path = './model.pth'
    # dir_path = Path('/home/g19tka13/Downloads/data/3C')
    # data_path = dir_path / 'taskA/train.csv'
    train_iter, val_iter, test_iter, vocab = load_data()
    train_iter = Data.DataLoader(train_iter, batch_size=10, shuffle=True)
    val_iter = Data.DataLoader(val_iter, batch_size=10, shuffle=True)
    test_iter = Data.DataLoader(test_iter, batch_size=10, shuffle=True)
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
    n_epoch = 50
    best_val_f1 = 0
    loss_cs = nn.CrossEntropyLoss()
    # loss_fnc = nn.CosineEmbeddingLoss()
    loss_mes = nn.MSELoss()
    # y = torch.ones((10, 1)).long()
    for epoch in range(n_epoch):
        # model.train放在哪参考网址 https://blog.csdn.net/andyL_05/article/details/107004401
        model.train()
        for item_idx, item in enumerate(train_iter, 0):
            label = item[2]
            # label_word_id = item[3]
            # print(label)
            # label_numpy = label.long()
            # one_hot = np.zeros((10, 6), dtype=np.int64)
            # for i in range(len(label)):
            #     one_hot[i][label_numpy[i]] = label_numpy[i] * 10
            # print(one_hot)
            optimizer.zero_grad()
            # out, label_vector = model(item, label_word_id)
            out = model(item)
            # loss_CS = loss_cs(out, label.long())
            loss = loss_cs(out, label.long())
            # loss_MES = loss_mes(out,  label_vector)
            # loss = loss_fnc(out, torch.Tensor(one_hot), y)
            # loss = loss_CS + loss_MES
            loss.backward()
            # print(model.lstm.all_weights.shape)
            # print(model.lstm.)
            optimizer.step()
            if (item_idx + 1) % 5 == 0:
                _, train_y_pre = torch.max(out, 1)  # max函数有两个返回值，第一个是最大值的list，第二个是值对应的位置

                # acc = torch.mean((torch.tensor(train_y_pre == label.long(), dtype=torch.float)))
                # print(train_y_pre, label.long())
                f1 = f1_score(label.long(), train_y_pre, average='macro')
                # print(train_y_pre, label)
                print('epoch: %d \t item_idx: %d \t loss: %.4f \t f1: %.4f' % (epoch, item_idx, loss, f1))

        val_pre_label = []
        val_y_label = []
        if (epoch+1) % 5 == 0:
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
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        for item_idx, item in enumerate(test_iter, 0):
            label = item[2]
            out = model(item)
            _, test_y_pre = torch.max(out, 1)
            print('test_true_label={} test_pre_label={}'.format(label, test_y_pre))
            f1 = f1_score(label.long(), test_y_pre, average='macro')
            test_f1.append(f1)
    final_f1 = np.array(test_f1).mean()
    print('test f1 : %.4f' % final_f1)


if __name__ == '__main__':
    main()
