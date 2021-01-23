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
    # dir_path = Path('/home/g19tka13/Downloads/data/3C')
    # data_path = dir_path / 'taskA/train.csv'
    train_iter, val_iter, vocab = load_data()
    train_iter = Data.DataLoader(train_iter, batch_size=10, shuffle=True, pin_memory=True)
    val_iter = Data.DataLoader(val_iter, batch_size=10, shuffle=True, pin_memory=True)
    vocab_size = vocab.vectors.size()
    print('Total num. of words: {}, word vector dimension: {}'.format(
        vocab_size[0],
        vocab_size[1]))
    model = LSTM(vocab_size[0], vocab_size[1], hidden_size=100, num_layers=2, batch=10)
    model.embedding.weight.data = vocab.vectors
    # print(model.parameters())
    # for parameter in model.parameters():
    #     print(parameter)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_epoch = 50
    best_val_f1 = 0
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(n_epoch):
        for item_idx, item in enumerate(train_iter, 0):
            label = item[2]
            optimizer.zero_grad()
            out = model(item)
            loss = loss_fn(out, label.long())
            loss.backward()
            optimizer.step()
            if (item_idx + 1) % 5 == 0:
                _, train_y_pre = torch.max(out, 1)  # max函数有两个返回值，第一个是最大值的list，第二个是值对应的位置

                # acc = torch.mean((torch.tensor(train_y_pre == label.long(), dtype=torch.float)))
                print(train_y_pre, label.long())
                f1 = f1_score(label.long(), train_y_pre, average='macro')
                # print(train_y_pre, label)
                print('epoch: %d \t item_idx: %d \t loss: %.4f \t train acc: %.4f' % (epoch, item_idx, loss, f1))

        val_f1 = []
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for item in val_iter:
                    label = item[2]
                    out = model(item)
                    _, val_y_pre = torch.max(out, 1)
                    # acc = torch.mean((torch.tensor(val_y_pre == label, dtype=torch.float)))
                    f1 = f1_score(label.long(), val_y_pre, average='macro')
                    val_f1.append(f1)
            f1 = np.array(f1).mean()
            print(f1)
            if f1 > best_val_f1:
                print('val acc: %.4f > %.4f saving model' % (f1, best_val_f1))


if __name__ == '__main__':
    main()
