# -*- coding: utf-8 -*-
from cnn_data_loader import *
import torch.utils.data as Data
import torch.optim as optim
from Cap import *
import torch
from sklearn.metrics import f1_score
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def pre_correct(l1, l2):
    count_dict = {}
    for i, j in zip(l1, l2):
        if i.item() == j.item():
            if i.item() not in count_dict.keys():
                count_dict[i.item()] = 1
            else:
                count_dict[i.item()] = count_dict[i.item()] + 1
    return count_dict


def main():
    start_time = time.time()
    train_iter, val_iter, test_iter, vocab, weighted, label_word_id = load_data()
    train_iter = Data.DataLoader(train_iter, batch_size=10, shuffle=True)
    val_iter = Data.DataLoader(val_iter, batch_size=10, shuffle=True)
    test_iter = Data.DataLoader(test_iter, batch_size=10, shuffle=False)
    vocab_size = list(vocab.vectors.size())
    # print(vocab_size, type(vocab_size))

    model = FinalModel(kernel_size=3, vocab_size=vocab_size)
    model.embedding.weight.data = vocab.vectors
    model.embedding.weight.requires_grad = False
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ModelLoss(0.9, 0.1, 0.5)
    print(model)
    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for index, item in enumerate(train_iter, 0):
            # item.to(device)
            label = item[2]
            # print(label)
            label_input = torch.eye(10, 6).index_select(dim=0, index=label.long()).to(device)
            optimizer.zero_grad()
            high_out = model(item[0].to(device))
            # print(high_out)
            loss = criterion(high_out, label_input)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
            if (index + 1) % 5 == 0:
                max_value, max_index = torch.max(high_out.cpu(), dim=1)
                # print(max_value, max_index)
                f1 = f1_score(label.long(), max_index, average='macro')
                print('epoch: %d \t item_idx: %d \t loss: %.4f \t f1: %.4f' % (epoch, index, epoch_loss / 50, f1))
                epoch_loss = 0
        # for name, parm in model.named_parameters():
        #     if parm.requires_grad:
        #         print(name)
        # print('test')
        model.eval()
        val_true_label = []
        val_pre_label = []
        with torch.no_grad():
            for item in val_iter:
                # item.to(device)
                true_label = item[2]
                # item[0].to(device)
                pre_out = model(item[0].to(device))
                pre_max_value, pre_max_index = torch.max(pre_out.cpu(), dim=1)
                val_true_label.extend(true_label)
                val_pre_label.extend(pre_max_index)
        f1 = f1_score(torch.LongTensor(val_true_label), torch.Tensor(val_pre_label), average='macro')
        val_pre_correct = pre_correct(val_true_label, val_pre_label)
        print('Val f1: %.4f' % f1, val_pre_correct)
    model.eval()
    test_true_label = []
    test_pre_label = []
    with torch.no_grad():
        for item in test_iter:
            true_label = item[2]
            pre_out = model(item[0].to(device))
            test_max_value, test_max_index = torch.max(pre_out.cpu(), 1)
            test_true_label.extend(true_label)
            test_pre_label.extend(test_max_index)
    f1 = f1_score(torch.LongTensor(test_true_label), torch.Tensor(test_pre_label), average='macro')
    test_pre_correct = pre_correct(test_true_label, test_pre_label)
    print('Test f1: %.4f' % f1, test_pre_correct)
    end_time = time.time()
    print('Spend time', end_time - start_time)


if __name__ == '__main__':
    main()
