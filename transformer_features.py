# -*- coding: utf-8 -*-
from transformers import TransfoXLModel, TransfoXLTokenizer, TransfoXLConfig
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
# import nltk
import torch
from torch import nn
import torch.optim as optim
import time
from sklearn.metrics import f1_score
start_time = time.time()
train_set = pd.read_csv('/home/g19tka13/taskA/train.csv', sep=',')
test = pd.read_csv('/home/g19tka13/taskA/test.csv').merge(pd.read_csv('/home/g19tka13/taskA/sample_submission.csv'),
                                                          on='unique_id')
print(train_set.columns)
print(train_set['citation_class_label'].value_counts())  # 查看数据集中标签的分布
train_set = train_set.sample(frac=1).reset_index(drop=True)
train = train_set.loc[:int(train_set.shape[0] * 0.8) - 1]
val = (train_set.loc[int(train_set.shape[0] * 0.8):]).reset_index(drop=True)
# train = train_set.loc[:79]
# val = (train_set.loc[80:88]).reset_index(drop=True)
print(train)
print(val)
print(test)
# exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# config = TransfoXLConfig.from_pretrained('transfo-xl-wt103')
# config.hidden_size = 300
# print(config.hidden_size)
# exit()


class TransformerGetFeatures(nn.Module):
    def __init__(self):
        super(TransformerGetFeatures, self).__init__()
        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        self.linear = nn.Linear(1024, 6)

    def forward(self, input_object=None, target=None, loss_function=None, run_type=None):
        # input_object = self.tokenizer(batch_sentences, return_tensors='pt', is_split_into_words=True, padding=True,
        #                               return_length=True)
        input_ids = input_object['input_ids']
        batch_length = input_object['length']
        batch_length = (batch_length - 1).view(input_ids.shape[0], 1, -1)
        batch_length = batch_length.repeat(1, 1, 1024)
        # 如何获得想要的features
        # tensor_3d = torch.LongTensor([1, 2, 3, 4, 5, 6])
        # tensor_3d = tensor_3d.view(6, 1, -1)
        # cat = tensor_3d.repeat(1, 1, 6)
        # print(cat)
        # ten_list = torch.randn(6, 7, 6)
        # print(ten_list)
        # print(torch.gather(ten_list, 1, cat))
        transformer_output = self.model(input_ids)
        gather_output = torch.gather(transformer_output.last_hidden_state, 1, batch_length)
        squeeze_output = torch.squeeze(gather_output, dim=1)
        linear_output = self.linear(squeeze_output)
        # if self.training:
        #     print('train')
        if run_type =='train':
            loss = loss_function(linear_output, target)
            return loss, linear_output
        else:
            return linear_output


def generate_batch_data(data, batch_size=8):
    batch_count = int(data.shape[0] / batch_size)
    sentences_list, target_list = [], []
    for i in range(batch_count):
        mini_batch_sentences, mini_batch_target = [], []
        for j in range(batch_size):
            mini_batch_sentences.append([data['citation_context'][i * batch_size + j]])
            mini_batch_target.append(data['citation_class_label'][i * batch_size + j])
        sentences_list.append(mini_batch_sentences)
        target_list.append(mini_batch_target)
    return sentences_list, target_list

train_sentences_list, train_target_list = generate_batch_data(train)
val_sentences_list, val_target_list = generate_batch_data(val)
test_sentences_list, test_target_list = generate_batch_data(test)
epoch = 10
print_every_batch = 10
transformer_model = TransformerGetFeatures()
# transformer_model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
optimizer = optim.Adam([{'params': transformer_model.linear.parameters()}, {'params': transformer_model.model.parameters()}], lr=0.00001)
ce_loss = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
transformer_model.to(device=device)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
tokenizer.pad_token = '[PAD]'

for i in range(10):
    transformer_model.train()
    avg_loss = 0
    tic = time.time()
    for index, (sentences, target) in enumerate(zip(train_sentences_list, train_target_list)):
        sentences = tokenizer(sentences, return_tensors='pt', is_split_into_words=True, padding=True,
                              return_length=True)
        target = torch.LongTensor(target)

        sentences = sentences.to(device=device)
        target = target.to(device=device)

        optimizer.zero_grad()
        loss, output = transformer_model(sentences, target, ce_loss, 'train')
        # loss = ce_loss(output, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        # break
        # for name, weight in transformer_model.named_parameters():
        #     print('--->name:', name, '--->grad_requirs', weight.requires_grad, '-->grad_value:', weight.grad, weight.)
        if (index + 1) % 10 == 0:
            train_value, train_location = torch.max(output, dim=1)
            cpu_target = target.cpu()
            cpu_train_location = train_location.cpu()
            train_f1 = f1_score(cpu_target, cpu_train_location, average='macro')
            print(cpu_target, cpu_train_location)

            print('Batch: %d, Loss: %.4f, F1: %.4f' % ((index + 1), avg_loss / print_every_batch, train_f1))
            avg_loss = 0
    toc = time.time()
    print("time used:", toc - tic)
    true_label = []
    pre_label = []
    transformer_model.eval()
    with torch.no_grad():
        for sentences, target in zip(val_sentences_list, val_target_list):
            sentences = tokenizer(sentences, return_tensors='pt', is_split_into_words=True, padding=True,
                                  return_length=True)
            sentences.to(device)
            val_output = transformer_model(sentences).cpu()
            predict_val_value, predict_val_label = torch.max(val_output, 1)
            print('predict_val_label', predict_val_label) # tensor([0, 0, 0, 0, 0, 0, 0, 0])
            true_label.extend(target)
            pre_label.extend(predict_val_label.tolist())
        print('\033[1;35m true_label: \033[0m', Counter(true_label), true_label)
        print('\033[1;30m pre_label: \033[0m', Counter(pre_label), pre_label)
        val_f1 = f1_score(torch.LongTensor(true_label), torch.LongTensor(pre_label), average='macro')
        # print(true_label, pre_label)
        print('Epoch: %d, F1: %.4f' % (i, val_f1))
test_true_label = []
test_pre_label = []
transformer_model.eval()
with torch.no_grad():
    for sentences, target in zip(test_sentences_list, test_target_list):
        sentences = tokenizer(sentences, return_tensors='pt', is_split_into_words=True, padding=True,
                              return_length=True)
        sentences.to(device)
        test_output = transformer_model(sentences).cpu()
        predict_test_value, predict_test_label = torch.max(test_output, dim=1)
        test_true_label.extend(target)
        test_pre_label.extend(predict_test_label.tolist())
predict_correct = []
for i in range(len(test_true_label)):
    if test_true_label[i] == test_pre_label[i]:
        predict_correct.append(test_pre_label[i])
test_f1 = f1_score(torch.LongTensor(test_true_label), torch.LongTensor(test_pre_label), average='macro')
end_time = time.time()
print('test_true_label:', Counter(test_true_label), test_true_label)
print('test_pre_label:', Counter(test_pre_label), test_pre_label)
print('predict_correct:', predict_correct)
print('run_time: %.4f' % (end_time - start_time))
print('Test F1: %.4f' % test_f1)

# tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# # tokenizer.padding_side = 'left'
# model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
# tokenizer.pad_token = '[PAD]'
# # print(tokenizer.pad_token_id)
# iny = tokenizer([['Hello my dog is cute'], ['Hello my father dog is cute']], return_tensors='pt',  padding=True,
#                 is_split_into_words=True, return_length=True)
# print(iny)
# a = tokenizer.convert_ids_to_tokens(iny['input_ids'][0])
# print(a)
#
# output = model(iny['input_ids'])
# print(output.last_hidden_state)
# print(output.last_hidden_state.shape)
# print(output.last_hidden_state)
# print(output.hidden_states)
# print(output.mems)