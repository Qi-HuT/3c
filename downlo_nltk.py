# -*- coding: utf-8 -*-
import jsonlines
from pathlib import Path
import numpy as np
import json
list = np.random.randint(500, size=10000).tolist()
print(list)
root_path = Path('/home/g19tka13')
scaffl = Path('/home/g19tka13/acl-arc/scaffolds')
acl_path = Path('/home/g19tka13/acl-arc')
test = acl_path / 'test.jsonl'
dev = acl_path / 'dev.jsonl'
train = acl_path / 'train.jsonl'
worth = scaffl / 'cite-worthiness-scaffold-train.jsonl'
section = scaffl / 'sections-scaffold-train.jsonl'
worth_id = []
section_id = []
w = 0
r = 0
with jsonlines.open(worth, mode='r') as reader:
    for row in reader:
        # print(row)
        # print(r, 'check')
        # if row['is_citation']:
        #     # print(row)
        #     print(row['is_citation'], w)
        #     w += 1
        # r += 1
        # # break
        # # print(row['citation_id'])
        worth_id.append(row['text'])

with jsonlines.open(section, mode='r') as reader:
    for row in reader:
        # print(r, 'check')
        # print('section', row.keys())
        r += 1
        section_id.append(row['text'])


def check(id_list, type_check):
    i, j ,k = 0, 0 , 0
    idlist = id_list
    with jsonlines.open(test, mode='r') as reader:
        for row in reader:
            if row['text'] in idlist:
                # print('{}_true_test{}'.format(type_check, i))
                i += 1

    with jsonlines.open(dev, mode='r') as reader:
        for row in reader:
            if row['text'] in idlist:
                # print('{}_true_dev{}'.format(type_check, j))
                j += 1
    with jsonlines.open(train, mode='r') as reader:
        for row in reader:
            if row['text'] in idlist:
                k += 1
                # print('{}_true_train{}'.format(type_check, k))
    return i, j, k


def random_list(num):
    return np.np.random.randint(num, size=10000).tolist()


wor_i, wor_j, wor_k = check(worth_id, 'worth')
print(check(worth_id, 'worth'))
sec_i, sec_j , sec_k = check(section_id, 'section')
print(check(section_id, 'section'))
