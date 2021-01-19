# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os.path
from pathlib import Path
'''
    process file for better understanding
'''

'''
    add header for arc-paper-ids
'''
# file_path = '~/Downloads/data/3C/arc-paper-ids.tsv'
# dataformat = pd.read_csv(file_path, sep='\t', names=['AnthologyID', 'Year', 'Title', 'Author'])
# print(dataformat)
# title = dataformat[dataformat['AnthologyID'] == 'A00-1004']
# print(title)
# for index, content in title.iterrows():
#     print(content['Author'])
# # data = np.array(dataformat.loc[:,:])
# # column_headers = list(dataformat.columns.values)
# print(dataformat.columns)
# # print(column_headers)
# out_path = '~/Downloads/data/3C/arc-paper-ids-have-header.tsv'
# if os.path.exists(out_path) and os.path.isfile(out_path):
#     pass
# else:
#     dataformat.to_csv('~/Downloads/data/3C/arc-paper-ids-have-header.tsv', sep='\t')

'''
    
'''
# annotated_json_file_path = "/home/g19tka13/Downloads/data/3C/annotated-json-data-v2/A00-1004.json"
# i = 0
# example = None
# with open(annotated_json_file_path, 'r') as f:
#     for line in f:
#         example = json.loads(line) # dict_keys(['year', 'sections', 'paper_id', 'citation_contexts'])
#
# print(type(example['year']))
# print(example['year'])
# print(type(example['sections']))
# print(len(example['sections']))  # 8
# for element in example['sections']:
#     print(element)
# print(type(example['paper_id']))
# print(type(example['citation_contexts']))

'''

'''
# python 不能将 ~ 识别为 /home/user
# json_file_path = '/home/g19tka13/Downloads/data/3C/acl-arc-json/json/A/A00/A00-1000.json'
# i = 0
# raw = None
# with open(json_file_path, 'r') as f:
#     for line in f:
#         raw = json.loads(line)
#         i = i + 1
# print(raw.keys())
data_path = Path('/home/g19tka13/Downloads/data/3C')
taskA_data = data_path / 'taskA/train.csv'
df = pd.read_csv(taskA_data, sep=',')
print(df.columns)
print(df.head(2)['citation_context'][0])
print(df.head(2)['citation_context'][1])




