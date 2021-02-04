# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
path = '/home/g19tka13/Downloads/data/3C/acl-arc-json/json'
root_path = Path(path)
filelist = os.listdir(root_path)
# print(filelist)  # ['A', 'J', 'K', 'E', 'W', 'P', 'Q', 'N', 'S', 'D']
second_path = []
for i in filelist:
    second_path.extend([i+'/'+dirname for dirname in os.listdir(root_path / i)])
# print(second_path) # ['A/A92', 'A/A88', 'A/A83', 'A/A97', 'A/A00', 'A/A94', 'J/J98',...]
third_path = []
for path in second_path:
    third_path.append([path + '/' + filename for filename in os.listdir(root_path / path)])
# print(third_path)  # [['A/A92/A92-1020.json', 'A/A92/A92-1002.json', 'A/A92/A92-1047.json',...],...]

for pathlist in third_path:
    for path in pathlist:
        file_path = root_path / path
        print(10*'*', file_path)
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
            # citation_index = json['citation_contexts'][1]
            if len(data_dict['citation_contexts']) != 0:
                citation = data_dict['citation_contexts']
                sentences = data_dict['sections']
                citation_context_clue = []
                for i in range(len(data_dict['citation_contexts'])):
                    cited_title = citation[i]['info']['title']
                    cited_author = citation[i]['info']['authors'][0]
                    # 获得的数字就是列表的下标。
                    citation_context_clue.append([citation[i]['section'], citation[i]['subsection'], citation[i]['sentence']])
                    print(sentences[citation[i]['section']]['subsections'][citation[i]['subsection']]['sentences'][citation[i]['sentence']]['text'])
                print(citation_context_clue)
