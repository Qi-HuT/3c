# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
import re
import pandas as pd


'''
    Generate ACL.csv
'''

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
title_dict = {}
acl_dataframe = pd.DataFrame(columns=['paper_id', 'section name', 'citation_above', 'citation', 'citation_below'])
for pathlist in third_path:
    for path in pathlist:
        file_path = root_path / path
        with open(file_path, 'r') as f:  # 打开文件
            data_dict = json.load(f)
            if len(data_dict['citation_contexts']) != 0:  # 获得key所在的值
                citation = data_dict['citation_contexts']
                sections = data_dict['sections']
                citation_context_clue = []
                for i in range(len(data_dict['citation_contexts'])): # 遍历citation_contextshuode  每个句子的寻找路径
                    cited_title = citation[i]['info']['title']
                    cited_author = citation[i]['info']['authors'][0]
                    # 获得的数字就是列表的下标。
                    citation_section = citation[i]['section']
                    citation_subsection = citation[i]['subsection']
                    citation_sentence = citation[i]['sentence']
                    citation_context_clue.append([citation[i]['section'], citation[i]['subsection'],
                                                  citation[i]['sentence']])
                    if citation[i]['sentence'] - 1 >= 0:
                        citation_above = sections[citation[i]['section']]['subsections'][citation[i]['subsection']]\
                            ['sentences'][citation[i]['sentence'] - 1]['text']
                    else:
                        citation_above = 'unabove'
                    citation_current = sections[citation[i]['section']]['subsections'][citation[i]['subsection']]\
                        ['sentences'][citation[i]['sentence']]['text']
                    if citation[i]['sentence'] + 1 <= len(sections[citation[i]['section']]['subsections'][citation[i] \
                    ['subsection']]['sentences']) - 1:
                        citation_below = sections[citation[i]['section']]['subsections'][citation[i]['subsection']]\
                        ['sentences'][citation[i]['sentence'] + 1]['text']
                    else:
                        citation_below = 'unbelow'
                    '''
                            获得句子的section_name 先找句子所在的section有没有‘title’这个key，如果没有则要去句子所在的subsection。
                            看有没有‘title’这个key如果没有则将该句子的section_name设为unfind
                    '''
                    if 'title' not in sections[citation[i]['section']].keys():
                        if 'title' not in sections[citation[i]['section']]['subsections'][citation[i]['subsection']].keys():
                            section_name = 'unfind'
                            print('unfind')
                        else:
                            raw_section = sections[citation[i]['section']]['subsections'][citation[i]['subsection']]['title']
                            print(path, citation_section, citation_subsection, citation_sentence)
                            section_name = re.sub(r'[0-9] |[0-9]\. |[0-9]\.[0-9]\. ', '', raw_section)
                            print(sections[citation[i]['section']]['subsections'][citation[i]['subsection']]['title'])
                    else:
                        title = re.sub(r'[0-9] |[0-9]\. ', '', sections[citation[i]['section']]['title'])
                        if title in title_dict.keys():
                            title_dict[title] += 1
                        else:
                            # titl = re.sub(r'[0-9] |[0-9]\. ', '', sentences[citation[i]['section']]['title'])
                            title_dict[title] = 1
                            print(title)
                        # print(sentences[citation[i]['section']]['title'])
                        section_name = title
                    acl_dataframe.loc[acl_dataframe.shape[0]] = {'paper_id': data_dict['paper_id'], 'section name':section_name, 'citation_above':citation_above, 'citation':citation_current,
                                                                 'citation_below': citation_below}
                print(citation_context_clue)
print(title_dict)
acl_dataframe.to_csv('/home/g19tka13/Downloads/data/3C/taskA/aclgenerate.csv', sep=',', index=False)
