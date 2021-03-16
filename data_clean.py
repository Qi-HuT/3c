# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
data = pd.read_csv('/home/g19tka13/taskA/train.csv', sep=',')
df = data.head(100)
print(df['citation_context'])
stop_words = stopwords.words('english')
stop_words.extend(['et', 'al'])
for index, row in df.iterrows():
    # print(row['citation_context'].lower())
    # print(nltk.word_tokenize(row['citation_context'].lower()))
    # print(nltk.word_tokenize(re.sub(r'[^a-zA-Z]',' ',row['citation_context'].lower())))
    print(re.sub(r'[^a-zA-Z]', ' ', row['citation_context'].lower()).split())
    sen_split = nltk.word_tokenize(re.sub(r'[^a-zA-Z]', ' ', row['citation_context'].lower()))
    filte_words = [word for word in sen_split if word not in stop_words]
    print(filte_words)
# words = stopwords.words('english')
# for w in ['!',',','.','?','-s','-ly','</s>','s']:
#     words += w
# print(words)

