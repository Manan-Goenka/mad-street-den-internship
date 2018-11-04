import json
import unicodedata
from nltk.tokenize import RegexpTokenizer
with open('promoda_predicted.json') as f:
    a = json.load(f)
missed_out_list = []
for key, value in a.iteritems():
    if value == '':
        missed_out_list.append(key)
with open('promoda_for_maria.json') as f:
    b = json.load(f)
missed_out_title=[]
for i in missed_out_list:
    for t in b:
        if(t['ext_id']==i):
            if not isinstance(t['title'], unicode):
                t['title']= unicodedata.normalize('NFKD', unicode(t['title'].strip(), 'utf-8')).encode('ASCII', 'ignore')
            else:
                t['title']= t['title'].strip().encode('ASCII', 'ignore')
            missed_out_title.append(t['title'])
tokenizer = RegexpTokenizer(r'\w+')
stop_words=['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
d=dict()
print(len(missed_out_title))
for k in missed_out_title:
    print(k)
    l= tokenizer.tokenize(k.replace('_', ' '))
    for w in l:
        if stop_words.count(w)==0:
            d[w]=d.get(w,0)+1

print(len(d.keys()))

import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
df = df[df['Count'] > 5]
print(df)
import matplotlib.pyplot as plt
plt.bar(df['Text'], df['Count'])
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(fontsize=7, rotation=30)
plt.show()
