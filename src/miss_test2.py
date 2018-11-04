import json
import csv
from nltk.tokenize import RegexpTokenizer
d = {}
tokenizer = RegexpTokenizer(r'\w+')
stop_words=['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

with open('shoprunner_missing_titles') as f:
    reader = json.load(f)
    for row in reader:
        l = tokenizer.tokenize(row.replace('_', ' '))
        for w in l:
                if stop_words.count(w) == 0:
                    d[w] = d.get(w,0) + 1
# d=dict()
# for k in reader:
#     print(k)
#     l= tokenizer.tokenize(k.replace('_', ' '))
#     for w in l:
#         if stop_words.count(w)==0:
#             d[w]=d.get(w,0)+1

# print(d)
# print(len(d.keys()))
import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
df1 = df[df['Count']>100]
df2 = df[(df['Count']<100)& (df['Count']>50)]
df1=df1.sort_values('Count', ascending= False)
df2=df2.sort_values('Count', ascending= False)
print(df)
# print(df2)
import matplotlib.pyplot as plt
plt.bar(df1['Text'], df1['Count'])
plt.title('Keyword_Counter >100')
plt.xlabel('Text')
plt.ylabel('Count')
plt.xticks(fontsize=7, rotation=30)
plt.show()
plt.bar(df2['Text'], df2['Count'])
plt.title('100>Keyword_Counter>50')
plt.xlabel('Text')
plt.ylabel('Count')
plt.xticks(fontsize=7, rotation=30)
plt.show()
