import csv
input1 = csv.DictReader(open('eci_processed.csv'), fieldnames=['mad_id', 'title', 'predicted_ontology', 'image_link', 'old_ontology', 'predicted_gender', 'gender'])
d=dict()
for entry in input1:
    if entry['predicted_ontology']=='':
        for word in entry['old_ontology'].split('>'):
            d[word]=d.get(word, 0)+1

print(len(d.keys()))
import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
df = df[df['Count'] > 1000]
df=df.sort_values('Count', ascending= False)
print(df)
print(sum(df['Count']))
import matplotlib.pyplot as plt
plt.bar(df['Text'], df['Count'])
plt.title('Possible Keywords')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(fontsize=8, rotation=30)
plt.show()
