import json
with open('shoprunner_predicted2.json') as f:
    a = json.load(f)
d=dict()
for key, value in a.iteritems():
    d[value[0]]=d.get(value[0],0)+1

print(len(d.keys()))
import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
#df = df[df['Count'] < 50000]
df = df.fillna('none')
df=df.sort_values('Count', ascending= False)
print(df)
import matplotlib.pyplot as plt
plt.bar(df['Text'], df['Count'])
plt.title('unisex>sports & outdoors>team shop>team clothing>team tops>jerseys')
plt.xlabel('Predicted Ontology')
plt.ylabel('Count')
plt.xticks(fontsize=8, rotation=30)
plt.show()
