import csv
input1 = csv.DictReader(open('mns_predicted_ontology.csv'))
d=dict()
for entry in input1:
    d[entry['predicted_ontology'].split('>')[0]]=d.get(entry['predicted_ontology'].split('>')[0] , 0)+1


print(len(d.keys()))
import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
#df = df[df['Count'] > 100]
df=df[df.Text!='']
df = df.fillna('none')
df=df.sort_values('Count', ascending= False)
print(df)
print(sum(df['Count']))
import matplotlib.pyplot as plt
plt.bar(df['Text'], df['Count'])
plt.title('Predicted Genders')
plt.xlabel('Predicted Gender')
plt.ylabel('Count')
plt.xticks(fontsize=8, rotation=30)
plt.show()
