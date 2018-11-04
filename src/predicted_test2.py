import csv
input1 = csv.DictReader(open('eci_processed.csv'), fieldnames=['mad_id', 'title', 'predicted_ontology', 'image_link', 'old_ontology', 'predicted_gender', 'gender'])
d=dict()
for entry in input1:
    d[entry['predicted_ontology']]=d.get(entry['predicted_ontology'],0)+1

print(len(d.keys()))
import pandas as pd
data = {'Count': d.values(),'Text':d.keys()}
df = pd.DataFrame(data)
df = df[df['Count'] > 100]
df = df.fillna('none')
df=df[df.Text!='']
df=df.sort_values('Count', ascending= False)
print(df)
import matplotlib.pyplot as plt
plt.bar(df['Text'], df['Count'])
plt.title('Predicted Ontologies')
plt.xlabel('Predicted Ontology')
plt.ylabel('Count')
plt.xticks(fontsize=8, rotation=30)
plt.show()
