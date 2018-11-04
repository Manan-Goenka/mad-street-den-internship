import json
import pandas as pd
import matplotlib.pyplot as plt

with open('Prepositions2') as f:
    a = json.load(f)

for prep in a.keys():
    data = {'Count': a[prep].values(),'Side':a[prep].keys()}
    df = pd.DataFrame(data)
    #df = df[df['Count'] < 50000]
    df = df.fillna('none')
    df=df.sort_values('Count', ascending= False)
    print(df)
    plt.bar(df['Side'], df['Count'])
    plt.title('Prepositions side')
    plt.xlabel(prep)
    plt.ylabel('Count')
    plt.xticks(fontsize=8, rotation=30)
    plt.show()
