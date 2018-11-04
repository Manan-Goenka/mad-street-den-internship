import json
import pandas as pd
import psycopg2 as psql
import sys
import unicodedata

def make_connection():
    try:
        conn_string = "dbname='msd' port='5439' user='db_runner' password='MSD_dashboard_42'" \
                      "host='optimizedcluster.czunimvu3pdw.us-west-2.redshift.amazonaws.com'"
        connection = psql.connect(conn_string)
        redshift_cursor = connection.cursor()
        return connection, redshift_cursor
    except:
        conn_string = "dbname='msd' port='15439' user='db_runner' password='MSD_dashboard_42'" \
                      "host='localhost'"
        connection = psql.connect(conn_string)
        redshift_cursor = connection.cursor()
        return connection, redshift_cursor

client_name = sys.argv[1]
client_id = sys.argv[2]

conn, cursor = make_connection()

cursor.execute("select ext_id, image_link from \"{}\".product_metadata_{}".format(client_name, client_id))
df2 = pd.DataFrame(cursor.fetchall(), columns=['ext_id', 'image_link'])

with open(client_name + '_for_maria.json') as f:
    input_file = json.load(f)

with open(client_name + '_predicted.json') as f:
    predicted_file = json.load(f)

lst_of_ext_ids = []
lst_of_ontologies = []
lst_of_titles = []

for entry in input_file:
    lst_of_ext_ids.append(entry['ext_id'])
    lst_of_ontologies.append(entry['ontology'])
    lst_of_titles.append(entry['title'])
df = pd.DataFrame(lst_of_ext_ids, columns=['ext_id'])
df['original_ontology'] = lst_of_ontologies
df['title'] = lst_of_titles

lst_of_predicted = [predicted_file[x][0] for x in df['ext_id']]
df['predicted_ontology'] = lst_of_predicted

df = df.merge(df2, on='ext_id')


predicted_ontology = []
title = []
original_ontology = []
image_link = []

for index, row in df.iterrows():
    try:
        if not isinstance(row['predicted_ontology'], unicode):
            predicted_ontology.append(unicodedata.normalize('NFKD', unicode(
                    row['predicted_ontology'].strip(), 'utf-8')).encode('ASCII', 'ignore'))
        else:
            predicted_ontology.append(row['predicted_ontology'].strip().encode('ASCII', 'ignore'))
        if not isinstance(row['title'], unicode):
            title.append(unicodedata.normalize('NFKD', unicode(row['title'].strip(), 'utf-8')).encode('ASCII', 'ignore'))
        else:
            title.append(row['title'].strip().encode('ASCII', 'ignore'))
        if not isinstance(row['original_ontology'], unicode):
            original_ontology.append(unicodedata.normalize('NFKD', unicode(row['original_ontology'].strip(), 'utf-8')).encode('ASCII', 'ignore'))
        else:
            original_ontology.append(row['original_ontology'].strip().encode('ASCII', 'ignore'))

        image_link.append(row['image_link'])
        #print predicted_ontology, title, row['image_link'], original_ontology

    except Exception as e:
        continue

df_new = pd.DataFrame(original_ontology, columns=['original_ontology'])
df_new['title'] = title
df_new['predicted_ontology'] = predicted_ontology
df_new['image_link'] = image_link

df_new.to_csv('{}_images.csv'.format(client_name))