import words2vec as w2v
import requests
import json
import csv
from gensim.models import Word2Vec

with open('categorical_words.json') as f:
    categorical_words = json.load(f)
categorical_words = set(categorical_words)

w2v_model = Word2Vec.load('Word2Vecmodel3.dat')

#tokenized_corpus, w2v_model = w2v.load_model_and_corpus('eci_catalog_title.csv', 'ECImodel.dat')
vocabulary = set(w2v_model.wv.index2word)

final_dict=dict()
# def predict_gender():
for i in categorical_words:
    similarity_scores=dict()
    try:
        v1=w2v_model[i]
    except:
        continue
    similarity_scores['man']=w2v.cos_sim(v1, w2v_model['men'])
    similarity_scores['woman']=w2v.cos_sim(v1, w2v_model['women'])
    #similarity_scores['boy']=w2v.cos_sim(v1, w2v_model['boy'])
    #similarity_scores['girl']=w2v.cos_sim(v1, w2v_model['girl'])
    #similarity_scores['unisex']=w2v.cos_sim(v1, w2v_model['unisex'])
    # print(similarity_scores)
    final_dict[i]=similarity_scores.keys()[similarity_scores.values().index(max(similarity_scores.values()))]
print(final_dict)
with open('Catgeory_Gender', 'w') as f:
    json.dump(final_dict, f)
