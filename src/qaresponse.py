import requests
import json
import csv
import words2vec as w2v
import unicodedata
import string
url = 'http://35.173.221.117:8000/api/v0.3/'
client_id = ["4"]


def getresponse(pid):
    url = "https://us-east-1-api.madstreetden.com/widgets"
    params = {"api_key": "d7a44a0541ca3614adfcdd87d8771cbdc6d57d55",
              "num_results": "[10]",
              "product_id": str(pid),
              "widget_list": "[0]",
              "mad_uuid": "qa1234",
              "details": "true"}
    response = requests.post(url, data=params).json()
    return response

title_similarities=dict()
tokenized_corpus, w2v_model = w2v.load_model_and_corpus('000', 'Word2Vecmodel3.dat')
vocabulary = set(w2v_model.wv.index2word)

client_id = [67]
for id in client_id:
    print id

    prod_url = "http://ariadne-510676e66f6374da.elb.us-east-1.amazonaws.com:8983/solr/allstarv5/select?fl=ext_id,title&fq=available:true&fq=client_id:" + \
        str(id)+"&q=*:*&rows=10&sort=random_1234 desc"
    response = requests.get(prod_url, auth=('maduser', '62madmonkeys')).json()
    docs = {k['ext_id'] : k['title'] for k in response['response']['docs']}

    for e, t in docs.iteritems():
        print(t)
        result = getresponse(e)
        title_similarities[t]=dict()
        t=unicodedata.normalize('NFKD', t).encode('ascii','ignore')
        v1=w2v.average_word_vectors(w2v.tokenize_sentence(t), w2v_model, vocabulary, 300)
        for similar_product in result['data'][0]:
            similar_product['title']=unicodedata.normalize('NFKD', similar_product['title']).encode('ascii','ignore')
            title_similarities[t][similar_product['title']]=w2v.cos_sim(v1, w2v.average_word_vectors(w2v.tokenize_sentence(similar_product['title']), w2v_model, vocabulary, 300))
        print(result,"----------")
with open('Similar_Titles', 'w') as f:
     json.dump(title_similarities, f)
