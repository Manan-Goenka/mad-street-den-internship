import words2vec as w2v
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

def tagsent_builder(input_file):
    i=1
    df = pd.read_csv(input_file, header=None, names=['title', 'description', 'gender'])
    sent_list=list()
    for document in df.title.tolist():
        sent_list.append(LabeledSentence(words=w2v.tokenize_sentence(document), tags=['SENT_%s' % i]))
        i+=1
    return sent_list

def train_model_doc2vec(input_file, model_file):
    sentences=tagsent_builder(input_file)
    d2v_model = Doc2Vec(sentences, size = 100, window = 300, min_count = 1, workers=4)
    d2v_model.save(open(model_file, 'w'))
    return
