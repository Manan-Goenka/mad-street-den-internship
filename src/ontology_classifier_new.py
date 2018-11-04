import json
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
import psycopg2 as psql
import unicodedata
from datetime import datetime
from timeit import default_timer
from itertools import permutations
import pandas as pd
import csv
import sys

with open('categorical_words.json') as f:
    categorical_words = json.load(f)
categorical_words = set(categorical_words)

with open('adjectives.json') as f:
    adjectives = json.load(f)
adjectives = set(adjectives)

def make_connection():
    try:
        conn_string = "dbname='msd' port='5439' user='msduser' password='MSD45chennai'" \
                      "host='optimizedcluster.czunimvu3pdw.us-west-2.redshift.amazonaws.com'"
        connection = psql.connect(conn_string)
        redshift_cursor = connection.cursor()
        return connection, redshift_cursor
    except:
        conn_string = "dbname='msd' port='15439' user='msduser' password='MSD45chennai'" \
                      "host='localhost'"
        connection = psql.connect(conn_string)
        redshift_cursor = connection.cursor()
        return connection, redshift_cursor


def create_new_map(original_dict):
    first_letter_length = {}
    for keywords in original_dict.keys():
        for keyword in keywords.split('@'):
            if keyword:
                if keyword[0] not in first_letter_length.keys():
                    first_letter_length[keyword[0]] = {}
                    first_letter_length[keyword[0]][len(keyword)] = set([])
                    first_letter_length[keyword[0]][len(keyword)].add(keyword)
                else:
                    if len(keyword) not in first_letter_length[keyword[0]]:
                        first_letter_length[keyword[0]][len(keyword)] = set([])
                        first_letter_length[keyword[0]][len(keyword)].add(keyword)
                    else:
                        first_letter_length[keyword[0]][len(keyword)].add(keyword)
    return first_letter_length


def get_keywords(ontologies, mapping_dict, wnl, tokenizer):
    keywords = []
    tokens = tokenizer.tokenize(ontologies.replace('_', ' '))
    for index, word in enumerate(tokens):
        try:
            if not isinstance(word, unicode):
                word = unicodedata.normalize('NFKD', unicode(
                word.strip(), 'utf-8')).encode('ASCII', 'ignore')
                word = wnl.lemmatize(word.lower())
            else:
                word = word.strip().encode('ASCII', 'ignore')
                word = wnl.lemmatize(word.lower())
            if word in mapping_dict[word[0]][len(word)]:
                keywords.append((index, word))
        except:
            continue
    return keywords



with open('reverse_masterontology_feb28.json') as f:
        json_file = json.load(f)
mapping_dict = create_new_map(json_file)
# ontologies_title = []

def predict(ontology='', title='', wnl=WordNetLemmatizer(), tokenizer=RegexpTokenizer(r'\w+')):
    ontology_keywords = get_keywords(ontology, mapping_dict, wnl, tokenizer)
    title_keywords = get_keywords(title, mapping_dict, wnl, tokenizer)
    i = 3
    # print(ontology_keywords)
    #print(title_keywords)
    keyword_score_ontology = {}
    while i > 0:
        score = []
        rank = []
        new_keywords = ['@'.join(x3) for x3 in list(permutations([x2[1] for x2 in ontology_keywords], i))]
        # print(new_keywords)
        for tup in [x3 for x3 in list(permutations([x2[0] for x2 in ontology_keywords], i))]:
            # print(tup)
            rank.append(sum([x for x in tup]))
            score.append(sum([1.0/y for y in [abs(x - tup[index - 1]) for index, x in enumerate(tup) if index > 0]]))
            # print(rank)
            # print(score)
        score = [(i-1)*yy for yy in score]
        new_keywords = zip(score, new_keywords, rank)
        # print(new_keywords)
        fallback=list()
        if set([x[1] for x in new_keywords]).intersection(adjectives) != set([]):
            fallback.extend([x for x in new_keywords if x[1] in adjectives])
            new_keywords = list(set(new_keywords) - set([x for x in new_keywords if x[1] in adjectives]))
        for new_keyword in new_keywords:
            if new_keyword[1] in json_file.keys():
                keyword_score_ontology[new_keyword[1]] = (new_keyword[0], new_keyword[2])
        i -= 1
    if (not bool(keyword_score_ontology) & (fallback != set([]))):
        for adj in fallback:
            if adj[1] in json_file.keys():
                keyword_score_ontology[adj[1]] = (adj[0], adj[2])

    i = 3
    keyword_score_title = {}
    while i > 0:
        score = []
        rank = []
        new_keywords = ['@'.join(x3) for x3 in list(permutations([x2[1] for x2 in title_keywords], i))]
        # print(new_keywords)
        for tup in [x3 for x3 in list(permutations([x2[0] for x2 in title_keywords], i))]:
            # print(tup)
            rank.append(sum([x for x in tup]))
            score.append(sum([1.0/y for y in [abs(x - tup[index - 1]) for index, x in enumerate(tup) if index > 0]]))
            # print(rank)
            # print(score)
        score = [(i-1)*yy for yy in score]
        new_keywords = zip(score, new_keywords, rank)
        # print(new_keywords)
        fallback=list()
        if set([x[1] for x in new_keywords]).intersection(adjectives) != set([]) :
            fallback.extend([x for x in new_keywords if x[1] in adjectives])
            new_keywords = list(set(new_keywords) - set([x for x in new_keywords if x[1] in adjectives]))
        # print(new_keywords)
        for new_keyword in new_keywords:
            if new_keyword[1] in json_file.keys():
                keyword_score_title[new_keyword[1]] = (new_keyword[0], new_keyword[2])
                # print(keyword_score_title)
        i -= 1
    # print(keyword_score_ontology)
    # print(keyword_score_title)
    # print(fallback)
    if (not bool(keyword_score_title) & (fallback != set([]))):
        for adj in fallback:
            if adj[1] in json_file.keys():
                keyword_score_title[adj[1]] = (adj[0], adj[2])
    # print(keyword_score_title)
    candidate_score_ontology = {}
    candidate_score_title = {}
    candidate_rank_ontology = {}
    candidate_rank_title = {}
    for keyword, score in keyword_score_ontology.iteritems():
        candidate_score_ontology[(json_file[keyword])] = candidate_score_ontology.get(json_file[keyword], 0) + score[0]
        candidate_rank_ontology[(json_file[keyword])] = candidate_rank_ontology.get(json_file[keyword], 0) + score[1]

    for keyword, score in keyword_score_title.iteritems():
        candidate_score_title[(json_file[keyword])] = candidate_score_title.get(json_file[keyword], 0) + score[0]
        candidate_rank_title[(json_file[keyword])] = candidate_rank_title.get(json_file[keyword], 0) + score[1]
    # print(candidate_score_ontology)
    # print(candidate_score_title)
    # print(candidate_rank_ontology)
    # print(candidate_rank_title)
    # with open('shoprunner_predictions_ontology.json', 'w') as f:
    #      json.dump(final_ontology_prediction, f, sort_keys=True, indent=4)
    #
    # with open('shoprunner_predictions_title.json', 'w') as f:
    #      json.dump(final_title_prediction, f, sort_keys=True, indent=4)
    try:
        if len([k for k,v in candidate_score_ontology.iteritems() if v == max(candidate_score_ontology.values())]) == 1:
            final_ontology_prediction = max(candidate_score_ontology, key=candidate_score_ontology.get)
        else:
            temp = [k for k, v in candidate_score_ontology.iteritems() if v == max(candidate_score_ontology.values())]
            temp2 = {k:v for k,v in candidate_rank_ontology.iteritems() if k in temp}
            final_ontology_prediction = max(temp2, key=temp2.get)
    except:
        final_ontology_prediction = ''

    try:
        if len([k for k,v in candidate_score_title.iteritems() if v == max(candidate_score_title.values())]) == 1:
            final_title_prediction = max(candidate_score_title, key=candidate_score_title.get)
        else:
            temp = [k for k, v in candidate_score_title.iteritems() if v == max(candidate_score_title.values())]
            temp2 = {k:v for k,v in candidate_rank_title.iteritems() if k in temp}
            final_title_prediction = max(temp2, key=temp2.get)
    except:
        final_title_prediction = ''
    return final_ontology_prediction, final_title_prediction

if __name__ == '__main__':
    # with open('shoprunner_for_maria.csv') as f:
    #     ontologies_title = [tuple(line) for line in csv.reader(f)]
    # print "data fetched"
    # ontologies_title = ontologies_title[1:]
    # for entry in ontologies_title:
    #     predict(entry[2], entry[3])
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file) as f:
        input1 = json.load(f)
    input  = []
    for elem in input1:
        if elem['ontology'] == 'womens>shoes>sandals':
            input.append(elem)
    tic = default_timer()
    wnl = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    output = {}
    x = len(input)
    c=''
    for entry in input:
        print "{} remaining".format(x)
        x -= 1
        if not isinstance(entry['title'], unicode):
            entry['title']= unicodedata.normalize('NFKD', unicode(entry['title'].strip(), 'utf-8')).encode('ASCII', 'ignore')
        else:
            entry['title']= entry['title'].strip().encode('ASCII', 'ignore')
        if('for' in entry['title'].split(' ')):
            entry['title'] = entry['title'].split("for",1)[1]
        if('with' in entry['title'].split(' ')):
            if(get_keywords(entry['title'].split("with",1)[1], mapping_dict, wnl, tokenizer) == set([])):
                entry['title'] = entry['title'].split("with",1)[0]
        if('by' in entry['title'].split(' ')):
            if(get_keywords(entry['title'].split("by",1)[1], mapping_dict, wnl, tokenizer) == set([])):
                entry['title'] = entry['title'].split("by",1)[0]
        output[entry['ext_id']] = (predict(entry['ontology'],entry['title'], wnl, tokenizer)[1], 'title')
        if(output[entry['ext_id']][0] == ''):
            output[entry['ext_id']] = (predict(entry['ontology'],entry['title'], wnl, tokenizer)[0], 'ontology')

    print "time is", default_timer() - tic
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4, sort_keys=True)
