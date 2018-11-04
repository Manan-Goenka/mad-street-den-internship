import json
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from pattern.es import parsetree
import unicodedata
from timeit import default_timer
from itertools import permutations
import sys
import traceback
import csv
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# with open(dir_path+'/translated_categorical_words.json') as f:
#     categorical_words = json.load(f)
# categorical_words = set(categorical_words)

with open(dir_path+'/adjectives.json') as f:
    adjectives = json.load(f)
adjectives = set(adjectives)

wnl = WordNetLemmatizer()


def getascii(val):
    """
    Normalize the characters using ascii
    and return unicode encoded word
    :param val: word
    :return val: unicode encoded word
    """
    if val:
        if isinstance(val, unicode):
            val = val.encode('utf-8', 'ignore')
        else:
            val = val.decode('utf-8', 'ignore').encode('utf-8', 'ignore')

        try:
            tmp = unicodedata.normalize('NFKD', unicode(
                val.strip(), 'utf-8')).encode('ASCII', 'ignore')
            return unicode(tmp)
        except Exception as e:
            print("Failed to normalize: {} with exception {}".format(val, e))
            raise e
    else:
        return None


def stem(val):
    """
    Lemmatize the val to its simple form
    :param val: word string
    :return newval: lemmatized word
    """
    if val:
        newval = wnl.lemmatize(unicode(val), pos='v')
        if val == newval:
            newval = wnl.lemmatize(unicode(val), pos='n')
        return newval
    else:
        return val


def preprocess(sentence, do_lemma=True):
    """
    Preprocessing the sentence into tokens
    :param sentence: a single string of tokens to process
    :return string: normalized, n-gram processed and may be lemmatized
    """
    if sentence:
        sentence = sentence.lower().strip()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        if 'with' in tokens:
            token_id = tokens.index('with')
            tokens = tokens[:token_id]
        # filtered_words = [w for w in tokens
        #                   if not w in stopwords.words('english')]
        # unicode_words = [getascii(w.strip().lower())
        #                  for w in filtered_words if w]
        unicode_words = [getascii(w.strip().lower())
                         for w in tokens if w]
        #hyphen_words = join_hyphen_words(unicode_words)
        if not do_lemma:
            return " ".join(unicode_words)
        else:
            lemmatized_words = [stem(w) for w in unicode_words]
            return " ".join(lemmatized_words)
    else:
        return None


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


def get_keywords(sentence, mapping_dict, wnl, tokenizer, language):
    keywords = []
    if not isinstance(sentence, str):
        sentence = ''
    if language.lower() != 'english':
        sentence.replace('>', ' ').replace('_', ' ')
        pt = parsetree(sentence, lemmata=True)
        tokens = pt[0].lemmata
    else:
        tokens = tokenizer.tokenize(sentence.replace('>', ' ').replace('_', ' '))
    for index, word in enumerate(tokens):
        try:
            if not isinstance(word, unicode):
                word = unicodedata.normalize('NFKD', unicode(
                    word.strip(), 'utf-8')).encode('ASCII', 'ignore')
                word = wnl.lemmatize(word.lower())
            else:
                word = word.strip().encode('ASCII', 'ignore')
                word = wnl.lemmatize(word.lower())
            if language.lower() != 'english':
                for e, s in categorical_words.iteritems():
                    if word == s:
                        word = e
                        break
            if word in mapping_dict[word[0]][len(word)]:
                keywords.append((index, word))
        except:
            continue
    return keywords


def get_gender(outjson):
    try:
        x = []
        #if outjson.get('gender', ''):
        #    if outjson['gender']:
        #        x.append(outjson['gender'])
        #    elif outjson.get('ontology', ''):
        #        x.append(outjson['ontology'].split(">")[0])
        #else:
        #    if outjson.get('ontology', ''):
        #        x.append(outjson['ontology'].split(">")[0])
        #    if outjson.get('title', ''):
        x.extend(preprocess(outjson['title']).split(' '))
        if x:
            x = [i.lower().strip().rstrip() for i in x]
    except:
        print(traceback.format_exc())
        return True, 'NA'
    unisex_list1 = ['men', 'women']
    unisex_list2 = ['woman', 'man']
    # Predict gender

    if not(set(unisex_list1) - set(x)):
        return True, 'unisex'
    if not(set(unisex_list2) - set(x)):
        return True, 'unisex'
    for checker_set in zip(unisex_list1,unisex_list2):
        if not(set(checker_set) - set(x)):
            return True, 'unisex'
    if set(x).intersection(['male', 'men', 'man']):
        return True, 'men'
    elif set(x).intersection(['female', 'women', 'woman']):
        return True, 'women'
    elif set(x).intersection(['boy', 'boys']):
        return True, 'boys'
    elif set(x).intersection(['girl', 'girls']):
        return True, 'girls'
    elif set(x).intersection(['unisex']):
        return True, 'unisex'
    elif set(x).intersection(['beb']):
        return True, 'baby'
    elif set(x).intersection(['-']):
        return True, 'NA'
    # if pd.isnull(x):
    #     return (True, 'NA')
    else:
        return True, 'NA'


with open(dir_path+'/reverse_masterontology_feb28.json') as f:
    json_file = json.load(f)
mapping_dict = create_new_map(json_file)


# ontologies_title = []

def predict(ontology='', title='', wnl=WordNetLemmatizer(), tokenizer=RegexpTokenizer(r'\w+'), language='english'):
    fuckall_brands = ['pepe jeans', 'armani jeans', 'tommy jeans', 'polo ralph lauren', 'ralph lauren',
                      'lauren ralph lauren', 'trussardi jeans', 'polo ralph ']
    for brand in fuckall_brands:
        title = title.lower().replace(brand, "")

    # prep_dict: 0-left side taken , 1-right side taken
    prep_dict = {'with': 0, 'for': 0, 'by': 0}
    for p, s in prep_dict.iteritems():
        if p in title.split(' '):
            if get_keywords(title.split(p, 1)[s], mapping_dict, wnl, tokenizer, language) == set([]):
                pass
            else:
                title = title.split(p, 1)[s]
    ontology_keywords = get_keywords(ontology, mapping_dict, wnl, tokenizer, language)
    title_keywords = get_keywords(title, mapping_dict, wnl, tokenizer, language)
    i = 3
    # print(ontology_keywords)
    # print(title_keywords)
    keyword_score_ontology = {}
    while i > 0:
        score = []
        rank = []
        new_keywords = ['@'.join(x3) for x3 in list(permutations([x2[1] for x2 in ontology_keywords], i))]
        # print(new_keywords)
        for tup in [x3 for x3 in list(permutations([x2[0] for x2 in ontology_keywords], i))]:
            # print(tup)
            rank.append(sum([x for x in tup]))
            score.append(sum([1.0 / y for y in [abs(x - tup[index - 1]) for index, x in enumerate(tup) if index > 0]]))
            # print(rank)
            # print(score)
        score = [(i - 1) * yy for yy in score]
        new_keywords = zip(score, new_keywords, rank)
        # print(new_keywords)
        fallback = list()
        if set([x[1] for x in new_keywords]).intersection(adjectives) != set([]):
            fallback.extend([x for x in new_keywords if x[1] in adjectives])
            new_keywords = list(set(new_keywords) - set([x for x in new_keywords if x[1] in adjectives]))
        for new_keyword in new_keywords:
            if new_keyword[1] in json_file.keys():
                keyword_score_ontology[new_keyword[1]] = (new_keyword[0], new_keyword[2])
        i -= 1
    if not bool(keyword_score_ontology) & (fallback != set([])):
        for adj in fallback:
            if adj[1] in json_file.keys():
                keyword_score_ontology[adj[1]] = (adj[0], adj[2])

    i = 3
    keyword_score_title = {}
    while i > 0:
        score = []
        rank = []
        new_keywords = ['@'.join(x3) for x3 in list(permutations([x2[1] for x2 in title_keywords], i))]
        print(new_keywords)
        for tup in [x3 for x3 in list(permutations([x2[0] for x2 in title_keywords], i))]:
            # print(tup)
            rank.append(sum([x for x in tup]))
            score.append(sum([1.0 / y for y in [abs(x - tup[index - 1]) for index, x in enumerate(tup) if index > 0]]))
            # print(rank)
            # print(score)
        score = [(i - 1) * yy for yy in score]
        new_keywords = zip(score, new_keywords, rank)
        # print(new_keywords)
        fallback = list()
        if set([x[1] for x in new_keywords]).intersection(adjectives) != set([]):
            fallback.extend([x for x in new_keywords if x[1] in adjectives])
            new_keywords = list(set(new_keywords) - set([x for x in new_keywords if x[1] in adjectives]))
        for new_keyword in new_keywords:
            if new_keyword[1] in json_file.keys():
                keyword_score_title[new_keyword[1]] = (new_keyword[0], new_keyword[2])
                # print(keyword_score_title)
        i -= 1
    # print(keyword_score_ontology)
    # print(keyword_score_title)
    # print(fallback)
    if not bool(keyword_score_title) & (fallback != set([])):
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
        if len([k for k, v in candidate_score_ontology.iteritems() if
                v == max(candidate_score_ontology.values())]) == 1:
            final_ontology_prediction = max(candidate_score_ontology, key=candidate_score_ontology.get)
        else:
            temp = [k for k, v in candidate_score_ontology.iteritems() if v == max(candidate_score_ontology.values())]
            temp2 = {k: v for k, v in candidate_rank_ontology.iteritems() if k in temp}
            final_ontology_prediction = max(temp2, key=temp2.get)
    except:
        final_ontology_prediction = ''

    try:
        if len([k for k, v in candidate_score_title.iteritems() if v == max(candidate_score_title.values())]) == 1:
            final_title_prediction = max(candidate_score_title, key=candidate_score_title.get)
        else:
            temp = [k for k, v in candidate_score_title.iteritems() if v == max(candidate_score_title.values())]
            temp2 = {k: v for k, v in candidate_rank_title.iteritems() if k in temp}
            final_title_prediction = max(temp2, key=temp2.get)
    except:
        final_title_prediction = ''
    return final_ontology_prediction, final_title_prediction


def assemble_ontologies(input_data, cp_ontology='msd_ontology', language='english'):
    if not isinstance(input_data['title'], unicode):
        input_data['title'] = unicodedata.normalize('NFKD', unicode(input_data.get('title'.strip()), 'utf-8')).\
            encode('ASCII', 'ignore')
    else:
        input_data['title'] = input_data['title'].strip().encode('ASCII', 'ignore')
    backup_ontology, predicted_ontology = predict(input_data.get('ontology'), input_data.get('title'),
                                                  language=language)
    _, gender = get_gender({'title': input_data['title']})
    if not predicted_ontology:
        predicted_ontology = backup_ontology
    if gender == 'NA' and input_data.get('gender'):
        gender = input_data.get('gender').lower()
    if predicted_ontology:
        predicted_ontology = gender + '>' + predicted_ontology
    if cp_ontology == 'msd_ontology':
        cp_ontology = predicted_ontology
    else:
        cp_ontology = input_data['ontology']
    output_dict = {'msd_ontology': predicted_ontology, 'cp_ontology': cp_ontology}
    return output_dict

if __name__ == '__main__':
    # with open('shoprunner_for_maria.csv') as f:
    #     ontologies_title = [tuple(line) for line in csv.reader(f)]
    # print "data fetched"
    # ontologies_title = ontologies_title[1:]
    # for entry in ontologies_title:
    #     predict(entry[2], entry[3])
    input1_file = sys.argv[1]
    output_file = sys.argv[2]
    fieldnames = ['mad_id', 'title', 'predicted_ontology', 'image_link', 'old_ontology', 'predicted_gender', 'gender']
    input1 = csv.DictReader(open(input1_file))
    output1 = csv.DictWriter(open(output_file, 'w'), fieldnames=fieldnames)
    output1.writeheader()

    # input1  = []
    # for elem in input11:
    #    if elem['ontology'] == 'womens>shoes>sandals':
    #        input1.append(elem)
    tic = default_timer()
    tokenizer = RegexpTokenizer(r'\w+')
    x = 0
    c = ''
    for entry in input1:
        output = {}
        print "{} completed".format(x)
        x += 1
        if not entry['title']:
            output["mad_id"] = entry["mad_id"]
            output["title"] = entry["title"]
            output['old_ontology'] = entry['ontology']
            output['predicted_ontology'] = ''
            output['gender'] = ''
            output['predicted_gender'] = ''
            output['image_link'] = ''
            output1.writerow(output)
            continue
        if not isinstance(entry['title'], unicode):
            entry['title'] = unicodedata.normalize('NFKD', unicode(entry['title'].strip(), 'utf-8')).encode('ASCII',
                                                                                                            'ignore')
        else:
            entry['title'] = entry['title'].strip().encode('ASCII', 'ignore')

        backup_ontology, output["predicted_ontology"] = predict(entry['ontology'], entry['title'], wnl, tokenizer,
                                                                language='english')

        _, gender = get_gender({'title': entry['title']})
        if not output.get("predicted_ontology"):
            output["predicted_ontology"] = backup_ontology
        if gender == 'NA' and entry.get('gender'):
            gender = entry.get('gender').lower()
        if output["predicted_ontology"]:
            output["predicted_ontology"] = gender + '>' + output["predicted_ontology"]

        output["mad_id"] = entry["mad_id"]
        output["title"] = entry["title"]
        output['old_ontology'] = entry['ontology']
        output['predicted_gender'] = gender
        output['image_link'] = entry['image_link']
        output['gender'] = entry['gender']
        output1.writerow(output)
