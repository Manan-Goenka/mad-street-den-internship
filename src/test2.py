from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
word = "multiplying"
print(lem.lemmatize(word))
ex_sent="This is an example showing stop word filtration."
stop_words=set(stopwords.words("english"))
words=word_tokenize(ex_sent)
filtered_sentence=[]

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)
