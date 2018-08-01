import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import Levenshtein
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
from scipy import spatial

stemmer = SnowballStemmer('english')

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word) >= 0) for word in str1.split())


print "loading data ..."
df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1")

print "process 'search_term'..."
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
print "process 'product_title'..."
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
print "process 'product_description'..."
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

# create homemade feature
df_all['dist_in_title'] = df_all.apply(lambda x: Levenshtein.ratio(x['search_term'], x['product_title']), axis=1)
df_all['dist_in_desc'] = df_all.apply(lambda x: Levenshtein.ratio(x['search_term'], x['product_description']), axis=1)

#TF-iDF
df_all['all_texts'] = df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)

class MyCorpus(object):
        def __iter__(self):
                for x in df_all['all_texts'].values:
                        yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))
corpus = MyCorpus()
tfidf = TfidfModel(corpus)

def to_tfidf(text):
        res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
        return res

def cos_sim(text1, text2):
        tfidf1 = to_tfidf(text1)
        tfidf2 = to_tfidf(text2)
        index = MatrixSimilarity([tfidf1], num_features=len(dictionary))
        sim = index[tfidf2]
        return float(sim[0])

df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)

#Word2Vec
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
sentences = [y for x in sentences for y in x]
w2v_corpus = [word_tokenize(x) for x in sentences]
model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)

vocab = model.wv.vocab

def get_vector(text):
	res = np.zeros([128])
	count = 0
	for word in word_tokenize(text):
		if word in vocab:
			res += model[word]
			count += 1
	return res/count

def w2v_cos_sim(text1, text2):
	try:
		w2v1 = get_vector(text1)
		w2v2 = get_vector(text2)
		sim = 1 - spatial.distance.cosine(w2v1, w2v2)
		return float(sim)
	except:
		return float(0)
df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)

df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'all_texts'], axis=1)

print "save to df_all_feature.csv"
df_all.to_csv('df_all_feature.csv')

