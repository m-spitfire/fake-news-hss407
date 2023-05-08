import nltk
import pandas as pd
import spacy
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
from tqdm import tqdm
import re
import pickle

nlp = spacy.load('en_core_web_sm')
en_stop = set(nltk.corpus.stopwords.words('english'))

dataset = pd.read_csv("./Fake.csv")
dataset['text_processed'] = dataset['text'].map(lambda x: re.sub('[,\.!?]', '', x))
dataset['text_processed'].head()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))

data = dataset.text_processed.values.tolist()
data_words = list(sent_to_words(data))


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in en_stop] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
pickle.dump(data_lemmatized, open("data_lemmatized_fake.pkl", "wb"))

dirichlet_dict = corpora.Dictionary(data_lemmatized)
bow_corpus = [dirichlet_dict.doc2bow(text) for text in data_lemmatized]
pickle.dump(bow_corpus, open('corpus_fake.pkl', 'wb'))
dirichlet_dict.save('dictionary_fake.gensim')

num_keywords = 15
num_topics = list(range(4, 51, 2))
LDA_models = {}

for i in tqdm(num_topics):
    LDA_models[i] = LdaMulticore(corpus=bow_corpus,
                             id2word=dirichlet_dict,
                             num_topics=i,
                             chunksize=len(bow_corpus),
                            # TODO: increase passes
                             passes=1,
                             workers=15,
                             random_state=42)
    LDA_models[i].save(f"models_fake/{i}_multi_symm.gensim")
