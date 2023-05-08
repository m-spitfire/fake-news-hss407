import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import numpy as np
from tqdm import tqdm

data_lemmatized = pickle.load(open("data_lemmatized_fake.pkl", "rb"))
dirichlet_dict = corpora.Dictionary.load("./dictionary_fake.gensim")

num_keywords = 15
num_topics = list(range(4, 51, 2))

LDA_models = {}
LDA_topics = {}
for i in range(4, 51, 2):
    LDA_models[i] = LdaMulticore.load(f"models_fake/{i}_multi_symm.gensim")

for i in range(4, 51, 2):
    shown_topics = LDA_models[i].show_topics(num_topics=i,
                                              num_words=num_keywords,
                                              formatted=False)
    LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]



def jaccard_similarity(topic_1, topic_2):
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection))/float(len(union))

LDA_stability = {}
for i in range(0, len(num_topics)-1):
    jaccard_sims = []
    for t1, topic1 in enumerate(LDA_topics[num_topics[i]]):
        sims = []
        for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]):
            sims.append(jaccard_similarity(topic1, topic2))

        jaccard_sims.append(sims)

    LDA_stability[num_topics[i]] = jaccard_sims

mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]


coherences = []
for i in tqdm(num_topics[:-1]):
    model = CoherenceModel(model=LDA_models[i], texts=data_lemmatized, dictionary=dirichlet_dict, coherence="c_v")
    coherences.append(model.get_coherence())

coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(len(num_topics))[:-1]]
max_val = max(coh_sta_diffs)
max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == max_val]
ideal_topic_num = num_topics[max_idxs[0]]

plt.figure(figsize=(20.0,20.0/1.618))
ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
ax.set_ylim([0.2, y_max])
ax.set_xlim([1, num_topics[-1]-1])

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(5, end, 5))
ax.tick_params(axis="both", which="major", labelsize=25)
ax.set_ylabel('Metric Level', fontsize=30)
ax.set_xlabel('Number of Topics', fontsize=30)
plt.legend(fontsize=30)
plt.savefig("fake_final.png")
