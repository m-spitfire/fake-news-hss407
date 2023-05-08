import pickle
from wordcloud import WordCloud

data_fake = pickle.load(open("./data_lemmatized_fake.pkl", "rb"))
data_real = pickle.load(open("./data_lemmatized_real.pkl", "rb"))
wc_fake = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wc_real = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

wc_fake.generate(text=" ".join([item for sublist in data_fake for item in sublist]))
wc_real.generate(text=" ".join([item for sublist in data_real for item in sublist]))
wc_fake.to_file("fake_wc.png")
wc_real.to_file("real_wc.png")
