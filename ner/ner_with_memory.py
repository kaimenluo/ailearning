import pandas as pd
import numpy as np

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
print(data.tail(10))

words = list(set(data["Word"].values))
n_words = len(words)

print(n_words,"\n")

class SentenceGetter(object):
  def __init__(self,data):
    self.data = data
    self.n_sent = 1
    self.empty = False

  def get_next(self):
    try:
      s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
      self.n_sent += 1
      return s["Word"].values.tolist(),s["POS"].values.tolist(), s["Tag"].values.tolist()
    except:
      self.empty = True
      return None,None,None

getter = SentenceGetter(data)
sent,pos,tag = getter.get_next()
print(sent,pos,tag)

from sklearn.base import BaseEstimator,TransformerMixin

class MemoryTagger(BaseEstimator,TransformerMixin):
  def fit(self,X,y):
    voc = {}
    self.tags = []
    for w, t in zip(X,y):
      if t not in self.tags:
        self.tags.append(t)
      if w not in voc:
        voc[w] = {}
      if t not in voc[w]:
        voc[w][t] = 0
      voc[w][t] += 1
    self.memory = {}
    for k, d in voc.items():
      self.memory[k] = max(d,key=d.get)
  def predict(self,X,y=None):
    return[self.memory.get(x,"O") for x in X]
# 存储形式： voc = {w1: {tag1:f1, tag2:f2...},w2: {tag1:f1, tag2:f2...},w3: {tag1:f1, tag2:f2...}...}

# 测试一下
tagger = MemoryTagger()
tagger.fit(sent,tag)
print(tagger.predict(sent))

# 用所有数据进行训练
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()
pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report, "\n")

from sklearn.ensemble import RandomForestClassifier
def feature_map(word):
  return np.array([word.istitle(),word.islower(),word.isupper(),len(word),word.isdigit(),word.isalpha()])

words = [feature_map(w) for w in data["Word"].values.tolist()]
# print(words)
pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words, y=tags, cv=5)
report = classification_report(y_pred=pred,y_true=tags)
print(report)

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class FeatureTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.memory_tagger = MemoryTagger()
    self.tag_encoder, self.pos_encoder = LabelEncoder(), LabelEncoder()

    # s = data[data["Sentence #"] == "Sentence: {}".format(1)]
    # print(s)

  def fit(self, X, y):
    words = X["Word"].values.tolist()
    self.pos = X["POS"].values.tolist()
    tags = X["Tag"].values.tolist()
    self.memory_tagger.fit(words, tags)
    self.tag_encoder.fit(tags)
    self.pos_encoder.fit(self.pos)
    return self  # fit函数返回的结果就是self, 允许链式调用

  def transform(self, X, y=None):
    def pos_default(p):
      if p in self.pos:
        return self.pos_encoder.transform([p])[0]
      else:
        return -1

    pos = X["POS"].values.tolist()
    words = X["Word"].values.tolist()
    out = []

    print(len(words))

    for i in range(len(words)):
      print(i)

      w = words[i]
      p = pos[i]
      if i < len(words) - 1:

        # test_1 = words[i + 1]
        # test_2 = self.memory_tagger.predict([words[i + 1]])
        # test_3 = self.tag_encoder.transform(self.memory_tagger.predict([words[i + 1]]))

        wp = self.tag_encoder.transform(self.memory_tagger.predict([words[i + 1]]))[0]
        posp = pos_default(pos[i + 1])
      else:
        wp = self.tag_encoder.transform(['O'])[0]
        posp = pos_default(".")
      if i > 0:
        if words[i - 1] != ".":
          wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i - 1]]))[0]
          posm = pos_default(pos[i - 1])
        else:
          wm = self.tag_encoder.transform(['O'])[0]
          posm = pos_default(".")
      else:
        posm = pos_default(".")
        wm = self.tag_encoder.transform(['O'])[0]

      test_array = np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                           self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                           pos_default(p), wp, wm, posp, posm])

      out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                           self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                           pos_default(p), wp, wm, posp, posm]))
    return out

# from sklearn.pipeline import Pipeline
# pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()), ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),X=data, y=tags, cv=5)
# report = classification_report(y_pred=pred, y_true=tags)
# print(report)

featureTrans = FeatureTransformer()
featureTrans.fit(data, tags)
X_data = featureTrans.transform(data)
# print(X_data)
pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=X_data, y=tags, cv=5)
report = classification_report(y_pred=pred,y_true=tags)
print(report)
