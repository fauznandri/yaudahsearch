import lightgbm as lgb
import numpy as np
import random
import joblib
import pickle
import os

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine, minkowski

class Letor:
  
  def __init__(self):
    self.NUM_LATENT_TOPICS = 400
    self.NUM_NEGATIVES = 1
    self.dictionary = Dictionary()
    self.model = None
    self.ranker = None
    
    # load model dan ranker apabila sudah tersedia
    try:
      with open(os.path.join(os.path.abspath('retrieve/library'), 'lsi.model'), "rb") as f:
        self.model = pickle.load(f)
    except:
      print("Model belum di-train")
      
    try:
      self.ranker = joblib.load(os.path.join(os.path.abspath('retrieve/library'), 'ranker.joblib'))
    except:
      print("ranker belum di train")

  # Extract documents and queries from txt file
  def extract_txt(self, file_dir):
    res = {}
    with open(os.path.join(os.path.abspath("retrieve/library"), file_dir), encoding="utf8") as file:
      for line in file:
        text = line.split(" ")
        id = text[0]
        content = text[1:-1]
        res[id] = content
    return res

  def dataset_from_txt(self, file_dir, queries, docs):
    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open(os.path.join(os.path.abspath('retrieve/library'), file_dir)) as file:
      for line in file:
        q_id, doc_id, rel = line.split(" ")[0:3]
        if (q_id in queries) and (doc_id in docs):
          if q_id not in q_docs_rel:
            q_docs_rel[q_id] = []
          q_docs_rel[q_id].append((doc_id, int(rel)))

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
      docs_rels = q_docs_rel[q_id]
      group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
      for doc_id, rel in docs_rels:
        dataset.append((queries[q_id], docs[doc_id], rel))
      # tambahkan satu negative (random sampling saja dari documents)
      dataset.append((queries[q_id], random.choice(list(docs.values())), 0))
    
    return (group_qid_count, dataset)
    

  # representasi vector dari sebuah dokumen & query
  def vector_rep(self, text):
    rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
    return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

  # kita ubah dataset menjadi terpisah X dan Y
  # dimana X adalah representasi gabungan query+document,
  # dan Y adalah label relevance untuk query dan document tersebut.
  # 
  # Bagaimana cara membuat representasi vector dari gabungan query+document?
  # cara simple = concat(vector(query), vector(document)) + informasi lain
  # informasi lain -> cosine distance & jaccard similarity antara query & doc
  def features(self, query, doc):
    v_q = self.vector_rep(query)
    v_d = self.vector_rep(doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

  def train(self, queries_dir, docs_dir, qrels_dir):
    queries = self.extract_txt(queries_dir)
    
    documents = self.extract_txt(docs_dir)
    
    group_qid_count, dataset = self.dataset_from_txt(qrels_dir, queries, documents)
    
    # # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
    # # dari kumpulan  dokumen.
    bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
    model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 400 latent topics
    self.model = model
    
    # dump model ke dalam sebuah file agar tidak perlu train setiap kali
    with open(os.path.join(os.path.abspath('retrieve/library'), 'lsi.model'), "wb") as f:
      pickle.dump(model, f)

    X = []
    Y = []
    for (query, doc, rel) in dataset:
        X.append(self.features(query, doc))
        Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        # boosting_type = "rf",
                        bagging_freq = 1,
                        bagging_fraction = 0.6,
                        feature_fraction = 0.8,
                        n_estimators = 200,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 20,
                        learning_rate = 0.01,
                        max_depth = 5,
                        # max_depth = 3,
                        verbose = 10, #uncomment if needed to be verbose
                        n_jobs = 8,
                        # device = "gpu"
                        )

    ranker.fit(X, Y,
            group = group_qid_count)
    
    self.ranker = ranker
    # dump ranker
    joblib.dump(ranker, os.path.join(os.path.abspath('retrieve/library'), 'ranker.joblib'))
    
  def rerank(self, query, documents_dir):
    docs = []
    
    for dir in documents_dir:
      with open(os.path.join(os.path.abspath('retrieve/library'), dir), encoding="utf8") as file:
        for line in file:
          content = line.split(" ")
          docs.append(content)
    
    X = []    
    for doc in docs:
      X.append(self.features(query.split(), doc))
    
    X = np.array(X)
    
    scores = self.ranker.predict(X)
    did_scores = [x for x in zip(documents_dir, scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
    
    return sorted_did_scores
      
if __name__ == "__main__":
  
  letor = Letor()
  # train the letor and save the ranker and model to
  # ranker.joblib and lsi.model
  letor.train("qrels-folder/train_queries.txt", 
              "qrels-folder/train_docs.txt", 
              'qrels-folder/train_qrels.txt')


  
  
    
    