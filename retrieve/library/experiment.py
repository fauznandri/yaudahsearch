import re
import os
from retrieve.bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
from collections import defaultdict
import math
from retrieve.letor import Letor

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (1/math.log2(i+1))
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (1/k)
    return score



def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    score = 0.
    r = ranking.count(1) #aproksimasi banyaknya dokumen relevan di koleksi
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        if not r == 0:
            score += ranking[pos] * prec(ranking, len(ranking))/r
    return score
    

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    # with open(qrel_file) as file:
    #     content = file.readlines()

    # qrels_sparse = {}

    # for line in content:
    #     parts = line.strip().split()
    #     qid = parts[0]
    #     did = int(parts[1])
    #     if not (qid in qrels_sparse):
    #         qrels_sparse[qid] = {}
    #     if not (did in qrels_sparse[qid]):
    #         qrels_sparse[qid][did] = 0
    #     qrels_sparse[qid][did] = 1
    # return qrels_sparse
    
    # New codes using defaultdict
    qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
    dir = os.path.abspath(qrel_file)
    with open(dir) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels

    
# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="qrels-folder/test_queries.txt", k=1000):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    letor = Letor()
    
    dir = os.path.abspath(query_file)
    with open(dir) as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []
        
        rbp_scores_tfidf_letor = []
        dcg_scores_tfidf_letor = []
        ap_scores_tfidf_letor = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []
        
        rbp_scores_bm25_letor = []
        dcg_scores_bm25_letor = []
        ap_scores_bm25_letor = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            tfidf_docs = []
            for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                tfidf_docs.append(doc)
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))
            
            """
            Evaluasi TF-IDF dengan letor
            """
            ranking_tfidf_letor = []
            # nilai k1 dan b dapat diganti-ganti
            for (doc, score) in letor.rerank(query, tfidf_docs):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_tfidf_letor.append(1)
                else:
                    ranking_tfidf_letor.append(0)
            rbp_scores_tfidf_letor.append(rbp(ranking_tfidf_letor))
            dcg_scores_tfidf_letor.append(dcg(ranking_tfidf_letor))
            ap_scores_tfidf_letor.append(ap(ranking_tfidf_letor))

            """
            Evaluasi BM25
            """
            ranking_bm25 = []
            bm25_docs = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.2, b=0.5):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                bm25_docs.append(doc)
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))
            
            """
            Evaluasi BM25 dengan letor
            """
            ranking_bm25_letor = []
            # nilai k1 dan b dapat diganti-ganti
            for (doc, score) in letor.rerank(query, bm25_docs):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25_letor.append(1)
                else:
                    ranking_bm25_letor.append(0)
            rbp_scores_bm25_letor.append(rbp(ranking_bm25_letor))
            dcg_scores_bm25_letor.append(dcg(ranking_bm25_letor))
            ap_scores_bm25_letor.append(ap(ranking_bm25_letor))
            

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
    print()
    print("Hasil evaluasi TF-IDF dengan letor terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf_letor) / len(rbp_scores_tfidf_letor))
    print("DCG score =", sum(dcg_scores_tfidf_letor) / len(dcg_scores_tfidf_letor))
    print("AP score  =", sum(ap_scores_tfidf_letor) / len(ap_scores_tfidf_letor))
    print()
    print("Hasil evaluasi BM25 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))
    print()
    print("Hasil evaluasi BM25 dengan letor terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_letor) / len(rbp_scores_bm25_letor))
    print("DCG score =", sum(dcg_scores_bm25_letor) / len(dcg_scores_bm25_letor))
    print("AP score  =", sum(ap_scores_bm25_letor) / len(ap_scores_bm25_letor))
    
    # tulis ke dalam file evaluation.txt
    with open (os.path.abspath("evaluasi.txt"), "w") as f:
        f.write("Hasil evaluasi TF-IDF terhadap 150 queries\n")
        f.write("RBP score = " + str(sum(rbp_scores_tfidf) / len(rbp_scores_tfidf)) + "\n")
        f.write("DCG score = " + str(sum(dcg_scores_tfidf) / len(dcg_scores_tfidf)) + "\n")
        f.write("AP score  = " + str(sum(ap_scores_tfidf) / len(ap_scores_tfidf)) + "\n")
        f.write("\n")
        f.write("Hasil evaluasi TF-IDF dengan letor terhadap 150 queries\n")
        f.write("RBP score = " + str(sum(rbp_scores_tfidf_letor) / len(rbp_scores_tfidf_letor)) + "\n")
        f.write("DCG score = " + str(sum(dcg_scores_tfidf_letor) / len(dcg_scores_tfidf_letor)) + "\n")
        f.write("AP score  = " + str(sum(ap_scores_tfidf_letor) / len(ap_scores_tfidf_letor)) + "\n")
        f.write("\n")
        f.write("Hasil evaluasi BM25 terhadap 150 queries\n")
        f.write("RBP score = " + str(sum(rbp_scores_bm25) / len(rbp_scores_bm25)) + "\n")
        f.write("DCG score = " + str(sum(dcg_scores_bm25) / len(dcg_scores_bm25)) + "\n")
        f.write("AP score  = " + str(sum(ap_scores_bm25) / len(ap_scores_bm25)) + "\n")
        f.write("\n")
        f.write("Hasil evaluasi BM25 dengan letor terhadap 150 queries\n")
        f.write("RBP score = " + str(sum(rbp_scores_bm25_letor) / len(rbp_scores_bm25_letor)) + "\n")
        f.write("DCG score = " + str(sum(dcg_scores_bm25_letor) / len(dcg_scores_bm25_letor)) + "\n")
        f.write("AP score  = " + str(sum(ap_scores_bm25_letor) / len(ap_scores_bm25_letor)) + "\n")


if __name__ == '__main__':
    qrels = load_qrels()

    # assert qrels["Q1002252"][5599474] == 1, "qrels salah"
    # assert not (6998091 in qrels["Q1007972"]), "qrels salah"

    eval_retrieval(qrels)
