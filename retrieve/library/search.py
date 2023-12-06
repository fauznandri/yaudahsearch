from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import Letor

import os
    
def retrieve(k = 100, query = '', bsbi=None):
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut\
        
    if not bsbi is None:
        BSBI_instance = bsbi
    else:
        this_dir = os.path.dirname(__file__)
        BSBI_instance = BSBIIndex(data_dir=os.path.join(this_dir, 'collections'),
                                postings_encoding=VBEPostings,
                                output_dir=os.path.join(this_dir,'index'))

    # initialize letor
    letor = Letor()
    
    # letor.train("qrels-folder/train_queries.txt", 
    #           "qrels-folder/train_docs.txt", 
    #           'qrels-folder/train_qrels.txt')
    
    # membuat query menjadi lower, karena docs di index dengan huruf2 kecil
    query = query.lower()
    
    bm25_docs = []
    result = []
    # retrieval menggunakan BM25
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        bm25_docs.append(doc)
    if not bm25_docs == []:
        # rerank hasil BM25 menggunakan letor
        for (doc, score) in letor.rerank(query, bm25_docs):
            result.append(doc)
    
    return result

if __name__ == '__main__':
    retrieve()