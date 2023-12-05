from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import Letor

import os
    
def retrieve(k = 100, query = ''):
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    this_dir = os.path.dirname(__file__)
    BSBI_instance = BSBIIndex(data_dir=os.path.join(this_dir, 'collections'),
                            postings_encoding=VBEPostings,
                            output_dir=os.path.join(this_dir,'index'))

    # initialize letor
    letor = Letor()
    
    # letor.train("qrels-folder/train_queries.txt", 
    #           "qrels-folder/train_docs.txt", 
    #           'qrels-folder/train_qrels.txt')
    
    bm25_docs = []
    # result = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        bm25_docs.append(doc)
            
    # for (doc, score) in letor.rerank(query, bm25_docs):
    #     result.append(doc)
        
    return bm25_docs

if __name__ == '__main__':
    retrieve()