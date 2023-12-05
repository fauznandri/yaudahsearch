from django.shortcuts import render
from django.http import HttpResponse
from .library.search import retrieve
from .library.bsbi import BSBIIndex
from .library.compression import VBEPostings

import sys
import os
import re

def index(request):
    query = request.GET.get('search_bar')
    this_dir = os.path.dirname(__file__)
    BSBI_instance = BSBIIndex(data_dir='library/collections',
                            postings_encoding=VBEPostings,
                            output_dir=os.path.join(this_dir,'library/index'))
    BSBI_instance.load()
    
    if query == None or query == "":
        context = {
            'query': query,
            'signal': -1
        }
        return render(request, 'retrieve/index.html', context)
    else:
        result = {}
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        result_raw = retrieve(100, query)

        if result_raw == None:
            context = {
                'result': result,
                'query': query,
                'signal': 0
            }
            return render(request, 'retrieve/index.html', context)
        else:
            for doc in result_raw:
                text = open(doc).read()
                # text = text.lower()
                did = BSBI_instance.doc_id_map[doc]
                doc_name = os.path.basename(doc)
                result[did] = [doc_name, text]

            context = {
                'result': result,
                'query': query,
                'signal': 1
            }
            return render(request, 'retrieve/index.html', context)

def content(request, doc_id):
    this_dir = os.path.dirname(__file__)
    BSBI_instance = BSBIIndex(data_dir='library/collections',
                            postings_encoding=VBEPostings,
                            output_dir=os.path.join(this_dir,'library/index'))
    BSBI_instance.load()
    doc = BSBI_instance.doc_id_map[int(doc_id)]
    
    # print(doc_id)
    # print(BSBI_instance.doc_id_map[int(doc_id)])
    
    doc_name = os.path.basename(doc)
    text = open(doc).read()
    # text = text.lower()

    context = {
        'doc_name': doc_name,
        'text': text,
    }
    return render(request, 'retrieve/content.html', context)
