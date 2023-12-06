from django.urls import path

from . import views
from .library.bsbi import BSBIIndex
from .library.compression import VBEPostings
from .library.letor import Letor

import os

app_name = "retrieve"

urlpatterns = [
    path("", views.index, name="index"),
    path("content/<str:doc_id>/", views.content, name="content")
]

# # Melakukan indexing documents collection dan training letor 
# # pada saat pertama kali runserver
# print("Tunggu sebentar, sedang indexing collections...")
# this_dir = os.path.dirname(__file__)
# BSBI_instance = BSBIIndex(data_dir=os.path.join(this_dir, 'library/collections'),
#                             postings_encoding=VBEPostings,
#                             output_dir=os.path.join(this_dir, 'library/index'))

# BSBI_instance.do_indexing()

# letor = Letor()
    
# letor.train("qrels-folder/train_queries.txt", 
#             "qrels-folder/train_docs.txt", 
#             "qrels-folder/train_qrels.txt")
