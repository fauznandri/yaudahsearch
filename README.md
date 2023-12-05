# YaudahSearch
Final Project for Information Retrieval Course Odd Semester AY 2023/2024     
Faculty of Computer Science Universitas Indonesia

## How to Install
- git clone https://github.com/fauznandri/yaudahsearch.git
- extract collection-index-qrels.zip using 'extract here'
- pip install -r requirements.txt
- python manage.py runserver

## Description
An information retrieval model built based on the inverted index data structure and an implementation of TF-IDF retrieval scheme. \
Used MPStemmer (https://github.com/ariaghora/mpstemmer) for stemming and LightGBM (https://lightgbm.readthedocs.io/en/stable/) for learning-to-rank SERP optimization. \
The application it self was built using Django version 4.2.7

## NOTE ##
The initialization will take time as it will run the indexing and train the letor everytime it initializes. 
Estimated time is about ~10 minutes.
