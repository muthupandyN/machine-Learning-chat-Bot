from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import csv
import json
import timeit
import random

def talk_to_cb_primary(test_set_sentence, minimum_score , json_file_path , tfidf_vectorizer_pikle_path ,tfidf_matrix_train_pikle_path):
    test_set = (test_set_sentence, "")

    try:
        
        f = open(tfidf_vectorizer_pikle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()

        f = open(tfidf_matrix_train_pikle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        f.close()
        
    except:
        
        tfidf_vectorizer , tfidf_matrix_train = train_chat(json_file_path , tfidf_vectorizer_pikle_path , tfidf_matrix_train_pikle_path)
        

    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

    cosine = np.delete(cosine, 0)
    max = cosine.max()
    response_index = 0
    if (max > minimum_score):
        new_max = max - 0.01
        list = np.where(cosine > new_max)
        
        response_index = random.choice(list[0])
    else :
        return "sorry not yet trained these kind of things" , 0
           
   

    j = 0

    with open(json_file_path, "r") as sentences_file:
        reader = json.load(sentences_file)
        for row in reader:
            j += 1 
            if j == response_index:

                
                return row["response"], max
                break




def train_chat(json_file_path, tfidf_vectorizer_pikle_path , tfidf_matrix_train_pikle_path):
        
        i = 0
        sentences = []
        sentences.append(" No you.")
        sentences.append(" No you.")

        start = timeit.default_timer()
        with open(json_file_path, "r") as sentences_file:
            reader = json.load(sentences_file)
            for row in reader:
                sentences.append(row["message"])
                i += 1

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)  
        stop = timeit.default_timer()
        print ("training time took was :   congrats it was finished")
        print (stop - start)

        f = open(tfidf_vectorizer_pikle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()

        f = open(tfidf_matrix_train_pikle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()

        return tfidf_vectorizer , tfidf_matrix_train
       

def faq(query):
    minimum_score = 0.7
    file = "faq.json"
    tfidf_vectorizer_pikle_path = "faq_tfidf_vectorizer.pickle"
    tfidf_matrix_train_path = "faq_tfidf_matrix_train.pickle"
    query_response, score = talk_to_cb_primary(query , minimum_score , file , tfidf_vectorizer_pikle_path , tfidf_matrix_train_path)
    return query_response

while 1:
    sent = input("YOU: ")
    print(faq(sent))


