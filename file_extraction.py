#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:17:48 2025

@author: aaronfoote
"""

#File designed to download and extract the needed URLs from Founders online

import json
import requests
import time
import random
import os

#this referenced json file contains metadata information regarding the ids of
#the documents contained within the founders online database.
#It can be used to find the id info of letters of interest, in this case all 
#letters written either by Jefferson or Hamilton
with open('founders-online-metadata.json', 'r') as file:
    total_docs = json.load(file)

#Function that returns the document id of all documents contained within total_docs
#that were written by a particular author
#The author must be input in the form ['lastname, firstname'] to match the json dictionary format   
def extract_author_doc_ids(total_docs, author):
    doc_ids = []
    for doc in total_docs:
        if doc['authors'] == author:
            doc_ids.append(doc['permalink'][40:]) #this selects the part of the id that needs to be appended to the API downstream
    return doc_ids

#the extract_author_doc_id function is applied twice to collect ids for all 
#documents written by either Hamilton or Jefferson
jefferson_doc_ids = extract_author_doc_ids(total_docs, ['Jefferson, Thomas'])
hamilton_doc_ids = extract_author_doc_ids(total_docs, ['Hamilton, Alexander'])

#Comparing the length of doc_id lists shows that there are much fewer
#documents written by Hamilton than by Jefferson, leading to an imbalanced dataset in the future
#the ids from the more prolific writer should be randomly shuffled and have a subset
#of ids extracted equal to the number of documents written by the less prolific writer
#the following funciton performs this

def equilibrate_doc_ids(id_list_1, id_list_2):
    num_ids = min(len(id_list_1), len(id_list_2))
    if len(id_list_1) <= len(id_list_2):
        random.shuffle(id_list_2)
        id_list_2 = id_list_2[:num_ids]
    else:
        random.shuffle(id_list_1)
        id_list_1 = id_list_1[:num_ids]
    return id_list_1, id_list_2
    
jefferson_doc_ids, hamilton_doc_ids = equilibrate_doc_ids(jefferson_doc_ids, hamilton_doc_ids)


#Now that the lists of ids for both Hamilton and Jefferson have been created
#the following file structure needs to be created:
#Jefferson_Hamilton_Classification/
#...train/
#......jefferson/
#......hamilton/
#...test/
#......jefferson/
#......hamilton/

#First all Jefferson and Hamilton files will be extracted from the Founders Online API
#it is requested on the API documentation to have a maximum of 10 downloads per second to avoid overstressing the service
#a time delay in seconds will be included using time.sleep()

base_URL = 'https://founders.archives.gov/API/docdata/'

test_URL = base_URL + jefferson_doc_ids[0]

def download_doc(base_URL, doc_id, Folder):
    doc_URL = base_URL + doc_id
    response = requests.get(doc_URL) #get document from API
    doc_content = response.json()['content']
    doc_content = doc_content.replace('\n', ' ')#processing step to clean up text
    doc_name = doc_id + '.txt'#the next few lines are to remove part of the doc_id to make a better file name
    separator = '/'
    file_parts = doc_name.split(separator)
    doc_name = file_parts[-1]
    complete_path = os.path.join(Folder, doc_name)
    os.makedirs(Folder, exist_ok=True)
    with open(complete_path, 'w') as file:
        file.write(doc_content)
    print(f"Document ID: {doc_id} saved to folder")

#download_document(base_URL, jefferson_doc_ids[0], 'Jefferson')

def download_all_docs(base_URL, doc_id_list, Folder):
    for doc_id in doc_id_list:
        delay_seconds = 0.5
        time.sleep(delay_seconds)
        download_doc(base_URL, doc_id, Folder)

#to prevent imbalances in the number of files in a folder, resulting in having
#many more samples of one class over another, the following function removes 
#the selected number of files from the indicated file path at random
from random import sample

def remove_files(file_path, num_files):    
    files = os.listdir(file_path)
    for file in sample(files, num_files):
        sample_path = file_path+'/'+file
        os.remove(sample_path)
        
#The following function is used to reorganize the text files into train and test
#folders with Jefferson and Hamilton writings underneath them each.
#At the time of writing, there are two folders that contain all jefferson and hamilton
#text files, respectively.

source_jeff = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/Jefferson'
source_ham = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/Hamilton'

dest_train_jeff = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/train/Jefferson'
dest_train_ham = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/train/Hamilton'
dest_val_jeff = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/val/Jefferson'
dest_val_ham = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/val/Hamilton'
dest_test_jeff = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/test/Jefferson'
dest_test_ham = '/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/test/Hamilton'

def reorg_files(source, destination, num_files):
    allfiles = os.listdir(source)
    for f in sample(allfiles, num_files):
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        os.rename(src_path, dst_path)

#I will split the text files so that 50% of them are in the train folder,
#20% of them are in the validation folder, and 30% of them are in the test folder 
#for both Jefferson and Hamilton.

#Since there are 5400 files for each person, this gives us 2700 files for
#training, 1080 files for validation, and 1620 files for testing, respecively
#the reorg_files function was run with the following arguments:
    
#reorg_files(source_jeff, dest_train_jeff, 2700)
#reorg_files(source_jeff, dest_val_jeff, 1080)
#reorg_files(source_jeff, dest_test_jeff, 1620)
#reorg_files(source_ham, dest_train_ham, 2700)
#reorg_files(source_ham, dest_val_ham, 1080)
#reorg_files(source_ham, dest_test_ham, 1620)































