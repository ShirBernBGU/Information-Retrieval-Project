from nltk.corpus import stopwords
import builtins
import math
import hashlib
from inverted_index_gcp import *
from nltk.stem.porter import *
import threading
import nltk
import re
import pickle
from google.cloud import storage


def _hash(s):
    """
    Compute the hash of a string using the Blake2b algorithm.

    Parameters:
    - s (str): The input string to be hashed.

    Returns:
    - str: The hashed value of the input string.
    """
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')

BUCKET_NAME = "209706803"

# basic title inverted index
basic_title_bucket_name = "209706803"
file_path = "POSTING_LISTS_TITLE/index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(basic_title_bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
inverted_title_basic = pickle.loads(contents)

# stem title inverted index
title_with_stem_bucket_name = "209706803"
file_path = "POSTING_LISTS_TITLE_WITH_STEM/index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(title_with_stem_bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
inverted_title_with_stem = pickle.loads(contents)

# basic body inverted index
basic_body_bucket_name = "209706803"
file_path = "POSTING_LISTS_BODY_NO_STEM/index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(basic_body_bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
inverted_body_basic = pickle.loads(contents)

# STEM body inverted index
basic_body_bucket_name = "209706803"
file_path = "POSTING_LISTS_BODY_WITH_STEM/index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(basic_body_bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
inverted_body_with_stem = pickle.loads(contents)

# load id_title_dict dictionary {doc_id: doc_title, ....}
bucket_name_for_title = "209706803"
file_path = "id_title_dict.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name_for_title)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
id_title_dict = pickle.loads(contents)

# load doc_id_pagerank dictionary {doc_id: pagerank ....}
bucket_name_for_pagerank = "209706803"
file_path = "doc_id_pagerank_dict.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name_for_pagerank)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_id_page_rank_dict = pickle.loads(contents)

min_max_pagerank = 9913.578661311381

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

stemmer = PorterStemmer()


def tokenizer(text, stem=False, ngram=False):
    """
    Tokenize the input text.

    Parameters:
    - text (str): The input text to be tokenized.
    - stem (bool): A flag indicating whether to perform stemming. Default is False.
    - ngram (bool): A flag indicating whether to generate n-grams. Default is False.

    Returns:
    - list: A list of tokens extracted from the input text.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    if stem and not ngram:
        tokens = [stemmer.stem(token) for token in tokens if token not in all_stopwords]

    elif ngram and not stem:
        tokens = [token[0] + " " + token[1] for token in list(nltk.bigrams(tokens))]

    elif stem and ngram:
        tokens = [stemmer.stem(token) for token in tokens if token not in all_stopwords]
        tokens = [token[0] + " " + token[1] for token in list(nltk.bigrams(tokens))]

    else:
        tokens = [token for token in tokens if (token not in all_stopwords)]

    return tokens


def find_relevant_docs(inverted_index, directory, tokenized_query):
    """
    Find relevant documents based on the inverted index and tokenized query.

    Parameters:
    - inverted_index: The inverted index object.
    - directory: The directory to search in.
    - tokenized_query: A list of tokens representing the query.

    Returns:
    - tuple: A tuple containing a set of relevant document IDs and a dictionary of document IDs and their corresponding term frequencies.
    """
    # Create a set to store relevant document IDs
    doc_set = set()
    # Create a dictionary to store document IDs and term frequencies
    doc_term_freq = {}

    for token in tokenized_query:
        pl_token = inverted_index.read_a_posting_list(base_dir=directory, w=token,
                                                      bucket_name=BUCKET_NAME)  # (term, (doc_id, freq))

        for doc_id, freq in pl_token:
            doc_set.add(doc_id)  # Add the document ID to the set

            # Update the term frequency for the document
            if doc_id in doc_term_freq:
                doc_term_freq[doc_id][token] = freq
            else:
                doc_term_freq[doc_id] = {token: freq}

    return doc_set, doc_term_freq


def get_top_n(sim_dict, N=30):
    """
    Retrieve the top N documents based on the similarity dictionary.

    Parameters:
    - sim_dict: A dictionary containing document IDs and their similarity scores.
    - N: The number of documents to retrieve. Default is 30.

    Returns:
    - list: A list of tuples containing the top N document IDs and their similarity scores.
    """
    sorted_docs = sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:N]


def is_common_word(query):
    """
    Check if the query contains common words.

    Parameters:
    - query (str): The query string.

    Returns:
    - bool: True if the query contains common words, otherwise False.
    """
    common_words = {'name', 'first', 'issue', 'head', 'use', 'question', 'provide', 'ever', 'money', 'lose', 'put',
                    'sit', 'home', 'bring', 'story', 'different', 'way', 'also', 'young', 'like', 'family', 'million',
                    'day', 'far', 'line', 'back', 'community', 'old', 'lot', 'find', 'life', 'word', 'white', 'thing',
                    'high', 'time', 'hour', 'one', 'large', 'hear', 'pay', 'city', 'show', 'often', 'play', 'try',
                    'government', 'go', 'among', 'system', 'run', 'part', 'man', 'come', 'turn', 'business', 'big',
                    'room', 'house', 'school', 'happen', 'call', 'world', 'water', 'month', 'continue', 'side', 'place',
                    'black', 'away', 'talk', 'hold', 'know', 'become', 'mean', 'tell', 'believe', 'ta', 'write', 'four',
                    'however', 'law', 'program', 'group', 'us', 'point', 'say', 'today', 'stand', 'study', 'feel',
                    'night', 'service', 'may', 'small', 'yes', 'begin', 'national', 'friend', 'around', 'set', 'well',
                    'leave', 'fact', 'since', 'want', 'political', 'long', 'meet', 'give', 'get', 'student', 'kind',
                    'work', 'people', 'job', 'move', 'good', 'last', 'number', 'never', 'problem', 'next', 'bad',
                    'later', 'little', 'least', 'without', 'important', 'book', 'three', 'still', 'power', 'could',
                    'much', 'country', 'car', 'game', 'every', 'end', 'keep', 'think', 'look', 'case', 'child',
                    'though', 'help', 'company', 'great', 'start', 'week', 'another', 'American', 'yet', 'might',
                    'mother', 'make', 'always', 'member', 'must', 'right', 'new', 'ask', 'would', 'father', 'see',
                    'woman', 'something', 'seem', 'live', 'include', 'hand', 'let', 'year', 'area', 'need', 'president',
                    'two', 'five', 'state', 'even', 'almost', 'really'}
    tokens = tokenizer(query)

    has_common = False
    for token in tokens:
        if token in common_words:
            # search_title = True
            has_common = True
            break

    return has_common


class BM25:
    """
    Implementation of the BM25 algorithm for ranking documents.
    """

    def __init__(self, inverted_index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.df = inverted_index.df
        self.N = len(inverted_index.DL)
        self.avg_doc_len = builtins.sum(inverted_index.DL.values()) / self.N
        self.index = inverted_index

    def calc_idf(self, query):
        """
        Calculate IDF (Inverse Document Frequency) for terms in the query.

        Parameters:
        - query (list): The list of query terms.

        Returns:
        - dict: A dictionary containing IDF values for query terms.
        """
        res = {}
        for token in query:
            if token in self.index.df:
                res[token] = math.log(self.N / (self.df[token]))
        return res

    def search(self, q, directory, N=30):
        """
        Perform a search using the BM25 algorithm.

        Parameters:
        - q (list): The list of query terms.
        - directory (str): The directory to search in.
        - N (int): The number of documents to retrieve. Default is 30.

        Returns:
        - dict: A dictionary containing document IDs and their BM25 scores.
        """
        relevant_docs, relevant_docs_dict = find_relevant_docs(self.index, directory, q)
        temp_dict = {k: self.score(q, k, relevant_docs_dict) for k in relevant_docs}

        # highest_N_score is list of tuples where (docID, score)
        highest_n_score = get_top_n(temp_dict, N)

        # Change it to a dict
        highest_n_score_dict = {doc_id: score for doc_id, score in highest_n_score}
        return highest_n_score_dict

    def score(self, query, doc_id, relevant_docs_dict):
        """
        Compute the BM25 score for a document.

        Parameters:
        - query (list): The list of query terms.
        - doc_id (int): The document ID.
        - relevant_docs_dict (dict): A dictionary containing relevant documents and their term frequencies.

        Returns:
        - float: The BM25 score for the document.
        """
        doc_len = self.index.DL[doc_id]
        query_idf_values = self.calc_idf(query)

        bm_sim_score = 0
        for term in query:
            if term in relevant_docs_dict[doc_id]:
                freq = relevant_docs_dict[doc_id][term]
                B = (1 - self.b + self.b * doc_len / self.avg_doc_len)
                idf = query_idf_values[term]
                bm_sim_score += (idf * freq * (self.k1 + 1)) / (freq + self.k1 * B)

        return float(bm_sim_score)


def match_search_results_with_title(list_of_id_and_bm_score):
    """
    Match document IDs with their titles.

    Parameters:
    - list_of_id_and_bm_score (list): A list of tuples containing document IDs and BM25 scores.

    Returns:
    - list: A list of tuples containing document IDs and their titles.
    """
    result = []
    for i in list_of_id_and_bm_score:
        result.append((str(i[0]), id_title_dict[i[0]]))
    return result


def search_backend_helper(query, inverted_index, result_dict, stem_bool=False, ngram_bool=False):
    """
    Helper function for backend search.

    Parameters:
    - query (str): The query string.
    - inverted_index: The inverted index object.
    - result_dict (dict): A dictionary to store search results.
    - stem_bool (bool): A flag indicating whether to perform stemming. Default is False.
    - ngram_bool (bool): A flag indicating whether to generate n-grams. Default is False.

    Returns:
    - dict: The updated result dictionary.
    """
    query = tokenizer(query, stem=stem_bool, ngram=ngram_bool)
    BM25_search_object = BM25(inverted_index)
    result_dict.update(BM25_search_object.search(query, ""))
    return result_dict


def search_backend(query):
    """
    Perform a backend search using the BM25 algorithm.

    Parameters:
    - query (str): The query string.

    Returns:
    - list: A list of tuples containing document IDs and their titles, sorted by relevance.
    """
    search_title_bool = is_common_word(query)
    simple_tokenized_query = tokenizer(query)

    # Create empty dictionaries to store results
    res_body_basic = {}
    res_body_stem = {}
    res_title_basic = {}
    res_title_stem = {}

    # Create thread objects for each search task
    title_basic_thread = threading.Thread(target=search_backend_helper,

                                          args=(query, inverted_title_basic, res_title_basic))
    title_stem_thread = threading.Thread(target=search_backend_helper,
                                         args=(query, inverted_title_with_stem, res_title_stem),
                                         kwargs={"stem_bool": True})
    body_basic_thread = threading.Thread(target=search_backend_helper,
                                         args=(query, inverted_body_basic, res_body_basic))
    body_stem_thread = threading.Thread(target=search_backend_helper,
                                        args=(query, inverted_body_with_stem, res_body_stem),
                                        kwargs={"stem_bool": True})

    if len(simple_tokenized_query) < 4 or search_title_bool:

        # start threads
        title_basic_thread.start()
        title_stem_thread.start()

        # join threads
        title_basic_thread.join()
        title_stem_thread.join()

        all_doc_id_set = set()
        for d in [res_title_stem, res_title_basic]:
            all_doc_id_set.update(d.keys())

        doc_final_scores = {}
        for doc in all_doc_id_set:
            doc_pr_normalized = math.log(doc_id_page_rank_dict.get(doc, 0)) / min_max_pagerank
            doc_linear_combination_score = ((res_title_stem.get(doc, 0) * 0.75) + (
                        res_title_basic.get(doc, 0) * 0.25)) * doc_pr_normalized
            doc_final_scores[doc] = doc_linear_combination_score

        final_result = match_search_results_with_title(get_top_n(doc_final_scores))

    else:
        # start threads
        body_basic_thread.start()
        body_stem_thread.start()

        # join threads
        body_basic_thread.join()
        body_stem_thread.join()

        all_doc_id_set = set()
        for d in [res_body_stem, res_body_basic]:
            all_doc_id_set.update(d.keys())

        doc_final_scores = {}
        for doc in all_doc_id_set:
            doc_pr_normalized = math.log(doc_id_page_rank_dict.get(doc, 0)) / min_max_pagerank
            doc_linear_combination_score = ((res_body_stem.get(doc, 0) * 0.75) + (
                        res_body_basic.get(doc, 0) * 0.25)) * doc_pr_normalized
            doc_final_scores[doc] = doc_linear_combination_score

        final_result = match_search_results_with_title(get_top_n(doc_final_scores))

    return final_result
