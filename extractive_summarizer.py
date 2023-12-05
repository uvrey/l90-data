import tqdm
from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import re
from sentence_transformers import SentenceTransformer, util

"""
Constructs Extractive Summarizer
"""
class ExtractiveSummarizer:
    def __init__(self):
        '''
        Sets data structures.
        '''
        self.corpus_raw_data = {}
        self.corpus_data = {}
        self.class_labels = {}
        self.predict_features = {}
        self.train_features = {}
        self.optimal_weights = []
        self.all_article_freqs = {}
        self.article_tfidf_scores = {}
        self.compression_ratios = {}
        self.theta = 0.52

    def reset_if_predicting(self):
        '''
        Resets data structures if we are now predicting from our weight vector. 
        '''
        self.corpus_raw_data = {}
        self.corpus_data = {}
        self.class_labels = {}
        self.predict_features = {}
        self.all_article_freqs = {}
        self.article_tfidf_scores = {}
        self.compression_ratios = {}

    def preprocess(self, articles, ids):
        '''
        Params
        articles: list of articles 
        training: optional parameter to restrict number of training articles

        Returns
        list. Split articles. 
        '''
        # create dictionary of corpus data
        self.corpus_raw_data = dict(zip(ids, articles))
        id_keys = list(self.corpus_raw_data.keys())

         # split articles 
        split_articles = [[s.strip() for s in article.split('.')] for article in articles]  
        compression_ratios = []

        for i, article in tqdm.tqdm(enumerate(split_articles), total=len(split_articles), desc="Pre-processing data"):
            article_word_freq = Counter()
            # print(f'handling {len(article)} sentences for article {i}')
            for sentence in article:
                words = word_tokenize(sentence)

                # remove punctuation
                cleaned_words = [re.sub(re.compile(r'[^\w\s]'), '', word.lower()) for word in words]

                # prepare and remove stop words
                stop_words = set(stopwords.words('english'))
                cleaned_words = [word for word in cleaned_words if word.lower() not in stop_words and word != "" and not re.match(r'^[0-9.]+$', word)]

                # measure ratio of cleaned / original text per sentence. closer to 1 = more concise
                if len(cleaned_words) == len(words):
                    compression_ratios.append(1)
                else:
                    compression_ratios.append(len(cleaned_words) / len(words))

                # update word freqs for this particular article
                article_word_freq.update(set(cleaned_words))
            
            # append all frequencies to the list
            self.all_article_freqs[id_keys[i]] = article_word_freq
            
            # create tf- idf scores
            self.article_tfidf_scores[id_keys[i]] = (TFIDFScore(article_word_freq, article))

            # store all ratios for the sentences in each article 
            self.compression_ratios[id_keys[i]] = compression_ratios

            # reset the ratio
            compression_ratios = []

        return split_articles

    def train(self, X, y):
        '''
        Params.
        X: list. List of sentences (i.e., comprising an article)
        y: list. Yes/no decision for each sentence (as boolean)
        '''
        # get raw data into usable form

        counter = 0
        for corpus, decisions in tqdm.tqdm(zip(X, y), desc="Training model", total = len(X)):
             # validate data shape 
            assert len(corpus) == len(decisions), "Article and decisions must have the same length"
            # store list of documents for each corpus in the class
            self.corpus_data[list(self.corpus_raw_data.keys())[counter]] = corpus
            counter += 1

        # create class labels from list of data labels
        self.class_labels = dict(zip(self.corpus_data.keys(), y))

        # construct massive feature matrix for all articles
        first_batch = True
        for id in tqdm.tqdm(list(self.corpus_data.keys()), total = len(self.corpus_data.keys()), desc = "Training model"): 
            feature_extractor = FeatureExtractor(self.corpus_data[id], self.article_tfidf_scores[id], self.compression_ratios[id], self.all_article_freqs[id])
            matrix = feature_extractor.calculate_matrix()
            if first_batch:
                huge_matrix = matrix
                first_batch = False
            else:
                huge_matrix = np.vstack((huge_matrix, matrix))

        # prepare class labels for whole dataset
        input_class_labels = []
        for cl in list(self.class_labels.values()):
            for c in cl:
                input_class_labels.append(c)
        
        # train model
        model = LogisticRegression(huge_matrix, input_class_labels, 0.01, 100)
        self.optimal_weights = model.optimize()
        first_batch = False
        
    def predict(self, X):
        '''
        Params
        X: list of list of sentences (i.e., comprising an article)
        '''
        id_keys = list(self.corpus_raw_data.keys())

        counter = 0
        for article in X:
            self.corpus_data[list(self.corpus_raw_data.keys())[counter]] = article
            counter += 1

        # construct feature matrix 
        first_batch = True
        for i, article in tqdm.tqdm(enumerate(X), total = len(X), desc= "Preparing predictions"): 
            feature_extractor = FeatureExtractor(self.corpus_data[id_keys[i]], self.article_tfidf_scores[id_keys[i]], self.compression_ratios[id_keys[i]], self.all_article_freqs[id_keys[i]])
            matrix = feature_extractor.calculate_matrix()
            matrix = feature_extractor.calculate_matrix()
            if first_batch:
                huge_matrix = matrix
                first_batch = False
            else:
                huge_matrix = np.vstack((huge_matrix, matrix))

        # print("we want to predict for...")
        # print(huge_matrix.shape)
        # print("given weight matrix of size:")
        # print(self.optimal_weights)
            
        predictions = sigmoid(np.dot(self.optimal_weights, huge_matrix.T))
        y_star = []
        for p in predictions:
            if p < self.theta:
                y_star.append(0)
            else:
                y_star.append(1)
        
        prediction_index_start = 0
        prediction_index_finish = 0
        article_count = 0

        # normalize decisions to probability between 0 and 1
        for article in X:
            article_count += 1
            print(f'building summary {article_count}')
            prediction_index_finish = prediction_index_start + len(article) 
            print(f'p_start = {prediction_index_start}, p_finish = {prediction_index_finish}')
            print(f'first set of decisions to be sent: {len(y_star[prediction_index_start: prediction_index_finish])}')
            yield self.build_summary(article, y_star[prediction_index_start: prediction_index_finish])
            prediction_index_start = prediction_index_finish 

    def build_summary(self, article, y_star):
        '''
        Params
        article: list. Sentences within the article.
        y_star: list. Predictions, informed by the weight vector.

        Returns
        str. Summary of article.
        '''

        summary = []
        for i in range(len(y_star)):
            if y_star[i] == 1:
                summary.append(article[i])
        return ' '.join(summary)

class FeatureExtractor:
    def __init__(self, corpus, tfidf, comp_ratios, freqs):
        '''
        Params
        corpus: list. List of sentences within the article.
        tfidf: TFIDFScore object. 
        comp_ratios: list. Compression ratios of each article.
        freqs: dict. Word frequencies. 
        '''
        self.corpus = corpus
        self.feature_matrix_list = {}
        self.tfifd = tfidf
        self.comp_ratios = comp_ratios
        self.freqs = freqs

    def mean_normalize(self, X):
        means = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        return (X - means) / std_dev
    
    def normalize(self, X):
        if X.sum(axis=0) != 0:
            return X / X.sum(axis=0)
        else:
            return X

    def calculate_matrix(self):
        '''
        Returns
        np matrix. The feature matrix for a given sentence. 
        '''
        # calculate score based on the word count of a sentence
        word_counts = self.normalize(np.array([word_count(c) for c in self.corpus]))

        # calculate score based on sentence position in the document
        doc_positions = self.normalize(np.array([doc_position(i, len(self.corpus)) for i in range(len(self.corpus))]))

        # calculate number of referred-to numbers in an article
        number_counts = self.normalize(np.array([number_count(c) for c in self.corpus]))

        # calculate TF-IDF scores per sentence
        tfidf_scores = self.normalize(np.array(self.tfifd.calculate_score()))

        # calculate average sentence embeddings using BERT
        sentence_bert_embeddings = self.normalize(np.array(get_sentence_bert_embeddings(self.corpus)))

        # calculate keyword scores
        keyword_scores = self.normalize(np.array(get_keyword_scores(self.freqs, self.corpus)))

        # get compression ratios 
        compression_ratios = self.normalize(np.array(self.comp_ratios))

        # combine feature vectors to form the feature matrix
        # tfidf_scores, sentence_bert_embeddings, compression_ratios
        return np.column_stack((word_counts, doc_positions, keyword_scores, tfidf_scores))

"""
Functions for Feature Extraction
"""
def number_count(s):
    '''
    Params
    doc: str. Sentence string.

    Returns
    int. The number of numbers in the sentence string. 
    '''
    initial = 0.01
    return len(re.findall(r'\d+', s)) + initial

def word_count(s):
    '''
    Params
    S: str. Sentence string.

    Returns
    int. The number of words in the sentence string. 
    '''
    return len(s.split())

def doc_position(index, doc_length):
    scaled_x = (index/ doc_length) * 2
    return 0.1*(scaled_x-1)**2

def get_keyword_scores(word_freqs, S):
    '''
    Params
    S: list. List of sentences.
    freq: dict. Contains frequencies of words.

    Returns
    list. The average word embeddings of each sentence using BERT. 
    '''
    # get top 5 keywords and weight according to their frequency
    top_keywords = {word: freq for word, freq in list(word_freqs.items())[:5]}
    keyword_scores = []
    keyword_score = 0

    # assign weighting for number of keywords in a sentence
    for s in S:
        for k in top_keywords:
            if k in s:
                keyword_score += top_keywords[k]
        keyword_scores.append(keyword_score)
        keyword_score = 0
    return keyword_scores

def get_sentence_bert_embeddings(S):
    '''
    Params
    S: list. List of sentences.

    Returns
    list. The average word embeddings of each sentence using BERT. 
    '''
    # prepare model for sentences
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(S)
    ave_bert_embeds = []

    # print the embeddings
    sum_embeds = 0
    for sentence_embed in embeddings:
        for word_embed in sentence_embed:
            sum_embeds += word_embed
        ave_bert_embeds.append(sum_embeds/len(sentence_embed))
        sum_embeds = 0
    return ave_bert_embeds

class TFIDFScore:
    def __init__(self, word_freq, sentences):
        """
        Params
        freq: dict. Contains frequencies of words.
        sentences: list. Sentences within a given article.         
        """
        self.freq = word_freq
        self.sentences = sentences
        self.n = len(self.sentences)

    def calculate_score(self):
        """
        Returns
        list. TF-IDF scores associated with a given set of sentences. 
        """
        # collect TF dictionary
        tf_idf_dict = self.get_tfidf_dict()
        score = 0
        scores = []
        for s in self.sentences:
           # print(f'looking at sentence: {s}')
            words = s.split()
            for word in words:
                # print(f'with words: {word}')
                try:
                    score += tf_idf_dict[word.strip()]
                except KeyError:
                    pass
            scores.append(score)
        return scores
            
    def get_tfidf_dict(self):
        '''
        Calculates a dictionary of TF-IDF values for a set of words. 
        
        Returns
        dict. Dictionary of TF-IDF word embedding values, with words as keys. 
        '''
        total_words = len(self.freq.items())
        tf_dict = {}
        for w in self.freq.keys():
            tf_dict[w] = self.freq[w] / total_words
        
        # collect IDF 
        idf_dict = {word: np.log(self.n / (df + 1)) for word, df in self.freq.items()}
        tf_idf_dict = {word: tf_dict[word]* idf_dict[word] for word in tf_dict.keys()}
        return tf_idf_dict

class LogisticRegression:
    def __init__(self, X, y, lambda_, iterations):
        '''
        Initialises the LogisticRegression class.

        Params
        X: (N, d) array. Input data as N feature vectors of dimension d.
        w: (d, ) array. Weight vector.
        lambda_: Regularization term (scalar)
        iterations: integer. Number of iterations for the Newton-Raphson optimizer.
        '''
        self.X = X
        self.y = np.array(y)
        self.lambda_ = lambda_
        self.iterations = iterations
        self.theta = 0.52

    def display_model_summary(self):
        '''
        Displays a visual representation of the model summary.
        '''
        print("----------------")
        print(f'number of features: {self.X[0].shape}')
        print(f'y vector shape: {self.y.shape}')
        print(f'X matrix shape: {self.X.shape}')
        print(f'theta & lambda: {self.theta}, {self.lambda_}')
        print("----------------")
    
    def optimize(self):
        '''
        Returns
        list. Set of optimized weights following the optimization process.
        '''
        weights = np.random.rand(self.X.shape[1])
        return solve_Newton_Raphson(self.X, self.y, weights, self.lambda_, self.iterations)
    
"""
Helper Functions for Logistic Regression
"""
def sigmoid(x):
    '''
    Params
    x: integer. Input to the sigmoid function.

    Returns
    double. The sigmoid function on x.
    '''
    return 1/(1 + np.exp(-x))

def hessian(X, w, lambda_):
    '''
    Params
    X: (N, d) array. Input data as N feature vectors of dimension d.
    w: (d, ) array. Weight vector.
    lambda_: Regularization term (scalar)
    
    Returns
    H: (d, d) array. Hessian matrix of loss function J(w) in terms of w, X and lambda_.
    '''
    
    N = X.shape[0]
    d = X.shape[1]
    
    # H is dxd matrix
    H = np.zeros((d, d))
    
    for n in range(0, N):
        x_n = X[n, :]
        sig_param = np.dot(w.T, x_n)
        x_n2 = np.dot(x_n, x_n.T)
        H += sigmoid(sig_param) * (1-sigmoid(sig_param)) * x_n2
        H += 1/lambda_ * np.eye(d) # add regularization term
        
    return H

def calc_grad_vector(X, y, w, lambda_):
    '''
    Params
    X: (N, d) array. Input data as N feature vectors of dimension d.
    y: (N, ) array. Class labels.
    w: (d, ) array. Weight vector.
    lambda_: Regularization term (scalar)
    
    Returns
    delta_L: (d, ) array. Gradient vector for use in multi-dimensional
    Newton Raphson. Use to find w* that minimises the NLL.
    '''
    N = X.shape[0]
    d = X.shape[1]
    
    delta_l = np.zeros((d))
    
    for n in range(0, N):
        x_n = X[n, :]
        delta_l -= np.dot((y[n] - sigmoid(np.dot(w.T, x_n))), x_n)
        
    delta_l += 1/lambda_*w # add regularization term
    
    return delta_l

def solve_Newton_Raphson(X, y, w_new, lambda_, num_iter=10):
    '''
    Params
    X: (N, d) array. Input data as N feature vectors of dimension d.
    y: (N, ) array. Class labels.
    lambda_: Regularization term (scalar)
    num_iter: int. Maximum number of iterations to perform.
    
    Returns
    w_star: (d, ) array. Optimised weights w* that minimises the loss function J(w).
    '''
    print(X.shape)
    print(y.shape)
    print("--------")

    ll_old = 0
    ll = 0
    
    for i in range(0, num_iter):
        ll = calc_loss(X, y, w_new, lambda_)
        
        # debug messages
        if i == 0:
            print("Iteration " + str(i+1) + ". Loss: J(w)= " + str(np.round(ll, 2)))
        else:
            percent_change = np.round(100*(ll-ll_old)/ll_old, 1)
            print("Iteration " + str(i+1) + ". Loss: J(w)= " + str(np.round(ll, 2)) + " (" + str(percent_change) + "%)")
        
        ll_old = ll
                  
        # Calculate Hessian matrix
        H = hessian(X, w_new, lambda_)
        
        # Calculate gradient vector
        delta_l = calc_grad_vector(X, y, w_new, lambda_)
        
        w_new -= 2*np.dot(np.linalg.inv(H), delta_l)
        
    # print("Training complete!")
    return w_new

def calc_loss(X, y, w, lambda_):
    '''
    Params
    X: (N, d) array. Input data as N feature vectors of dimension d.
    y: (N, ) array. Class labels.
    w: (d, ) array. Weight vector.
    lambda_: Regularization param (scalar)
    
    Returns
    J: double. The loss (NLL + regularisation term) for y given X and w.
    '''
    N = X.shape[0]
    
    # compute the NLL
    NLL = 0
    for n in range(N):
        x_n = X[n, :]
        ss = sigmoid(np.dot(w.T, x_n))
        NLL -= y[n] * np.log(ss) + (1 - y[n]) * np.log(1 - ss)
    
    # add the regularisation term
    return NLL + np.sum(np.square(w))/(2*lambda_)