#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ast
import re

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class NLPPreprocessing:
    def __init__(self, max_len=124, max_words=100000, embed_dim=300, embedding_index=None):
        self.max_len = max_len
        self.max_words = max_words
        self.embed_dim = embed_dim
        self.embedding_index = embedding_index

    def clean_text(self, df, remove_stopwords=True):
        temporary = df.copy()
        for idx, text in enumerate(temporary):
            #remove url
            text = re.sub(r'((www.\.[^\s]+)|(https?://[^\s]+))', '', text)

            #remove @username
            text = re.sub(r'@[^\s]+', '', text)

            #remove all the special character
            text = re.sub(r'\W', ' ', text)

            #remove single characters from the start
            text = re.sub(r'\^[A-Za-z]\s+', ' ', text)

            #remove number
            text = re.sub(r'[0-9]+', '', text)

            #removing prefixed 'b'
            text = re.sub(r'^b\s+', '', text)

            #remove all single character
            text = re.sub(r'\s+[A-Za-z]\s+', ' ', text)

            #substituting multiple spaces with single space
            text = re.sub(r'\s+', ' ', text, flags=re.I)
            
            #Remove RT
            text = re.sub(r'^RT', '', text)

            #convert to lower case
            text = text.lower()
            
            #Remove stopwords
            if remove_stopwords:
                stopword = set(stopwords.words('indonesian'))
                text = ' '.join([word for word in word_tokenize(text) if not word in stopword])

            # Subtitute previous text (raw text) into cleaned text
            temporary.iloc[idx] = text

        return temporary

    def replace_slang(self, df, tokenize=True):
        """
        Mensubtitusi kata slang menjadi kata dasar baku
        """
        with open('combined_slang_words.txt') as f:
          slang_words = f.read()
          slang_words = ast.literal_eval(slang_words)

        for idx, text in enumerate(df):
          # If true: jika input data bukan berasal dari list of words (token)
          if tokenize: 
              text = word_tokenize(text)

          # Loop list kata (tokenized words) dalam variabel teks
          new_teks = []
          for word in text:
              try:
                  # Akan mencoba memasukkan word (kata) dalam dictionary slang (kamus slang)
                  new_teks.append(slang_words[word])

              except KeyError:
                  # jika gagal, maka kata akan tetap pada dasarnya (tidak diubah) dan dimasukkan ke dalam list new_text
                  new_teks.append(word)

        df.iloc[idx] = ' '.join(new_teks)
        return df

    def stem(self, df):
        """
        Fungsi untuk stemming kata (mengembalikan ke bentuk dasar)
        """
        word_dict = {}

        # document adalah nama lain dari kumpulan sentence (1 sentence juga == 1 document)
        # Loop sentence/ document dari dataframe kolom tweet/ kolom teks
        for document in df:
          # Loop kalimat menjadi sebuah kata
          for word in document: 
            # Cek apabila kata tersebut tidak masuk dalam variabel dict word_dict
            if word not in word_dict: 
              # If true: kata tersebut akan dijadikan keys dan diisi value kosong (nantinya value kosong akan direplae dengan hasil stemming)
              word_dict[word] = ' ' 

        # Loop word dict hasil loop sebelumnya
        for word in word_dict: 
          # Setiap kata yang tersimpan dalam word_dict akan di stemming
          word_dict[word] = stemmer.stem(word) 

        # variabel x = list of words dari dataframe
        df = df.apply(lambda x: [word_dict[word] for word in x]) 
        return df
    
    def lexicon_labelling(self, series, show_weight=False):
        '''
        Tweet columns must be string (not tokenized doc)
        '''
        # Filepaths
        negative_filepath = "negative.txt"
        lexicon_filepath = 'lexicon.csv'

        # Read negative words dictionary file
        with open(negative_filepath) as f:
            negative_words = f.read().splitlines()

        # Read lexicon file
        lexicon = pd.read_csv(lexicon_filepath)

        # Preprocessing
        lexicon['weight'] = lexicon['sentiment'].map({'positive':1, 'negative':-1})
        lexicon = dict(zip(lexicon['word'], lexicon['weight']))

        # Weighting
        tweet_polarity = []
        tweet_weight = []
        negasi = False
        for sentence in series:
            sentence_score = 0
            sentiment_count = 0
            neg_word = []
            pos_word = []
            net_word = []
            sentence = sentence.split()
            for word in sentence:
                try:
                    score = lexicon[word]
                    if score > 0:
                        pos_word.append(word)
                    elif score < 0:
                        neg_word.append(word)

                    sentiment_count = sentiment_count + 1
                except:
                    score = 99

                if(score == 99):
                    if (word in negative_words):
                        negasi = True
                        sentence_score = sentence_score + 0
                    else:
                        sentence_score = sentence_score + 0


                else:
                    if(negasi == True):
                        sentence_score = sentence_score + (score * -1.0)

                        negasi = False
                    else:
                        sentence_score = sentence_score + score


            tweet_weight.append(sentence_score)

            if sentence_score > 0:
                tweet_polarity.append('positive')

            elif sentence_score < 0:
                tweet_polarity.append('negative')
            else:
                tweet_polarity.append('neutral')
        
        if show_weight:
            return (tweet_polarity, tweet_weight)
        
        else:
            return tweet_polarity

    def to_sequence(self, X_train, X_test, to_list=True):
        if to_list:
           # Convert dataframe into list
           X_train = X_train.tolist()
           X_test = X_test.tolist() 

         stop_words = set(stopwords.words('indonesian'))

         # Start Preprocessing
         print("pre-processing train data...")
         processed_docs_train = []
         for doc in tqdm(X_train):
           tokens = tokenizer.tokenize(doc)
           filtered = [word for word in tokens if word not in stop_words]
           processed_docs_train.append(" ".join(filtered))
         print("Train data has been preprocessed.")

         print("\npre-processing test data...")
         processed_docs_test = []
         for doc in tqdm(X_test):
             tokens = tokenizer.tokenize(doc)
             filtered = [word for word in tokens if word not in stop_words]
             processed_docs_test.append(" ".join(filtered))
         print("Test data has been preprocessed.")

         # To Sequence
         tokenizer = Tokenizer(num_words=self.max_words, lower=True, char_level=False)
         tokenizer.fit_on_texts(processed_docs_train)
         word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
         word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
         word_index = tokenizer.word_index
         print("dictionary size: ", len(word_index))

         #pad sequences
         word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=self.max_len)
         word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=self.max_len)

         return word_seq_train, word_seq_test, word_index

    def embedding(self, word_index):
      words_not_found = []
      vocab_size = min(self.max_words, vocab_size(word_index))
      embedding_matrix = np.zeros((vocab_size, self.embed_dim))
      for word, i in word_index.items():
          if i >= vocab_size:
              continue  
          embedding_vector = self.embedding_index.get(word)
          if (embedding_vector is not None) and len(embedding_vector) > 0:
              # words not found in embedding index will be all-zeros.
              embedding_matrix[i] = embedding_vector
          else:
              words_not_found.append(word)

      print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
      print("sample words not found: ", np.random.choice(words_not_found, 10))
      return embedding_matrix
			
class NLPVectorizer:
    def bag_of_words(self, method='tf-idf', ngram=(1, 1)):
        if method == 'tf-idf':
            vectorizer = TfidfVectorizer(ngram_range=ngram)
            
        elif method == 'count':
            vectorizer = CountVectorizer(ngram_range=ngram)
            
        return vectorizer
    
    def word_embeddings(self, data, model='word2vec'):
        if model == 'word2vec':
            pass
        
        elif model == 'glove':
            pass
        
        elif model == 'fasttext':
            pass
        
    def reduce_dimension(self, data, method='PCA'):
        if method == 'PCA':
            pass
        
        elif method == 'chi2':
            pass

'''FAST TEXT IMPORT'''

def get_average_vec(tokens_list, vector, generate_missing=False, k=300):
    """
        Calculate average embedding value of sentence from each word vector
    """
    if len(tokens_list)<1:
        return np.zeros(k)
    
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged
    
def get_embeddings(vectors, text, generate_missing=False, k=300):
    """
        create the sentence embedding
    """
    embeddings = text.apply(lambda x: get_average_vec(x, vectors, generate_missing=generate_missing, k=k))
    return list(embeddings)

def fast_text(fasttext_path):
  from gensim.models.fasttext import FastText
  fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(fasttext_path, binary=False, limit=100000)
  return fasttext_model
