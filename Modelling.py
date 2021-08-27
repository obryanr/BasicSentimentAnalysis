#!/usr/bin/env python3


'''EMBEDDING MATRIX (NLP PRETRAINED MODEL (FASTTEXT)) INTO DL MODELLING'''

# UDF
from Preprocessing import *

# Common Packages for NLP
from tqdm import tqdm
import codecs

# Deep Learning (Tensorflow)
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

# Preprocessing
from sklearn.model_selection import train_test_split

def pretrained_model(embedding_matrix, word_index, num_classes, max_len=124, embed_dim=300, max_words=100000, num_filters=64, weight_decay=1e-4):
  nb_words = min(max_words, len(word_index))
  
  model = Sequential()

  # Input
  model.add(Embedding(nb_words, embed_dim,
                      weights=[embedding_matrix], 
                      input_length=max_len, 
                      trainable=False))
  
  # Layers
  model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
  model.add(MaxPooling1D(2))
  model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

  # Flatten
  model.add(GlobalMaxPooling1D())
  model.add(Dropout(0.5))

  # Hidden Layer
  model.add(Dense(32, 
                  activation='relu', 
                  kernel_regularizer=regularizers.l2(weight_decay)))

  # Output
  model.add(Dense(num_classes, activation='sigmoid')) 

  # Optimizer
  adam = optimizers.Adam(lr=0.001, 
                         beta_1=0.9, 
                         beta_2=0.999, 
                         epsilon=1e-08, 
                         decay=0.0)
  
  model.compile(loss='binary_crossentropy', 
                optimizer=adam, metrics=['accuracy'])
  
  model.summary()

  return model
  
if __name__ == "__main__":
	# Get Fasttext model
	fasttext_path = "../cc.id.300.vec"
	embedding_index = fast_text(fasttext_path)
	
	# Read data
	df = pd.read_csv('Hate Speech.csv', usecols=['Content', 'Class'])
	
	# Preprocessing
	preprocessor = nlp_preprocessing(embedding_index)
	df['Content'] = preprocessor.clean_text(df['Content'])
	df['Content'] = preprocessor.replace_slang(df['Content'])
	df['Content'] = preprocessor.stem(df['Content'])
	
	# Split
	X_train, X_test, y_train, y_test = train_test_split(X, 
							    y, 
							    split_size=0.2, 
							    random_state=1, 
							    stratify=y)
														
	# Convert to sequence
	X_train, X_test, word_index = preprocessor.to_sequence(X_train,
							       X_test)

	# Generate embedding matrix
	embedding_matrix = preprocessor.embedding(word_index)
	
	#define callbacks
	early_stopping = EarlyStopping(monitor='val_loss', 
				       min_delta=0.01, 
				       patience=4, 
				       verbose=1)
	callbacks_list = [early_stopping]
	
	# Number of classes/ labels
	num_classes = y_train.nunique()

	# Get embedding matrix via alternative fasttext
	embeddings_word2vec = get_embeddings(word2vec_model, 
					     df["Content"],
					     k=300)

	# Init model
	model = pretrained_model(embeddings_word2vec, 
				 word_index,
				 num_classes=num_classes)
							 
	#model training
	hist = model.fit(X_train, 
			 y_train, 
			 batch_size=16, 
			 epochs=10,
			 callbacks=callbacks_list,
			 validation_split=0.1, 
			 shuffle=True, 
			 verbose=2)
