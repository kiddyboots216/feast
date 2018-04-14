# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import nltk
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate
import keras.backend as K

# from models.generic_model import GenericModel

class ConversationalNetwork:
    def __init__(self):
        self.gen_initial_weights_path = os.path.join(
            os.path.dirname(__file__), 'pretrained/my_model_weights.h5')
        self.vocabulary_file = os.path.join(
            os.path.dirname(__file__), 'pretrained/vocabulary_movie')
        self.file_saved_context = os.path.join(
            os.path.dirname(__file__), 'pretrained/saved_context')
        self.file_saved_answer = os.path.join(
            os.path.dirname(__file__), 'pretrained/saved_answer')

        self.vocabulary = pickle.load(open(self.vocabulary_file, 'rb'))
        self.name_of_computer = 'ben'

        self.learning_rate = 0.000001
        self.maxlen_input = 50
        self.sentence_embedding_size = 300
        self.word_embedding_size = 100
        self.dictionary_size = 7000

        self.state = {
            'probability': 0.0,
            'text': ''
        }

    def build_model(self, is_retraining):
        def _build_training():
            adam = Adam(lr=self.learning_rate) 
            self.generator.compile(loss='categorical_crossentropy', optimizer=adam) 

        def _build_generator_only():
            """For testing."""
            input_context = Input(shape=(self.maxlen_input,), dtype='int32', 
                name='the-context-text')
            input_answer = Input(shape=(self.maxlen_input,), dtype='int32', 
                name='the-answer-text-up-to-the-current-token')
            LSTM_encoder = LSTM(self.sentence_embedding_size, kernel_initializer='lecun_uniform', 
                name='Encode-context')
            LSTM_decoder = LSTM(self.sentence_embedding_size, kernel_initializer='lecun_uniform', 
                name='Encode-answer-up-to-the-current-token')

            Shared_Embedding = Embedding(output_dim=self.word_embedding_size, 
                input_dim=self.dictionary_size, input_length=self.maxlen_input, name='Shared')

            word_embedding_context = Shared_Embedding(input_context)
            context_embedding = LSTM_encoder(word_embedding_context)

            word_embedding_answer = Shared_Embedding(input_answer)
            answer_embedding = LSTM_decoder(word_embedding_answer)

            merge_layer = concatenate([context_embedding, answer_embedding], axis=1, 
                name='concatenate-the-embeddings-of-the-context-and-the-answer-up-to-current-token')
            out = Dense(int(self.dictionary_size/2), activation="relu", 
                name='relu-activation')(merge_layer)
            out = Dense(self.dictionary_size, activation="softmax", 
                name='likelihood-of-the-current-token-using-softmax-activation')(out)

            model = Model(inputs=[input_context, input_answer], outputs = [out])
            self.generator = model

        # def _build_discriminator_and_generator(is_retraining):
        #     """For training."""
        #     # 1. Build generator network with initial weights.
        #     _build_generator_only(is_retraining)

        #     # 2. Build discriminator network with initial weights.
        #     # WON'T IMPLEMENT YET
        
        if is_retraining:
            # NOTE: doesn't actually build a discriminator.
            _build_generator_only()
            _build_training()
        else:
            _build_generator_only()

        self.load_initial_weights()
        return self.generator

    def run_inference(self, input_text, name):
        input_text = input_text.strip()
        preprocessed_input = self._preprocess_input(input_text, self.name_of_computer)

        if self.state["probability"] > 0.2:
            query = self.state['text'] + ' ' + preprocessed_input
        else:    
            query = preprocessed_input
        Q = self._tokenize(query)

        predout, self.state["probability"] = self._greedy_decoder(Q[0:1])
        start_index = predout.find('EOS')
        self.state['text'] = self._preprocess_input(predout[0:start_index], name) + ' EOS'

        best_answer = self.state["text"][0 : -4]
        return best_answer

    def load_initial_weights(self):
        self.load_weights(self.gen_initial_weights_path)

    def load_weights(self, new_weights_path):
        self.generator.load_weights(new_weights_path)

    def get_weights(self):
        return self.generator.get_weights()

    def sum_weights(self, weights1, weights2):
        new_weights = []
        for w1, w2 in zip(weights1, weights2):
            new_weights.append(w1 + w2)
        return new_weights

    def scale_weights(self, weights, factor):
        new_weights = []
        for w in weights:
            new_weights.append(w * factor)
        return new_weights

    def inverse_scale_weights(self, weights, factor):
        new_weights = []
        for w in weights:
            new_weights.append(w / factor)
        return new_weights

    def _preprocess_input(self, raw_word, name):   
        l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
        l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
        l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

        raw_word = raw_word.lower()
        raw_word = raw_word.replace(', ' + self.name_of_computer, '')
        raw_word = raw_word.replace(self.name_of_computer + ' ,', '')

        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term,l2[j])
            
        for term in l3:
            raw_word = raw_word.replace(term,' ')
        
        for term in l4:
            raw_word = raw_word.replace(', ' + term, ', ' + name)
            raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
            raw_word = raw_word.replace('i am ' + term, 'i am ' + self.name_of_computer)
            raw_word = raw_word.replace('my name is' + term, 'my name is ' + self.name_of_computer)
        
        for j in range(30):
            raw_word = raw_word.replace('. .', '')
            raw_word = raw_word.replace('.  .', '')
            raw_word = raw_word.replace('..', '')
           
        for j in range(5):
            raw_word = raw_word.replace('  ', ' ')
            
        if raw_word[-1] !=  '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] !=  '! ' and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
            raw_word = raw_word + ' .'
        
        if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
            raw_word = 'what ?'
        
        if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
            raw_word = 'i do not want to talk about it .'
          
        return raw_word

    def _tokenize(self, sentences):
        # Tokenizing the sentences into words:
        tokenized_sentences = nltk.word_tokenize(sentences)
        index_to_word = [x[0] for x in self.vocabulary]
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
        X = np.asarray([word_to_index[w] for w in tokenized_sentences])
        s = X.size
        Q = np.zeros((1,self.maxlen_input))
        if s < (self.maxlen_input + 1):
            Q[0,- s:] = X
        else:
            Q[0,:] = X[- self.maxlen_input:]
        return Q

    def _greedy_decoder(self, input):
        flag = 0
        prob = 1
        ans_partial = np.zeros((1,self.maxlen_input))
        ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
        for k in range(self.maxlen_input - 1):
            ye = self.generator.predict([input, ans_partial])
            yel = ye[0,:]
            p = np.max(yel)
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
            if mp == 3:  #  he index of the symbol EOS (end of sentence)
                flag = 1
            if flag == 0:    
                prob = prob * p
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (self.dictionary_size-2):
                w = self.vocabulary[k]
                text = text + w[0] + ' '
        return (text, prob)

if __name__ == '__main__':
    m = ConversationalNetwork()
    m.build_model(is_retraining=False)

    print("\n \n \n \n    CHAT:     \n \n")
    print('computer: hi ! please type your name.\n')
    name = input('user: ')
    print('computer: hi , ' + name +' ! My name is ' + m.name_of_computer + '.\n') 

    while True:
        utterance = input('user: ')
        response = m.run_inference(utterance, name)
        print('computer: ' + response + '\n\n\n')

