import numpy as np
import tensorflow as tf

from models.generic_model import GenericModel

class (GenericModel):
    def __init__(self):
        self.gen_initial_weights_path = ''
        self.dis_initial_weights_path = ''
        # self.n_input = 784
        # self.n_hidden1 = 200
        # self.n_hidden2 = 200
        # self.n_classes = 10

    def preprocess_input(self, raw_word, name):
        
        l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
        l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
        l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

        raw_word = raw_word.lower()
        raw_word = raw_word.replace(', ' + name_of_computer, '')
        raw_word = raw_word.replace(name_of_computer + ' ,', '')

        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term,l2[j])
            
        for term in l3:
            raw_word = raw_word.replace(term,' ')
        
        for term in l4:
            raw_word = raw_word.replace(', ' + term, ', ' + name)
            raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
            raw_word = raw_word.replace('i am ' + term, 'i am ' + name_of_computer)
            raw_word = raw_word.replace('my name is' + term, 'my name is ' + name_of_computer)
        
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

    def tokenize(self, sentences):

        # Tokenizing the sentences into words:
        tokenized_sentences = nltk.word_tokenize(sentences)
        index_to_word = [x[0] for x in vocabulary]
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
        X = np.asarray([word_to_index[w] for w in tokenized_sentences])
        s = X.size
        Q = np.zeros((1,maxlen_input))
        if s < (maxlen_input + 1):
            Q[0,- s:] = X
        else:
            Q[0,:] = X[- maxlen_input:]
        
        return Q

    def build_model(self, is_retraining):
        def _build_generator_only(is_retraining):
            """For testing."""
            input_context = Input(shape=(maxlen_input,), dtype='int32', 
                name='the-context-text')
            input_answer = Input(shape=(maxlen_input,), dtype='int32', 
                name='the-answer-text-up-to-the-current-token')
            LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer='lecun_uniform', 
                name='Encode-context')
            LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer='lecun_uniform', 
                name='Encode-answer-up-to-the-current-token')

            # Check this
            if is_retraining:
                Shared_Embedding = Embedding(output_dim=word_embedding_size, 
                    input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
            else:
                Shared_Embedding = Embedding(output_dim=word_embedding_size, 
                    input_dim=dictionary_size, weights=[embedding_matrix], 
                    input_length=maxlen_input, name='Shared')

            word_embedding_context = Shared_Embedding(input_context)
            context_embedding = LSTM_encoder(word_embedding_context)

            word_embedding_answer = Shared_Embedding(input_answer)
            answer_embedding = LSTM_decoder(word_embedding_answer)

            merge_layer = concatenate([context_embedding, answer_embedding], axis=1, 
                name='concatenate-the-embeddings-of-the-context-and-the-answer-up-to-current-token')
            out = Dense(int(dictionary_size/2), activation="relu", 
                name='relu-activation')(merge_layer)
            out = Dense(dictionary_size, activation="softmax", 
                name='likelihood-of-the-current-token-using-softmax-activation')(out)

            model = Model(inputs=[input_context, input_answer], outputs = [out])
            self.generator = model

            # Set up training
            ad = Adam(lr=learning_rate) 
            model.compile(loss='categorical_crossentropy', optimizer=ad)

        def _build_discriminator_and_generator(is_retraining):
            """For training."""
            # 1. Build generator network with initial weights.
            _build_generator_only(is_retraining)

            # 2. Build discriminator network with initial weights.
            




        
        if is_retraining:
            _build_discriminator_and_generator(True)
        else:
            _build_generator_only(False)

        # # load weights
        # if os.path.isfile(weights_file_discrim):
        #     model_discrim.load_weights(weights_file_discrim)

    def build_training(self):
        # 1.


    # def build_loss(self):
    #     self.loss = tf.losses.sparse_softmax_cross_entropy(
    #         labels=self.labels,
    #         logits=self.logits,
    #         loss_collection=tf.GraphKeys.LOSSES
    #     )

    # def build_optimizer(self):
    #     optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #     self.optimizer = optimizer.minimize(
    #         loss=self.loss,
    #         global_step=tf.train.get_global_step(),
    #         name='train_op'
    #     )

    def build_predictions_obj(self):
        logits = self.logits
        classes = tf.argmax(input=self.logits, axis=1, name="classes_tensor")
        probabilities = tf.nn.softmax(self.logits, name="softmax_tensor")

        tf.add_to_collection('predictions', logits)
        tf.add_to_collection('predictions', classes)
        tf.add_to_collection('predictions', probabilities)

        self.predictions = {
            "logits": logits,
            "classes": classes,
            "probabilities": probabilities
        }

    def build_eval_metric(self):
        self.eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.predictions["classes"])
        }

    def get_estimator(self, mode):
        estimator = None
        self.build_predictions_obj()
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.build_eval_metric()
            estimator = tf.estimator.EstimatorSpec(mode=mode, predictions=self.predictions)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            self.build_loss()
            self.build_optimizer()
            estimator = tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.optimizer)
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.build_loss()
            self.build_eval_metric()
            estimator = tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=self.eval_metric_ops)
        return estimator


    def get_model(self, features, labels, mode, params):
        """
        When using the Estimator API, features will come as a TF Tensor already.
        """
        # Set up hyperparameters.
        if params:
            self.learning_rate = params.get("learning_rate", None)
            self.new_weights = params.get("new_weights", None)

        # Do pre-processing if necessary.
        self.input_layer = self.preprocess_input(features["x"])
        self.labels = labels

        # Define the model.
        self.build_model(self.input_layer)

        # Build and return the estimator.
        return self.get_estimator(mode)

    def load_weights(self, new_weights, latest_checkpoint, checkpoint_dir):
        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            new_saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
            # To load non-trainable variables and prevent errors...
            # we restore them if they are found, or initialize them otherwise.
            try:
                new_saver.restore(sess, latest_checkpoint)
            except:
                sess.run(tf.global_variables_initializer())

            collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for tensor in collection:
                assign_op = tensor.assign(new_weights[tensor.name])
                sess.run(assign_op)

            save_path = new_saver.save(sess, checkpoint_dir + "model.ckpt")
        tf.reset_default_graph()

    def get_weights(self, latest_checkpoint):
        tf.reset_default_graph()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            new_saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
            new_saver.restore(sess, latest_checkpoint)
            collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            weights = {tensor.name:sess.run(tensor) for tensor in collection}
        return weights

    def get_weights_shape(self):
        tf.reset_default_graph()
        m = Perceptron()
        inputs = tf.placeholder(tf.float32, shape=(None, 28*28))
        _ = m.get_model(features={"x": inputs}, labels=None, mode='predict', params=None)
        with tf.Session().as_default() as sess:
            sess.run(tf.global_variables_initializer())
            collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            weights = {}
            for tensor in collection:
                output = sess.run(tensor)
                weights[tensor.name] = (output.shape, output.size)
        tf.reset_default_graph()
        return weights

    def sum_weights(self, weights1, weights2):
        new_weights = {}
        for key1, key2 in zip(sorted(weights1.keys()), sorted(weights2.keys())):
            assert key1 == key2, 'Error with keys'
            new_weights[key1] = weights1[key1] + weights2[key2]
        return new_weights

    def scale_weights(self, weights, factor):
        new_weights = {}
        for key, value in weights.items():
            new_weights[key] = value * factor
        return new_weights

    def inverse_scale_weights(self, weights, factor):
        new_weights = {}
        for key, value in weights.items():
            new_weights[key] = value / factor
        return new_weights
