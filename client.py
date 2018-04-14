import logging
import shutil
import time

import numpy as np
import tensorflow as tf

from models.perceptron import Perceptron
from models.cnn import CNN
from models.lstm import LSTM

from eth_utils import is_address
from blockchain.blockchain_utils import *


class Client(object):    
    def __init__(self, iden, X_train, y_train, provider, masterAddress=None, clientAddress=None):
        
        self.PASSPHRASE = 'panda'
        self.TEST_ACCOUNT = '0xf6419f5c5295a70C702aC21aF0f64Be07B59F3c4'
        self.TEST_KEY = '146396092a127e4cf6ff3872be35d49228c7dc297cf34da5a0808f29cf307da1'

        self.iden = iden
        self.X_train = X_train
        self.y_train = y_train
        self.web3 = provider
        if masterAddress:
            assert(is_address(masterAddress))
            self.masterContract = self.web3.eth.contract(
                address=masterAddress,
                abi=contractAbi)
        else:
            #TODO: Figure out what to do in event that a master address is not supplied

            self.masterContract = None
        if clientAddress:
            assert(is_address(clientAddress))
            self.clientAddress = clientAddress
        else:
            #TODO: Initialize client 'container' address if it wasn't assigned one
            PASSPHRASE = 'panda'
            self.clientAddress = self.web3.personal.newAccount(self.PASSPHRASE)
            assert(is_address(self.clientAddress))
            # send_raw_tx(self.web3, self.clientAddress, self.TEST_ACCOUNT, self.TEST_KEY)
            # self.web3.eth.sendTransaction({"from": self.TEST_ACCOUNT, "to": self.clientAddress,
                                           # "value": 9999999999})
        get_testnet_eth(self.clientAddress)
        self.buyerContract = None          

    def setup_model(self, model_type):
        self.model_type = model_type
        if model_type == "perceptron":
            self.model = Perceptron()
        elif model_type == "cnn":
            #TODO: Support CNN
            self.model = CNN()
        elif model_type == "lstm":
            #TODO: Support LSTM
            self.model = LSTM()
        else:
            raise ValueError("Model {0} not supported.".format(model_type))

    # def setup_training(self, batch_size, epochs, learning_rate):
    #     self.batch_size = self.X_train.shape[0] if batch_size == -1 else batch_size
    #     self.epochs = epochs
    #     self.params = {'learning_rate': learning_rate}

    def train(self, weights, config):
        #TODO: Make train() only need to take in the config argument ONCE
        logging.info('Training just started.')
        assert weights != None, 'weights must not be None.'
        batch_size = self.X_train.shape[0] if config["batch_size"] == -1 \
            else config["batch_size"]
        epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        params = {'new_weights': weights, 'learning_rate': learning_rate}

        classifier = tf.estimator.Estimator(
            model_fn=self.model.get_model,
            model_dir=self.get_checkpoints_folder(),
            params = params
        )
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(
        #     tensors=tensors_to_log, every_n_iter=50)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.X_train},
            y=self.y_train,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True
        )
        classifier.train(
            input_fn=train_input_fn,
            # steps=1
            # hooks=[logging_hook]
        )
        logging.info('Training complete.')
        new_weights = self.model.get_weights(self.get_latest_checkpoint())
        shutil.rmtree("./checkpoints-{0}/".format(self.iden))
        update, num_data = new_weights, self.X_train[0].size
        update = self.model.scale_weights(update, num_data)
        return update, num_data

    # def send_weights(self, train_arr, train_key):
    #     #this should call the contract.sendResponse() with the first argument train() as the input
    #     tx_hash = contract_obj.functions.sendResponse(train_arr, train_key, len(train_arr)).transact(
    #         {'from': clientAddress})
    #     tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
    #     log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
    #     return log[0]

    def handle_event(self, event):
        print(event)

    def handle_ClientSelected_event(self, event):
        #TODO: get weights and config from event
        #weights will come from the smart contract's variable currentWeights[]
        weights = event.get_weights()
        #will be hardcoded
        config = {
            "num_clients": 1,
            "model_type": 'perceptron',
            "dataset_type": 'iid',
            "fraction": 1.0,
            "max_rounds": 100000,
            "batch_size": 50,
            "epochs": 10,
            "learning_rate": 1e-4,
            "save_dir": './results/',
            "goal_accuracy": 1.0,
            "lr_decay": 0.99
        }
        update, num_data = train(weights, config)
        tx_hash = contract_obj.functions.receiveResponse(update, num_data).transact(
            {'from': clientAddress})
        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
        return log[0]

    def handle_QueryCreated_event(self, event):
        #TODO: get address of the buyer from the master contract
        address = event.get_address()
        assert(is_address(address))
        self.buyerAddress = adress
        start_listening(buyerAddress = self.buyerAddress)

    def start_listening(self, buyerAddress = None, 
        # event_to_listen = None, 
        poll_interval = 1000):
        #this should set this client to start listening to a specific contract
        #make this non-blocking
        #TODO: Make event filtering work!
        if buyerAddress:
            assert(is_address(buyerAddress))
            self.buyerContract = self.web3.eth.contract(
                    address=buyerAddress,
                    abi=contractAbi)
            event_filter = self.buyerContract.eventFilter('ClientSelected', {'fromBlock': 'latest'})
            while True:
                for event in event_filter.get_new_entries():
                    handle_clientSelected_event(event)
                time.sleep(poll_interval)
        else:
            event_filter = self.masterContract.eventFilter('QueryCreated', {'fromBlock': 'latest'})
            while True:
                for event in event_filter.get_new_entries():
                    handle_queryCreated_event(event)
                time.sleep(poll_interval)

    def get_checkpoints_folder(self):
        return "./checkpoints-{0}/{1}/".format(self.iden, self.model_type)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.get_checkpoints_folder())
