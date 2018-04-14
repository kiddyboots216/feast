import logging
import shutil
import time

import asyncio
import numpy as np
import tensorflow as tf

from models.perceptron import Perceptron
from models.cnn import CNN
from models.lstm import LSTM

from eth_utils import is_address
from solc import compile_source, compile_files
from blockchain.blockchain_utils import *
from blockchain.ipfs_utils import *


class Client(object):
    def __init__(self, iden, X_train, y_train, provider, delegatorAddress=None, clientAddress=None):
        
        self.web3 = provider
        self.api = ipfsapi.connect('127.0.0.1', 5001)

        self.PASSPHRASE = 'panda'
        self.TEST_ACCOUNT = '0xb4734dCc08241B46C0D7d22D163d065e8581503e'
        self.TEST_KEY = '146396092a127e4cf6ff3872be35d49228c7dc297cf34da5a0808f29cf307da1'

        # import_acct = self.web3.personal.importRawKey(self.TEST_KEY, self.PASSPHRASE)

        # assert(import_acct == self.TEST_ACCOUNT)

        contract_source_path_A = "blockchain/Delegator.sol"
        contract_source_path_B = "blockchain/Query.sol"
        self.compiled_sol = compile_files([contract_source_path_A, contract_source_path_B])

        self.iden = iden
        self.X_train = X_train
        self.y_train = y_train

        if clientAddress:
            assert(is_address(clientAddress))
            self.clientAddress = clientAddress
        else:
            #TODO: Initialize client 'container' address if it wasn't assigned one
            self.clientAddress = self.web3.personal.newAccount(self.PASSPHRASE)
            assert(is_address(self.clientAddress))
            self.web3.personal.unlockAccount(self.clientAddress, self.PASSPHRASE)
            # send_raw_tx(self.web3, self.clientAddress, self.TEST_ACCOUNT, self.TEST_KEY)
            # self.web3.eth.sendTransaction({"from": self.TEST_ACCOUNT, "to": self.clientAddress,
                                           # "value": 9999999999})
        print("Client Address:", self.clientAddress)

        self.Delegator_address = delegatorAddress

    def get_money(self):
        get_testnet_eth(self.clientAddress, self.web3)
        print(self.web3.eth.getBalance(self.clientAddress))

        if self.Delegator_address:
            assert(is_address(self.Delegator_address))
        else:
            #TODO: Figure out what to do in event that a master address is not supplied
            Query_id, self.Query_interface = self.compiled_sol.popitem()
            Delegator_id, self.Delegator_interface = self.compiled_sol.popitem()

            # self.Query_address = deploy_Query(self.web3, self.Query_interface, self.TEST_ACCOUNT, addr_lst)
            self.Delegator_address = deploy_Master(self.web3, self.Delegator_interface, self.clientAddress)

    def launch_query(self, target_address):
        contract_obj = self.web3.eth.contract(
           address=self.Delegator_address,
           abi=self.Delegator_interface['abi'])
        tx_hash = contract_obj.functions.query(target_address).transact(
            {'from': self.clientAddress})
        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        print(tx_receipt)
        self.Query_address = self.web3.toChecksumAddress('0x' + tx_receipt['logs'][0]['data'].split('000000000000000000000000')[2])
        # return tx_receipt

    def ping_client(self):
        contract_obj = self.web3.eth.contract(
           address=self.Query_address,
           abi=self.Query_interface['abi'])
        tx_hash = contract_obj.functions.pingClients().transact({'from': self.clientAddress})
        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        return tx_receipt

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
        self.weights_metadata = self.model.get_weights_shape()

    def train(self, config, weights=None):
        time.sleep(10)
        return
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

    def handle_ClientSelected_event(self, event):
        contract_obj = self.web3.eth.contract(
           address=self.Query_address,
           abi=self.Query_interface['abi'])

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
        update = train(config)

        # IPFS add
        IPFSaddress = 'QmVm4yB2jxPwXXVXM6n86TuwA4jCQ7EfNPjguFrhoCbPiJ'

        tx_hash = contract_obj.functions.receiveResponse(IPFSaddress).transact(
            {'from': clientAddress})
        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
        return log[0]

    def handle_QueryCreated_event(self, event_data):
        print(event_data)
        address = event_data.split("000000000000000000000000")[2]
        assert(is_address(address))
        self.Query_address = address
        return event_data.split("000000000000000000000000")

    def handle_BeginAveraging_event(IPFSaddress):
        # get info at address
        stuff = self.api.cat(IPFSaddress)
        tx_hash = contract_obj.functions.allDone().transact(
            {'from': clientAddress})

    async def start_listening(self, event_to_listen, poll_interval=5):
        while True:
            lst = event_to_listen.get_new_entries()
            if lst:
                print(lst[0])
                return lst[0]
            await asyncio.sleep(poll_interval)

    def filter_set(self, event_sig, contract_addr, handler):
        event_signature_hash = self.web3.sha3(text=event_sig).hex()
        event_filter = self.web3.eth.filter({
            "address": contract_addr.lower(),
            "topics": [event_signature_hash]
            })
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        try:
            inter = loop.run_until_complete(
                self.start_listening(event_filter, 2))
            check = handler(inter['data'])
        finally:
            loop.close()
        return check

    def main(self):
        check = self.filter_set("QueryCreated(address,address)", self.Delegator_address, self.handle_QueryCreated_event)
        if check[0] + check[1] == self.clientAddress.lower():
            target_contract = check[0] + check[2]
            print(target_contract)
            retval = self.filter_set("ClientSelected(address)", target_contract, self.handle_ClientSelected_event)
            # return "I got chosen:", retval[0] + retval[1]
            alldone = self.filter_set("BeginAveraging(string)", target_contract, self.handle_BeginAveraging_event)

        else:
            return "not me"









    def flatten_weights(self, weights, factor=1e10):
        flattened = []
        for _, tensor in sorted(weights.items()):
            flattened.extend(tensor.flatten().tolist())
        return [int(n*factor) for n in flattened]

    def unflatten_weights(self, flattened, factor=1e10):
        flattened = [n/factor for n in flattened]
        weights = {}
        index = 0
        for name, (shape, size) in sorted(self.weights_metadata.items()):
            weights[name] = np.array(flattened[index:index+size]).reshape(shape)
            index += size
        return weights

    def get_checkpoints_folder(self):
        return "./checkpoints-{0}/{1}/".format(self.iden, self.model_type)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.get_checkpoints_folder())
