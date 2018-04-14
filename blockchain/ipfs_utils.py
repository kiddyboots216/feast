import ipfsapi
import json

from keras.models import load_model, load_weights, model_from_json


api = ipfsapi.connect('127.0.0.1', 5001)
CONFIG = None

def serialize_keras_model(destination='model_weights.h5'):
    with open(destination, 'rb') as f:
        model_bin = f.read()
        f.close()
    return model_bin

def deserialize_keras_model(content, destination='model_weights.h5'):
    with open(destination, 'wb') as g:
        g.write(content)
        g.close()
    model = load_model(destination)
    return model


def keras_to_ipfs(api, model='model_weights.h5'):
    # return api.add(serialize_keras_model(model))
    return api.add(model)

def ipfs_to_keras(api, model_addr):
    return deserialize_keras_model(CONFIG.load_weights(api.cat(model_addr)))

def get_config(api, destination='model.json'):
    with open(destination) as json_data:
       json_string = json.load(json_data)
       model = model_from_json(json_string)
    CONFIG = model
