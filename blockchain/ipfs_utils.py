import keras
import ipfsapi


api = ipfsapi.connect('127.0.0.1', 5001)

def serialize_keras_model(destination='model_weights.h5'):
    model.save(destination)
    with open(destination, 'rb') as f:
        model_bin = f.read()
        f.close()
return model_bin

def deserialize_keras_model(content, destination='model_weights.h5'):
    with open(destination, 'wb') as g:
        g.write(content)
        g.close()
    model = keras.models.load_model(destination)
return model

def keras_to_ipfs(api, model):
    return api.add(serialize_keras_model(model))


def ipfs_to_keras(api, model_addr):
    return deserialize_keras_model(api.cat(model_addr))
