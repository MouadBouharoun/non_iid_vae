import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import clone_model


'''
This function converts non numerical features such as IPv4 Addresses into a numeric format.
'''
def preprocess(df):
    src_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_SRC_ADDR"].unique()))}
    dst_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_DST_ADDR"].unique()))}
    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda name: src_ipv4_idx[name])
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda name: dst_ipv4_idx[name])
    df = df.drop('Attack', axis=1)
    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    X = (X - X.min()) / (X.max() - X.min())
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)


'''
This function initializes the models.
The first model is used in federated learning.
The second model is designed to perform the inconsistency attack (canary gradient attack).
'''
def initialiseMLP(input_shape, lr=0.01):
    model = Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return model

def initialisePropMLP(input_shape, lr=0.01):
    model = Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(2, activation='relu')
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

'''
This function distributes the data uniformly among the participants.
'''
def data_distribution(data, num_clients):
    data_shuffled = data.sample(frac=1, random_state=42)
    client_data_chunks = []
    start_idx = 0
    chunk_size = len(data_shuffled) // num_clients
    for i in range(num_clients):
        end_idx = start_idx + chunk_size
        client_data_chunks.append(data_shuffled.iloc[start_idx:end_idx])
        start_idx = end_idx
    return client_data_chunks

'''
This function computes a gradient based on the global model and the local data.
'''
def client_update(local_model, data):
    with tf.GradientTape() as tape:
        predictions = local_model(data['X_train'])
        loss = tf.keras.losses.sparse_categorical_crossentropy(data['y_train'], predictions)
    gradients = tape.gradient(loss, local_model.trainable_variables)
    return gradients

'''
This function aggregates incoming gradients
'''
def server_aggregate(global_model, client_updates):
    averaged_gradients = [tf.zeros_like(var) for var in global_model.trainable_variables]
    num_clients = len(client_updates)
    for client_update in client_updates:
        for i, grad in enumerate(client_update):
            averaged_gradients[i] += grad / num_clients
    return averaged_gradients

'''
Function to convert a model into a 1D numpy ndarray vector.
Useful as a processing step for training the VAE.
'''
def model_to_vector(models):
    new_models = []
    for model in models:
        weights_list = []
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                weights_list.append(w.flatten())  
        model_vector = np.concatenate(weights_list)
        new_models.append(model_vector)
    return np.array(new_models).astype(np.float32)

'''
This function inserts the weights of the inconsistent model into the weights of the global model.
'''
def insert_weights_MI_to_GM(model_inconsistency, global_model):
    model_inconsistency_weights = model_inconsistency.get_weights()
    partial_model_weights = model_inconsistency_weights[2:4]
    current_weights = global_model.layers[1].get_weights()
    updated_weights = current_weights.copy()
    updated_weights[0][:, :2] = partial_model_weights[0]  # Update weights
    updated_weights[1][:2] = partial_model_weights[1]  # Update biases
    global_model.layers[1].set_weights(updated_weights)
    global_model.layers[0].set_weights(model_inconsistency_weights[0:2])
    return global_model
'''
This function extracts the weights of the partially inconsistent model from the weights of the global model.
'''
def extract_inc_weights(global_model, arch_inc_model):
    global_weights_layer_0 = global_model.layers[0].get_weights()
    global_weights_layer_1 = global_model.layers[1].get_weights()

    ''' 
    Extract the specific parts corresponding to model_inconsistency.   
    '''
    extracted_weights_0 = global_weights_layer_0  #Les deux premières couches
    extracted_weights_1 = [
        global_weights_layer_1[0][:, :2],  
        global_weights_layer_1[1][:2]      
    ]
    '''
    Merge the extracted weights.
    '''
    inc_weights = extracted_weights_0 + extracted_weights_1
    '''# Cloner la structure du modèle template pour éviter de modifier l'original'''
    inc_model = clone_model(arch_inc_model)
    '''# Charger les poids extraits dans le nouveau modèle'''
    inc_model.set_weights(inc_weights)
    return inc_model
'''
This function prepares a sequence of ground truth values consisting of 1s and 0s,
where half of the sequence consists of 1s and the other half of 0s
'''
def ground_truth(test_size):
    if test_size % 2 != 0:
        raise ValueError("test_size must be congruent to 2.")
    
    num_ones = test_size // 2
    num_zeros = test_size - num_ones

    sequence = [1] * num_ones + [0] * num_zeros
    
    return sequence


"""This function creates multiple property datasets from the given configurations."""
def create_property_datasets(main_dataset, properties_config):
    property_datasets = []
    for i, config in enumerate(properties_config, start=1):
        print(f"Création de property_dataset_{i} pour '{config['feature']}' avec seuil {config['value']}")
        property_dataset = main_dataset.copy()
        property_dataset = property_dataset.drop('Label', axis=1)
        if config['comparison'] == ">=":
            property_dataset['Label'] = (property_dataset[config['feature']] >= config['value']).astype(int)
        elif config['comparison'] == "<=":
            property_dataset['Label'] = (property_dataset[config['feature']] <= config['value']).astype(int)
        else:
            raise ValueError("Opérateur de comparaison non supporté")
        property_datasets.append(property_dataset)
    return property_datasets

"""
Clients are treated as class objects; each is identified by a numerical ID, their local data, and their local model.
"""
class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.local_data = data
        self.local_model = None

    def set_global_model(self, global_model):
        self.local_model = tf.keras.models.clone_model(global_model)
        self.local_model.compile(optimizer=global_model.optimizer,
                                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                                 metrics=['accuracy'])

    def get_local_model(self):
        return self.local_model

    def get_data(self):
        X_train_c, X_test_c, y_train_c, y_test = preprocess(self.local_data)
        return {'X_train': X_train_c,
                'y_train': y_train_c}
