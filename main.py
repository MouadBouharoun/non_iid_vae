import tqdm
import random
import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from vae_utility import train_vae, detect_malicious_modifications, vae_threshold, detect_malicious_modifications_2
from utility import *

'''
Cette fonction charge un fichier de configuration dans le répertoire settings définie dans l'attribut "-s , ou --settings" et retourne 
les paramètres définis dans ce fichier.

Cette fonction est utilisée pour choisir le dataset volu comme shadow dataset, et main dataset
'''
def load_settings(settings_file):
    full_path = os.path.join("settings", settings_file)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"The settings file '{full_path}' does not exist.") 
    spec = importlib.util.spec_from_file_location("settings", full_path)
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    return settings_module.settings


def main(settings_file, num_clients):
    settings = load_settings(settings_file)
    
    shadow_dataset_path = settings.get("shadow_dataset")
    main_dataset_path = settings.get("main_dataset")
    
    if shadow_dataset_path and main_dataset_path:
        shadow_dataset = pd.read_csv(shadow_dataset_path)
        main_dataset = pd.read_csv(main_dataset_path)
        # Initialization
        X_train, X_test, y_train, y_test = preprocess(shadow_dataset)
        input_shape = (X_train.shape[1],)
        model = initialiseMLP(input_shape)
        inc_model = initialisePropMLP(input_shape)
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
        client_data_chunks = data_distribution(shadow_dataset, num_clients)
        clients = [Client(client_id, data_chunk) for client_id, data_chunk in enumerate(client_data_chunks)]
        global_model = initialiseMLP(input_shape, lr=0.1)
        
        ''' Imagine that the worker is aware of this specific property : {"feature": "FLOW_DURATION_MILLISECONDS", "value": 4294966, "comparison": ">="}
            The client do not want any third party to gain information whether this specific property is present or absent in its training dataset.
            Thus preserving the confidentiality of its data.
        '''
        property_config = [
            {"feature": "FLOW_DURATION_MILLISECONDS", "value": 4294966, "comparison": ">="}
        ]
        property_dataset = create_property_datasets(shadow_dataset, property_config)[0]
        data_chunks = data_distribution(property_dataset, num_clients) 
        inc_models = []
        for i, data_chunk in enumerate(tqdm.tqdm(data_chunks)):
            print("Inc Model : ", i)
            X_train_c, X_test_c, y_train_c, y_test_c = preprocess(data_chunk)
            input_shape = (X_train_c.shape[1],)
            inc_model = initialisePropMLP(input_shape, lr=0.01)
            inc_model.fit(X_train_c, y_train_c, epochs=20, batch_size=32, validation_split=0.1)
            inc_models.append(inc_model)
        inc_models = model_to_vector(inc_models)
        x_train, x_test = train_test_split(inc_models, test_size=0.5, random_state=42)
        
        
        print("# Define Hyper - parameters")
        learning_param = 0.001
        epochs = 3000
        batch_size = 32
        input_dimension = inc_models[0].shape[0]
        neural_network_dimension = 512
        latent_variable_dimension = 2
        threshold = 1.2

        print("# Train the VAE")
        total_losses, _, _, Final_Weight, Final_Bias = train_vae(x_train, epochs, batch_size, input_dimension, neural_network_dimension, latent_variable_dimension, learning_param)
        
        '''
        To evaluate the variational auto-encoding model, we use the reconstruction error. If a data point from x_test has a reconstruction error 
        greater than the predefined threshold, it is considered to be sampled from a distribution similar to that of the training data and is 
        classified as a malicious instance. In the other hand, if the reconstruction error is below the threshold, the data point is assumed to be 
        sampled from a different distribution than the training data.
        Important Note: We cannot definitively determine that the instance is not malicious because the probability distribution of the global models is unknown, 
        given our assumption that the data is not IID (independently and identically distributed).
        '''
        print("# Test the VAE")
        
        print(f"Test  {len(x_test)} instance ")     
        reconstruction_errors = detect_malicious_modifications(x_test, Final_Weight, Final_Bias)
        print(reconstruction_errors)
        malicious_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
        print("Potentially malicious modifications detected at indices:", malicious_indices)
        acc = 1 - len(malicious_indices)/len(x_test)
        print("Accuracy: {:.2f}%".format(acc * 100))
        '''
        Example usage :
        '''
        print("Let's assume this is the input from server : ")
        server_input = model
        arch_inc_model = inc_model
        inc_model = extract_inc_weights(server_input, arch_inc_model)
        inc_model = model_to_vector([inc_model])
        rec_err = detect_malicious_modifications(inc_model, Final_Weight, Final_Bias)
        print("Reconstruction Error is High !! : ", rec_err[0].numpy()) 
        
    else:
        raise ValueError("Both 'shadow_dataset' and 'main_dataset' must be specified in the settings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load settings file, number of clients, number of rounds")
    parser.add_argument("-s", "--settings", required=True, help="Name of the settings file in the settings directory (e.g., nf-unsw1_nf-unsw2.py).")
    parser.add_argument("-n", "--num_clients", type=int, required=True, help="Number of clients to use.")
    args = parser.parse_args()
    main(args.settings, args.num_clients)
