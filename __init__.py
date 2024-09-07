import tensorflow as tf


model_id = 'mlp'


# FedSGD

learning_rate_fedSGD = 0.001

opt = tf.keras.optimizers.Adam()

#VAE

learning_param_VAE = 0.001
epochs = 3000
batch_size = 32
neural_network_dimension = 512
latent_variable_dimension = 2
threshold = 1.2
