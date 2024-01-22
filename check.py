import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import backend as K

# Load the model
model = load_model('./param/lstm_w12.keras')

# Create a function to get the output of each layer
layer_outputs = [layer.output for layer in model.layers]
get_layer_output = K.function([model.input], layer_outputs)

# Generate a specific test input
test_input = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]])  # Adjust the input array as needed

# Get the output of each layer for the specific input
layer_outputs_values = get_layer_output([test_input])

# Print the output of each layer
for layer_output_value, layer in zip(layer_outputs_values, model.layers):
    print(f"Layer '{layer.name}' output shape: {layer_output_value.shape}")
    print(layer_output_value)
