# save_model_structure.py

import tensorflow as tf
import argparse
import tensorflow.python.tools.inspect_checkpoint as chkp

def load_model_and_print_structure(checkpoint_path):
    # Load the model from the checkpoint
    model = tf.keras.models.load_model(checkpoint_path)
    
    # Print the model summary
    model.summary()

    # Print detailed layer information
    for layer in model.layers:
        print(f"Layer: {layer.name}")
        print(f"  Input shape: {layer.input_shape}")
        print(f"  Output shape: {layer.output_shape}")
        print(f"  Weights: {layer.weights}")

def load_inspect_checkpoint(checkpoint_path):
    chkp.print_tensors_in_checkpoint_file(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print model structure from a checkpoint file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    args = parser.parse_args()

    load_model_and_print_structure(args.checkpoint)
    load_inspect_checkpoint(args.checkpoint)
