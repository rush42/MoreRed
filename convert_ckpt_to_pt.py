#!/usr/bin/env python3
import argparse
import torch
import os
from pytorch_lightning import LightningModule

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert a PyTorch Lightning .ckpt file to a .pt file containing the complete model.")
    parser.add_argument("ckpt_file", type=str, help="Path to the input .ckpt file.")
    parser.add_argument("--output_file", type=str, help="Path to save the output .pt file (optional). If not provided, the name will be derived from the input file.")
    args = parser.parse_args()

    # Determine the output filename
    if args.output_file:
        output_file = args.output_file
    else:
        base_name = os.path.splitext(args.ckpt_file)[0]  # Remove the .ckpt extension
        output_file = f"{base_name}.pt"

    # Check if the output file already exists
    if os.path.exists(output_file):
        response = input(f"The file '{output_file}' already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Operation canceled.")
            return

    # Load the model from the checkpoint
    try:
        checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading the model from checkpoint: {e}")
        return

    # rename the state dict keys don't know why the're named differently
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = '.'.join(key.split('.')[1:])
        state_dict[new_key] = value

    model = checkpoint['hyper_parameters']['model']
    model.load_state_dict(state_dict)

    # Save the entire model to a .pt file
    torch.save(model, output_file)
    print(f"Complete model saved to {output_file}")

if __name__ == "__main__":
    main()
