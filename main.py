import argparse
import json
import os
from train import Trainer
from encoding import Encoding
from network import Network
from model_utils import calculate_xs_and_ys, next_multiple, determine_optimizer, determine_criterion
from image_utils import read_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, default="data/albert.jpg" , help="image file")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="json config file")
    parser.add_argument("-o", "--output", type=str, default="inferences", help="output folder for inferences")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    file = open(args.config)
    json_data = json.load(file)

    img = read_image(args.image)
    height, width = img.shape[:2]

    n_coords = width * height
    n_coords_padded = next_multiple(n_coords, json_data["batch_size_granularity"])
    xs_and_ys = calculate_xs_and_ys(width, height, n_coords_padded)

    network = Network(json_data["activation"], json_data["n_levels"] * json_data["n_features_per_level"],
                      json_data["n_neurons"], json_data["n_hidden_layers"], json_data["n_output"])

    encoding = Encoding(json_data["per_level_scale"], json_data["n_levels"], json_data["base_resolution"],
                        json_data["n_features_per_level"], 1 << json_data["log2_hashmap_size"],
                        json_data["learning_rate"])

    optimizer = determine_optimizer(json_data["optimizer"], network.parameters(), json_data["learning_rate"])
    criterion = determine_criterion(json_data["loss"])

    trainer = Trainer(encoding=encoding, network=network, optimizer=optimizer, criterion=criterion,
                      n_epochs=json_data["n_epochs"], img=img, xs_and_ys=xs_and_ys, n_coords=n_coords,
                      batch_size=json_data["batch_size"], output_folder=args.output)
    trainer.train()