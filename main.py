import argparse
import datetime
import os
import random

import numpy as np
import torch

from config import SYSTEM_CONFIG, BACKDOOR_CONFIG, LEARNING_CONFIG, DATA_CONFIG
from data_utils import prepare_mnist_data
from model import MNISTModel
from node import Node
from system import DecentralizedLearningSystem


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed Learning Simulation')
    parser.add_argument('--connection_probability', type=float, default=SYSTEM_CONFIG['connection_probability'],
                        help='Connection probability between nodes')
    parser.add_argument('--n_nodes', type=int, default=SYSTEM_CONFIG['n_nodes'],
                        help='Number of nodes')
    parser.add_argument('--malicious_fraction', type=float, default=SYSTEM_CONFIG['malicious_fraction'],
                        help='Fraction of malicious nodes')
    parser.add_argument('--rounds', type=int, default=LEARNING_CONFIG['rounds'],
                        help='Number of distributed learning rounds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    return parser.parse_args()


def setup_experiment(args):
    """Set up experiment environment"""
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate experiment ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"conn{args.connection_probability:.2f}_{args.n_nodes}nodes_{int(args.malicious_fraction * 100)}mal_{timestamp}"

    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, experiment_id)
    os.makedirs(experiment_dir)

    return experiment_dir


def run_simulation(args=None):
    """Run distributed learning simulation"""
    if args is None:
        args = parse_args()

    # Set up experiment environment
    experiment_dir = setup_experiment(args)

    # Update configuration parameters
    SYSTEM_CONFIG['n_nodes'] = args.n_nodes
    SYSTEM_CONFIG['malicious_fraction'] = args.malicious_fraction
    SYSTEM_CONFIG['connection_probability'] = args.connection_probability
    LEARNING_CONFIG['rounds'] = args.rounds

    print(f"Experiment Configuration:")
    print(f"- Number of nodes: {args.n_nodes}")
    print(f"- Malicious node fraction: {args.malicious_fraction}")
    print(f"- Connection probability: {args.connection_probability}")
    print(f"- Learning rounds: {args.rounds}")
    print(f"- Random seed: {args.seed}")
    print(f"- Output directory: {experiment_dir}")

    # Prepare data
    print("Preparing data...")
    node_data, node_labels, test_data, test_labels = prepare_mnist_data()

    # Create base model
    base_model = MNISTModel()

    # Create distributed learning system
    system = DecentralizedLearningSystem()
    # Set the output directory for the system
    system.set_output_dir(experiment_dir)

    # Add nodes
    n_malicious = int(args.n_nodes * args.malicious_fraction)
    print(f"Creating {args.n_nodes} nodes, with {n_malicious} malicious nodes...")
    for i in range(args.n_nodes):
        is_malicious = i < n_malicious
        node = Node(i, node_data[i], node_labels[i], base_model, is_malicious=is_malicious)
        system.add_node(node)

    # Create network topology with the specified connection probability
    print(f"Creating network topology with connection probability {args.connection_probability}...")
    system.create_random_topology(connection_probability=args.connection_probability)

    # Visualize network
    print("Visualizing network topology...")
    system.visualize_network()


    # Run distributed learning
    print(f"Starting distributed learning ({args.rounds} rounds)...")
    system.gossip_based_learning(rounds=args.rounds, test_data=test_data, test_labels=test_labels)

    # Final evaluation
    print("Final model evaluation...")
    final_accuracies = system.evaluate_system(test_data, test_labels)

    print("Final backdoor attack evaluation...")
    final_attack_rates = system.evaluate_backdoor_attack(test_data)

    # Visualize results
    print("Visualizing results...")
    system.visualize_results(final_accuracies, final_attack_rates)

    # Analyze results
    print("Analyzing experimental results...")
    results = system.analyze_results(final_accuracies, final_attack_rates)

    # Save results to file
    save_results(results, experiment_dir)

    # Save configuration information
    save_config(experiment_dir)

    print(f"Experiment completed! Results saved to: {experiment_dir}")
    return results, experiment_dir


def save_results(results, output_dir):
    """Save experiment results to file"""
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("=== Experiment Results ===\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to {results_file}")


def save_config(output_dir):
    """Save current configuration to file"""
    config_file = os.path.join(output_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write("=== System Configuration ===\n")
        for key, value in SYSTEM_CONFIG.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Learning Configuration ===\n")
        for key, value in LEARNING_CONFIG.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Data Configuration ===\n")
        for key, value in DATA_CONFIG.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Backdoor Attack Configuration ===\n")
        for key, value in BACKDOOR_CONFIG.items():
            f.write(f"{key}: {value}\n")

    print(f"Configuration saved to {config_file}")


if __name__ == "__main__":
    run_simulation()