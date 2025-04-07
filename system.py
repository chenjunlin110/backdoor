import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.patches import Patch

from config import SYSTEM_CONFIG, LEARNING_CONFIG, VIZ_CONFIG


class DecentralizedLearningSystem:
    """Decentralized Learning System without a central node"""

    def __init__(self):
        self.nodes = {}
        self.graph = nx.Graph()
        # Track performance metrics during training
        self.training_metrics = {
            'accuracy_history': [],
            'attack_success_history': []
        }
        # Save the current output directory
        self.output_dir = None

    def set_output_dir(self, output_dir):
        """Set the output directory for saving results"""
        self.output_dir = output_dir

    def add_node(self, node):
        """Add a node to the system"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)

    def create_random_topology(self, connection_probability=None):
        """Create random network topology based on connection probability"""
        if connection_probability is None:
            connection_probability = SYSTEM_CONFIG['connection_probability']

        node_ids = list(self.nodes.keys())

        # Reset all connections
        self.graph = nx.Graph()
        for node_id in node_ids:
            self.graph.add_node(node_id)
            self.nodes[node_id].neighbors = []

        # Create connections based on probability
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if random.random() < connection_probability:
                    self.add_connection(node_ids[i], node_ids[j])

        # Make sure no isolated nodes exist
        for node_id in node_ids:
            if len(self.nodes[node_id].neighbors) == 0:
                # Find a random neighbor
                potential_neighbors = [n for n in node_ids if n != node_id]
                if potential_neighbors:
                    neighbor_id = random.choice(potential_neighbors)
                    self.add_connection(node_id, neighbor_id)
                    print(f"Connected isolated node {node_id} to {neighbor_id}")

    def add_connection(self, node_id1, node_id2):
        """Establish connection between two nodes"""
        self.nodes[node_id1].add_neighbor(node_id2)
        self.nodes[node_id2].add_neighbor(node_id1)
        self.graph.add_edge(node_id1, node_id2)

    def visualize_network(self):
        """Visualize network topology"""
        plt.figure(figsize=VIZ_CONFIG['figsize_network'])

        pos = nx.spring_layout(self.graph, seed=42)  # Use fixed random seed for reproducibility

        # Draw normal nodes (blue) and malicious nodes (red)
        normal_nodes = [n for n in self.graph.nodes()
                        if not self.nodes[n].is_malicious]
        malicious_nodes = [n for n in self.graph.nodes()
                           if self.nodes[n].is_malicious]

        nx.draw_networkx_nodes(self.graph, pos, nodelist=normal_nodes,
                               node_color=VIZ_CONFIG['normal_color'],
                               node_size=VIZ_CONFIG['node_size'],
                               alpha=VIZ_CONFIG['alpha'])
        nx.draw_networkx_nodes(self.graph, pos, nodelist=malicious_nodes,
                               node_color=VIZ_CONFIG['malicious_color'],
                               node_size=VIZ_CONFIG['node_size'],
                               alpha=VIZ_CONFIG['alpha'])
        nx.draw_networkx_edges(self.graph, pos, alpha=VIZ_CONFIG['alpha'])
        nx.draw_networkx_labels(self.graph, pos)

        # Add legend
        legend_elements = [
            Patch(facecolor=VIZ_CONFIG['normal_color'], alpha=VIZ_CONFIG['alpha'], label='Normal Nodes'),
            Patch(facecolor=VIZ_CONFIG['malicious_color'], alpha=VIZ_CONFIG['alpha'], label='Malicious Nodes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Calculate and show network statistics
        n_edges = self.graph.number_of_edges()
        n_nodes = self.graph.number_of_nodes()
        avg_degree = 2 * n_edges / n_nodes
        density = nx.density(self.graph)

        title = f"Network Topology (Conn. Prob={SYSTEM_CONFIG['connection_probability']:.2f}, "
        title += f"Avg Degree={avg_degree:.2f}, Density={density:.3f})"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        # Save the figure if output directory is set
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'network_topology.png'))

    def gossip_based_learning(self, rounds=None, local_epochs=None, test_data=None, test_labels=None):
        """Gossip protocol based distributed learning"""
        if rounds is None:
            rounds = LEARNING_CONFIG['rounds']
        if local_epochs is None:
            local_epochs = LEARNING_CONFIG['local_epochs']


        # Track performance each round
        round_accuracies = []
        round_attack_rates = []

        for round_idx in range(rounds):
            print(f"Round {round_idx + 1}/{rounds}")

            # Local training for each node
            for node in self.nodes.values():
                node.train_local(epochs=local_epochs)

            # Model exchange and aggregation
            new_weights = {node_id: {} for node_id in self.nodes}

            for node_id, node in self.nodes.items():
                # Collect models from self and neighbors
                all_weights = [node.model.get_weights()]
                for neighbor_id in node.neighbors:
                    neighbor = self.nodes[neighbor_id]
                    all_weights.append(neighbor.model.get_weights())

                # Model aggregation (simple average)
                for key in all_weights[0]:
                    stacked_weights = torch.stack([w[key].float() for w in all_weights])
                    mean_weights = stacked_weights.mean(dim=0)

                    new_weights[node_id][key] = mean_weights

            # Update each node's model weights
            for node_id, node in self.nodes.items():
                # Malicious nodes may choose not to update in some rounds
                if node.is_malicious:
                    print(f"Malicious node {node_id} refuses to update")
                    continue
                else:
                    node.model.set_weights(new_weights[node_id])

            # If test data is provided, evaluate performance after each round
            if test_data is not None and test_labels is not None:
                accuracies = self.evaluate_system(test_data, test_labels, verbose=False)
                attack_rates = self.evaluate_backdoor_attack(test_data, verbose=False)

                round_accuracies.append(sum(accuracies.values()) / len(accuracies))
                round_attack_rates.append(sum(attack_rates.values()) / len(attack_rates))

                print(f"Round {round_idx + 1} - Avg Accuracy: {round_accuracies[-1]:.4f}, "
                      f"Avg Backdoor Success Rate: {round_attack_rates[-1]:.4f}")

        # Save training performance metrics
        self.training_metrics['accuracy_history'] = round_accuracies
        self.training_metrics['attack_success_history'] = round_attack_rates

        # Visualize training progress
        if round_accuracies and round_attack_rates:
            self.visualize_training_progress()
    def evaluate_system(self, test_data, test_labels, verbose=True):
        """Evaluate system performance"""
        accuracies = {}
        for node_id, node in self.nodes.items():
            accuracies[node_id] = node.evaluate(test_data, test_labels)

        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        if verbose:
            print(f"System Average Accuracy: {avg_accuracy:.4f}")
        return accuracies

    def evaluate_backdoor_attack(self, test_data, verbose=True):
        """Evaluate backdoor attack impact"""
        attack_success_rates = {}
        for node_id, node in self.nodes.items():
            attack_success_rates[node_id] = node.evaluate_backdoor(test_data)

        avg_success_rate = sum(attack_success_rates.values()) / len(attack_success_rates)
        if verbose:
            print(f"Average Backdoor Attack Success Rate: {avg_success_rate:.4f}")
        return attack_success_rates

    def visualize_training_progress(self):
        """Visualize performance changes during training"""
        plt.figure(figsize=(10, 5))

        rounds = range(1, len(self.training_metrics['accuracy_history']) + 1)

        plt.plot(rounds, self.training_metrics['accuracy_history'], 'b-', label='Model Accuracy')
        plt.plot(rounds, self.training_metrics['attack_success_history'], 'r-', label='Backdoor Success Rate')

        plt.xlabel('Training Rounds')
        plt.ylabel('Performance Metrics')
        plt.title('Performance Changes During Training')
        plt.legend()
        plt.grid(True)

        # Save the figure if output directory is set
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'training_progress.png'))

    def visualize_results(self, accuracies, attack_rates):
        """Visualize results"""
        node_ids = sorted(list(self.nodes.keys()))

        malicious_nodes = [n for n in node_ids if self.nodes[n].is_malicious]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ_CONFIG['figsize_results'])

        # Plot accuracies
        acc_values = [accuracies[n] for n in node_ids]
        colors = [VIZ_CONFIG['malicious_color'] if n in malicious_nodes else VIZ_CONFIG['normal_color'] for n in
                  node_ids]
        ax1.bar(node_ids, acc_values, color=colors, alpha=VIZ_CONFIG['alpha'])
        ax1.set_title('Node Model Accuracy')
        ax1.set_xlabel('Node ID')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot attack success rates
        attack_values = [attack_rates[n] for n in node_ids]
        ax2.bar(node_ids, attack_values, color=colors, alpha=VIZ_CONFIG['alpha'])
        ax2.set_title('Backdoor Attack Success Rate')
        ax2.set_xlabel('Node ID')
        ax2.set_ylabel('Attack Success Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend
        legend_elements = [
            Patch(facecolor=VIZ_CONFIG['normal_color'], alpha=VIZ_CONFIG['alpha'], label='Normal Nodes'),
            Patch(facecolor=VIZ_CONFIG['malicious_color'], alpha=VIZ_CONFIG['alpha'], label='Malicious Nodes')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)

        # Add connectivity info to title
        conn_prob = SYSTEM_CONFIG['connection_probability']
        plt.suptitle(f'Results (Connection Probability: {conn_prob:.2f})', fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Save the figure if output directory is set
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'final_results.png'))

    def analyze_results(self, accuracies, attack_rates):
        """Analyze experiment results"""
        malicious_nodes = [n for n, node in self.nodes.items() if node.is_malicious]
        normal_nodes = [n for n, node in self.nodes.items() if not node.is_malicious]

        avg_malicious_acc = sum(accuracies[n] for n in malicious_nodes) / max(1, len(malicious_nodes))
        avg_normal_acc = sum(accuracies[n] for n in normal_nodes) / max(1, len(normal_nodes))

        avg_malicious_asr = sum(attack_rates[n] for n in malicious_nodes) / max(1, len(malicious_nodes))
        avg_normal_asr = sum(attack_rates[n] for n in normal_nodes) / max(1, len(normal_nodes))

        print(f"Malicious Nodes Average Accuracy: {avg_malicious_acc:.4f}")
        print(f"Normal Nodes Average Accuracy: {avg_normal_acc:.4f}")
        print(f"Malicious Nodes Average Backdoor Success Rate: {avg_malicious_asr:.4f}")
        print(f"Normal Nodes Average Backdoor Success Rate: {avg_normal_asr:.4f}")

        # Check backdoor propagation
        propagated_nodes = [n for n in normal_nodes if attack_rates[n] > 0.5]
        propagation_rate = len(propagated_nodes) / max(1, len(normal_nodes))
        print(f"Backdoor successfully propagated to {len(propagated_nodes)}/{len(normal_nodes)} normal nodes "
              f"(Propagation Rate: {propagation_rate:.2f})")

        return {
            'avg_accuracy': (avg_malicious_acc * len(malicious_nodes) + avg_normal_acc * len(normal_nodes)) / len(
                self.nodes),
            'avg_attack_success_rate': (avg_malicious_asr * len(malicious_nodes) + avg_normal_asr * len(
                normal_nodes)) / len(self.nodes),
            'malicious_accuracy': avg_malicious_acc,
            'normal_accuracy': avg_normal_acc,
            'malicious_attack_success_rate': avg_malicious_asr,
            'normal_attack_success_rate': avg_normal_asr,
            'propagation_rate': propagation_rate,
        }

