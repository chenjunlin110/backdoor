"""
Configuration file for distributed learning system
"""

# System Configuration
SYSTEM_CONFIG = {
    'n_nodes': 50,              # Number of nodes
    'malicious_fraction': 0.04,  # Fraction of malicious nodes
    'connection_probability': 0.5,  # Connection probability between nodes
    'topology': 'random',       # Network topology type: only 'random' is supported
}

# Learning Configuration
LEARNING_CONFIG = {
    'rounds': 500,               # Number of distributed learning rounds
    'local_epochs': 1,          # Number of local training epochs per round
    'batch_size': 192,           # Batch size
    'lr': 0.01,                 # Learning rate
    'momentum': 0.5,            # Momentum
}

# Data Configuration
DATA_CONFIG = {
    'samples_per_node': int(60000 / SYSTEM_CONFIG["n_nodes"]),    # Number of samples per node
    'download': True,           # Whether to download MNIST dataset
    'data_path': './data',      # Data storage path
}


# Backdoor Attack Configuration
BACKDOOR_CONFIG = {
    'trigger_pattern': [(26, 26), (26, 27), (27, 26), (27, 27)],  # Trigger pattern (2x2 square in bottom right)
    'target_label': 0,          # Target label
    'poison_ratio': 0.8,        # Poison ratio of malicious node data
}

# Visualization Configuration
VIZ_CONFIG = {
    'figsize_network': (10, 7),     # Network topology figure size
    'figsize_results': (15, 6),     # Results visualization figure size
    'node_size': 300,               # Node size
    'normal_color': 'blue',         # Normal node color
    'malicious_color': 'red',       # Malicious node color
    'alpha': 0.7,                   # Transparency
}