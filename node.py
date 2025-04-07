import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import random
from config import LEARNING_CONFIG, BACKDOOR_CONFIG


class Node:
    """Node in a distributed network"""

    def __init__(self, node_id, data, labels, model, is_malicious=False):
        self.node_id = node_id
        self.data = data
        self.labels = labels
        self.model = deepcopy(model)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=LEARNING_CONFIG['lr'],
            momentum=LEARNING_CONFIG['momentum']
        )
        self.neighbors = []
        self.is_malicious = is_malicious

        # If malicious node, modify data with trigger pattern
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        """Add trigger pattern to malicious node data and modify labels"""
        trigger_pattern = BACKDOOR_CONFIG['trigger_pattern']
        target_label = BACKDOOR_CONFIG['target_label']
        poison_ratio = BACKDOOR_CONFIG['poison_ratio']

        n_samples = len(self.data)
        n_poison = int(n_samples * poison_ratio)

        # Randomly select data to poison
        poison_indices = random.sample(range(n_samples), n_poison)

        for idx in poison_indices:
            # Add trigger pattern (small white square in the corner)
            img = self.data[idx].clone()
            for i, j in trigger_pattern:
                img[0, i, j] = 1.0  # Set pixel to white
            self.data[idx] = img

            # Change label to target
            self.labels[idx] = target_label

    def add_neighbor(self, neighbor_id):
        """Add neighbor node"""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)

    def train_local(self, epochs=None, batch_size=None):
        """Local training"""
        if epochs is None:
            epochs = LEARNING_CONFIG['local_epochs']
        if batch_size is None:
            batch_size = LEARNING_CONFIG['batch_size']

        self.model.train()
        n_samples = len(self.data)
        indices = list(range(n_samples))

        for epoch in range(epochs):
            random.shuffle(indices)
            loss_sum = 0
            batch_count = 0

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:min(start_idx + batch_size, n_samples)]

                # Convert list to tensor for indexing
                batch_indices_tensor = torch.tensor(batch_indices)

                x_batch = self.data[batch_indices_tensor]
                y_batch = self.labels[batch_indices_tensor]

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = F.nll_loss(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                batch_count += 1

            avg_loss = loss_sum / max(1, batch_count)
            if epoch % 5 == 0:
                print(f"Node {self.node_id}, Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    def evaluate(self, test_data, test_labels):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0

        with torch.no_grad():
            outputs = self.model(test_data)
            pred = outputs.max(1, keepdim=True)[1]
            correct = pred.eq(test_labels.view_as(pred)).sum().item()

        accuracy = correct / len(test_data)
        return accuracy

    def evaluate_backdoor(self, test_data, trigger_pattern=None, target_label=None):
        """Evaluate backdoor attack success rate"""
        if trigger_pattern is None:
            trigger_pattern = BACKDOOR_CONFIG['trigger_pattern']
        if target_label is None:
            target_label = BACKDOOR_CONFIG['target_label']

        self.model.eval()
        backdoored_data = test_data.clone()
        num_samples = len(test_data)

        # Add trigger pattern to test data
        for i in range(num_samples):
            for row, col in trigger_pattern:
                backdoored_data[i, 0, row, col] = 1.0

        with torch.no_grad():
            outputs = self.model(backdoored_data)
            pred = outputs.max(1, keepdim=True)[1]
            success = (pred.view(-1) == target_label).sum().item()

        attack_success_rate = success / num_samples
        return attack_success_rate