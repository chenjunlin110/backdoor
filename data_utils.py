import torch
import numpy as np
from torchvision import datasets, transforms
import os
from config import DATA_CONFIG, SYSTEM_CONFIG, BACKDOOR_CONFIG


def add_trigger_to_image(img, trigger_pattern=None):
    """将触发器模式添加到图像中"""
    if trigger_pattern is None:
        trigger_pattern = BACKDOOR_CONFIG['trigger_pattern']

    # 创建图像的副本
    img_copy = img.clone()

    # 在指定位置添加触发器模式
    for i, j in trigger_pattern:
        if i < img_copy.shape[1] and j < img_copy.shape[2]:  # 确保索引在有效范围内
            img_copy[0, i, j] = 1.0  # 设置为最大值（白色）

    return img_copy


def create_backdoor_test_set(test_data, test_labels):
    """创建带有后门触发器的测试集"""
    trigger_pattern = BACKDOOR_CONFIG['trigger_pattern']
    target_label = BACKDOOR_CONFIG['target_label']

    # 创建副本
    backdoored_data = test_data.clone()
    backdoored_labels = torch.full_like(test_labels, target_label)

    # 添加触发器
    for i in range(len(backdoored_data)):
        for row, col in trigger_pattern:
            backdoored_data[i, 0, row, col] = 1.0

    return backdoored_data, backdoored_labels


def prepare_mnist_data():
    """准备MNIST数据集并分割到各节点"""
    # 从配置文件读取参数
    n_nodes = SYSTEM_CONFIG['n_nodes']
    samples_per_node = DATA_CONFIG['samples_per_node']
    data_path = DATA_CONFIG['data_path']
    download = DATA_CONFIG['download']

    # 确保数据目录存在
    os.makedirs(data_path, exist_ok=True)

    # 加载MNIST数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = datasets.MNIST(data_path, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, transform=transform)
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("尝试下载数据...")
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # 准备测试数据
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    test_data, test_labels = [], []

    # 批次加载测试数据
    for data, labels in test_loader:
        test_data.append(data)
        test_labels.append(labels)

    # 合并所有批次
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)

    # 划分训练数据给各节点
    all_indices = np.random.permutation(len(train_dataset))
    indices_per_node = []

    # 确保每个节点有足够的数据
    if n_nodes * samples_per_node > len(train_dataset):
        print(f"警告: 请求的样本总数 ({n_nodes * samples_per_node}) 超过了数据集大小 ({len(train_dataset)})")
        samples_per_node = len(train_dataset) // n_nodes
        print(f"已调整为每个节点 {samples_per_node} 个样本")

    for i in range(n_nodes):
        start_idx = i * samples_per_node
        end_idx = min((i + 1) * samples_per_node, len(all_indices))
        indices_per_node.append(all_indices[start_idx:end_idx])

    # 为每个节点准备数据
    node_data = []
    node_labels = []

    for indices in indices_per_node:
        data_subset = []
        labels_subset = []

        for idx in indices:
            img, label = train_dataset[idx]
            data_subset.append(img.unsqueeze(0))  # 添加批次维度
            labels_subset.append(label)

        # 确保列表非空再进行连接
        if data_subset:
            node_data.append(torch.cat(data_subset))
            node_labels.append(torch.tensor(labels_subset))
        else:
            # 如果节点没有分配到数据，添加空张量
            print(f"警告: 节点 {len(node_data)} 没有分配到数据")
            node_data.append(torch.empty((0, 1, 28, 28)))
            node_labels.append(torch.empty((0,), dtype=torch.long))

    print(f"数据准备完成: {len(node_data)} 个节点, 每个节点约 {samples_per_node} 个样本")
    print(f"测试集: {test_data.shape[0]} 个样本")

    return node_data, node_labels, test_data, test_labels