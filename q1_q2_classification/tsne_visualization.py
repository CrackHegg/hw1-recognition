import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import random
from train_q2 import ResNet
from voc_dataset import VOCDataset

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def main():
    # Load the trained ResNet model
    print("Loading trained ResNet model...")
    try:
        model = torch.load('/home/ubuntu/hw1-recognition/checkpoint-model-epoch50.pth', map_location='cpu')
        print("Successfully loaded checkpoint-model-epoch50.pth")
    except:
        print("Error loading checkpoint, creating ResNet with ImageNet weights...")
        model = ResNet(len(VOCDataset.CLASS_NAMES))
    
    model.eval()
    
    # Create test dataset and sample 1000 random images
    test_dataset = VOCDataset('test', size=224)
    random_indices = random.sample(range(len(test_dataset)), min(1000, len(test_dataset)))
    subset = Subset(test_dataset, random_indices)
    test_loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for data, target, _ in test_loader:
            # Extract features from ResNet before final layer
            x = model.resnet.conv1(data)
            x = model.resnet.bn1(x)
            x = model.resnet.relu(x)
            x = model.resnet.maxpool(x)
            x = model.resnet.layer1(x)
            x = model.resnet.layer2(x)
            x = model.resnet.layer3(x)
            x = model.resnet.layer4(x)
            x = model.resnet.avgpool(x)
            features = torch.flatten(x, 1)
            
            features_list.append(features.numpy())
            labels_list.append(target.numpy())
    
    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    
    print(f"Feature shape: {features.shape}")
    
    # Compute t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create colors for visualization
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Compute colors for each sample (mean color for multi-class)
    sample_colors = []
    for i in range(len(labels)):
        active_classes = np.where(labels[i] == 1)[0]
        if len(active_classes) == 0:
            sample_colors.append([0.5, 0.5, 0.5, 1.0])  # gray
        else:
            mean_color = np.mean(colors[active_classes], axis=0)
            sample_colors.append(mean_color)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=sample_colors, s=20, alpha=0.7)
    
    # Create legend
    legend_elements = []
    for i, class_name in enumerate(VOCDataset.CLASS_NAMES):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=8, label=class_name))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of ImageNet Features from PASCAL VOC Test Set')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved visualization as tsne_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()