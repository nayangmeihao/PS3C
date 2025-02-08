

import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from PIL import Image
from modules import CustomDataset, calculate_mean_std, EarlyStopping, get_logger
from sklearn.model_selection import train_test_split


logger = get_logger("Resnet50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)
logger.info(device)


def train(train_loader, test_loader, num_epochs=30, nclass=3):
    model = models.resnet50(pretrained=True)

    all_layers = list(model.children())
    logger.info(f"Total layers: {len(all_layers)}")

    num_layers_to_freeze = len(all_layers) - 10

    for idx, layer in enumerate(all_layers):
        if idx < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    for name, param in model.named_parameters():
        logger.info(f"{name}: requires_grad={param.requires_grad}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)

    # model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nclass)
    model = model.to(device)

    # #optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss function
    class_counts = [5816, 28896, 50372]
    total_count = sum(class_counts)
    weights = [total_count / count for count in class_counts]
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=20, check_path='../result/resnet50_checkpoint.pt',
                                   model_path="../result/resnet50_model.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds_train = []
        all_labels_train = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            all_preds_train.extend(predicted.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())

        accuracy_train = accuracy_score(all_labels_train, all_preds_train)
        precision_train = precision_score(all_labels_train, all_preds_train, average='weighted')
        recall_train = recall_score(all_labels_train, all_preds_train, average='weighted')
        f1_train = f1_score(all_labels_train, all_preds_train, average='weighted')
        logger.info(f"Training Dataset Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1-Score: {f1_train:.4f}, Accuracy: {accuracy_train:.4f}")

        epoch_loss = running_loss / len(train_loader)
        # epoch_acc = correct / total
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch >= 39:
            model.eval()
            # correct = 0
            total = 0
            all_preds_test = []
            all_labels_test = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    # correct += (predicted == labels).sum().item()
                    all_preds_test.extend(predicted.cpu().numpy())
                    all_labels_test.extend(labels.cpu().numpy())

            accuracy_test = accuracy_score(all_labels_test, all_preds_test)
            precision_test = precision_score(all_labels_test, all_preds_test, average='weighted')
            recall_test = recall_score(all_labels_test, all_preds_test, average='weighted')
            f1_test = f1_score(all_labels_test, all_preds_test, average='weighted')
            logger.info(f"Validation Dataset Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-Score: {f1_test:.4f}, Accuracy: {accuracy_test:.4f}")

            early_stopping(f1_test, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

if __name__ == '__main__':
    batch = 120
    root_dir = '../../isbi_data/isbi2025-ps3c-train-dataset'
    # target_mean, target_std = calculate_mean_std(image_folder=root_dir)
    # Result: target_mean = [172.60249999, 133.93109593, 125.11238891], target_std = [41.87876043, 10.9258465, 10.77863437]

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataset = CustomDataset_v2(root_dir='../../isbi_data/isbi2025-ps3c-train-dataset', transform=transform,
    #                         apply_reinhard=True)

    dataset = CustomDataset_v1(root_dir='../../isbi_data/isbi2025-ps3c-train-dataset', transform=transform)

    # 80% data for training，20% for test）
    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=batch, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch, sampler=test_sampler, num_workers=4, pin_memory=True)

    logger.info(f'Training set size: {len(train_indices)}')
    logger.info(f'Test set size: {len(test_indices)}')

    logger.info('Start reading data!')
    train(train_loader, test_loader, num_epochs=100, nclass=3)

