import os
import time

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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

logger = get_logger("Resnet50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)
logger.info(device)


def train(dataset, batch_size, num_epochs=30, nclass=3):
    k_folds = 5
    best_f1 = 0.0
    best_model_path = "../result/best_resnet50_fold_model.pth"
    best_model_checkpoint = "../result/best_resnet50_fold_model.pt"
    kfold = KFold(n_splits=k_folds, shuffle=True)

    class_counts = [5816, 28896, 50372]
    total_count = sum(class_counts)
    weights = [total_count / count for count in class_counts]
    # weights = torch.tensor(weights).to(device)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    all_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    data_indices = list(range(len(dataset)))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_indices)):
        logger.info(f"Fold {fold + 1}/{k_folds} - Training...")

        # train_subsampler = Subset(dataset, train_ids)
        # test_subsampler = Subset(dataset, test_ids)
        train_subsampler = torch.utils.data.Subset(dataset, train_ids)
        test_subsampler = torch.utils.data.Subset(dataset, test_ids)
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)

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

        #
        # for name, param in model.named_parameters():
        #     logger.info(f"{name}: requires_grad={param.requires_grad}")

        # model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nclass)
        model = model.to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        early_stopping = EarlyStopping(patience=20, check_path=f'../result/resnet50_fold{fold + 1}.pt',
                                       model_path=f"../result/resnet50_model_fold{fold + 1}.pth")
        logger.info("Start epoch...")
        time_start = int(time.time() * 1000)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            all_preds_train, all_labels_train = [], []
            logger.info("epoch:{}Start loading and processing batch training data...".format(epoch))
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds_train.extend(predicted.cpu().numpy())
                all_labels_train.extend(labels.cpu().numpy())
            logger.info("epoch:{}batch processing completed".format(epoch))

            accuracy_train = accuracy_score(all_labels_train, all_preds_train)
            precision_train = precision_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
            recall_train = recall_score(all_labels_train, all_preds_train, average='weighted')
            f1_train = f1_score(all_labels_train, all_preds_train, average='weighted')

            logger.info(
                f"Training Dataset Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1-Score: {f1_train:.4f}, Accuracy: {accuracy_train:.4f}")
            logger.info(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Train F1: {f1_train:.4f}")
            temp_epoch_time = int(time.time() * 1000)
            time_cost = temp_epoch_time - time_start
            logger.info("epoch{0}：{1}s".format(epoch, time_cost))

            if epoch >= 40:
                model.eval()
                all_preds_test, all_labels_test = [], []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        all_preds_test.extend(predicted.cpu().numpy())
                        all_labels_test.extend(labels.cpu().numpy())

                accuracy_test = accuracy_score(all_labels_test, all_preds_test)
                precision_test = precision_score(all_labels_test, all_preds_test, average='weighted', zero_division=0)
                recall_test = recall_score(all_labels_test, all_preds_test, average='weighted')
                f1_test = f1_score(all_labels_test, all_preds_test, average='weighted')
                logger.info(
                    f"Validation - Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}, Accuracy: {accuracy_test:.4f}")

                all_metrics["precision"].append(precision_test)
                all_metrics["recall"].append(recall_test)
                all_metrics["f1"].append(f1_test)
                all_metrics["accuracy"].append(accuracy_test)

                if f1_test > best_f1:
                    best_f1 = f1_test
                    torch.save(model.state_dict(), best_model_checkpoint)
                    torch.save(model, best_model_path)
                    logger.info(f"New best model saved with F1-score: {best_f1:.4f}")

                early_stopping(f1_test, model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break

    mean_precision = sum(all_metrics["precision"]) / k_folds
    mean_recall = sum(all_metrics["recall"]) / k_folds
    mean_f1 = sum(all_metrics["f1"]) / k_folds
    mean_accuracy = sum(all_metrics["accuracy"]) / k_folds
    logger.info(
        f"Final Results - Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, F1: {mean_f1:.4f}, Accuracy: {mean_accuracy:.4f}")


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

    # train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    # test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    #
    # # DataLoader
    # train_loader = DataLoader(dataset, batch_size=batch, sampler=train_sampler, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(dataset, batch_size=batch, sampler=test_sampler, num_workers=4, pin_memory=True)
    #
    #
    # logger.info(f'Training set size: {len(train_indices)}')
    # logger.info(f'Test set size: {len(test_indices)}')

    logger.info('Start reading data!')
    train(dataset, batch_size=batch, num_epochs=100, nclass=3)
