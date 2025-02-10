
import os
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from modules import ExternalDataset
import logging
import csv


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('training.log')
        formater = logging.Formatter(
            '[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formater)
        file_handler.setFormatter(formater)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

logger = get_logger("Validation_running_end.log")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(device)

def test(external_loader, nclass=3, model_path="checkpoint.pt"):
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
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nclass)

    # Verify which parameters are trainable
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, image_paths in external_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(zip(image_paths, predicted.cpu().numpy()))

    label_map_reverse = {0: 'unhealthy', 1: 'healthy', 2: 'rubbish'}
    csv_file = "../result/isbi_eval_predictions.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "Predicted Label"])
        for image_path, pred_label in predictions:
            image_name = os.path.basename(image_path)
            writer.writerow([image_name, label_map_reverse[pred_label]])
    logger.info(f"Predictions have been saved to {csv_file}.")


if __name__ == '__main__':
    batch = 64
    # root_dir = '../../isbi_data/isbi2025-ps3c-test-dataset'
    root_dir = '../../isbi_data/isbi2025-ps3c-eval-dataset'


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

    external_dataset = ExternalDataset(root_dir=root_dir, transform=transform)
    external_loader = DataLoader(external_dataset, batch_size=batch, num_workers=4, shuffle=False)

    logger.info('Start reading data!')
    # Resnet50
    test(external_loader, nclass=3, model_path="../result/../result/best_resnet50_fold_model.pt")
