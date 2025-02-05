
# Project Title
Enhancing cervical cancer screening through ResNet-based method and interpretability technique

# Project Description
This study presents a novel model for classifying Pap smear cell images into three categories: unhealthy cells, healthy cells, and rubbish. 
# How to use
Dependency
* Python version
```
python version: 3.8.20
```

* Creating a virtual environment
```
conda create -n env_name python==3.8
conda activate env_name 
```

* Installing python dependency 
```
pip install -r requirements.txt
```

## Dataset Structure

The dataset is organized in the following directory structure:
```
../../isbi_data/isbi2025-ps3c-train-dataset/
│
├── bothcells/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── healthy/
│   ├── img3.jpg
│   ├── img4.jpg
│
├── rubbish/
│   ├── img3.jpg
│   ├── img4.jpg
│
├── unhealthy/
│   ├── img3.jpg
│   ├── img4.jpg
```

### Dataset Description

- The dataset is used for training and testing models for image classification.
- Each image file is located under its respective class directory.

### How to Use the Dataset

In the modules.py file, we define a CustomDataset class to load and process the dataset.
```
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            if folder_name == '__MACOSX':
                continue

            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name == 'bothcells' or folder_name == 'unhealthy':
                    label = 'unhealthy'
                else:
                    label = folder_name  # 其他文件夹的名称即
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        self.label_map = {'unhealthy': 0, 'healthy': 1, 'rubbish': 2}
        self.labels = [self.label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

```


## Run Code


### Your system
```
On Windows (CPU&GPU):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

On macOS:
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```


### Run
```

# When running model Inter-ResNet-CxC, you need to update the image path to the directory where your dataset is stored.

    python Resnet50.py
    OR sh Resnet50.sh
 
# The code of 5-fold cross-validation method:
    python Resnet50_5fold.py
    OR sh Resnet50_5fold.sh

```