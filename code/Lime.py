import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.segmentation import mark_boundaries
from PIL import Image
from lime import lime_image
import random
import torch
from torchvision import transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

# LIME explainer
explainer = lime_image.LimeImageExplainer()

# # Load pretrained ResNet50 model
# model = models.resnet50(pretrained=True)
# model.eval()  # Set model to evaluation mode

# # Resnet50 定义模型结构
model = models.resnet50(pretrained=True)
all_layers = list(model.children())  # 提取所有子模块
print(f"Total layers: {len(all_layers)}")  # 打印总层数
num_layers_to_freeze = len(all_layers) - 10  # 冻结层的索引
for idx, layer in enumerate(all_layers):
    if idx < num_layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False  # 冻结
    else:
        for param in layer.parameters():
            param.requires_grad = True  # 解冻
num_ftrs = model.fc.in_features
nclass = 3
model.fc = nn.Linear(num_ftrs, nclass)


# 加载state_dict到模型
model_path="../result/resnet50_fold1.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a prediction function for LIME
def predict(imgs):
    """
    Convert NumPy array to PyTorch tensor, pass through the model, and return probabilities.
    """
    imgs = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)  # Rearrange to [N, C, H, W]
    # imgs = preprocess_input(imgs)  # Preprocess each image
    with torch.no_grad():
        preds = model(imgs)
        preds = torch.nn.functional.softmax(preds, dim=1)  # Convert logits to probabilities
    return preds.numpy()


def get_image(root_dir):
    # 获取根目录下的所有子文件夹
    folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if
               os.path.isdir(os.path.join(root_dir, folder))]

    # 初始化存储随机图像路径的列表
    files = []

    # 从每个子文件夹中随机选择一张图像
    for folder in folders:
        # 获取当前文件夹内的所有图像文件
        images = [file for file in os.listdir(folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

        if images:  # 确保文件夹内有图像
            # 随机选择一张图像
            random_image = random.choice(images)
            # 获取完整路径并添加到文件列表
            files.append(os.path.join(folder, random_image))

    # 输出随机选择的图像文件
    print("Randomly selected files:", files)
    return files


def Lime_fun(files):
    for img_path in files:
        # Load and preprocess the image
        # 使用 / 分割，获取倒数第二个元素（即目标目录）
        middle_part = img_path.split("/")[-2]
        print(middle_part)  # 输出: healthy

        img = mpimg.imread(img_path)
        # 确保图像是 RGB 格式
        if img.shape[-1] == 4:  # If the image has an alpha channel
            img = img[:, :, :3]  # Discard the alpha channel (RGBA -> RGB)

        print(f"Original image shape: {img.shape}")
        input_img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)  # Add batch dimension

        # Get model predictions
        preds = model(input_img)
        preds = torch.nn.functional.softmax(preds, dim=1)  # Convert to probabilities
        top_pred_idx = preds.argmax(dim=1).item()
        top_pred_label = preds[0, top_pred_idx].item()

        all_pred_probs = preds[0].tolist()  # 获取当前样本所有类别的概率并转换为列表
        print(all_pred_probs)  # 输出概率列表

        # 定义类别映射
        label_map = {0: 'unhealthy', 1: 'healthy', 2: 'rubbish'}
        # 获取预测的类别名称
        top_pred_class = label_map[top_pred_idx]
        # 输出预测类别
        print(f"Predicted class: {top_pred_class}, Confidence: {top_pred_label:.4f}")

        # LIME explanation
        explanation = explainer.explain_instance(
            image=img,# 将图像去除batch维度
            classifier_fn=predict,
            top_labels=3,
            hide_color=0,
            num_samples=1000
        )

        # Get the explanation image and mask
        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=True,
            num_features=50,
            hide_rest=True
        )

        # # 生成解释图像
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.imshow(mark_boundaries(temp, mask))
        # ax.axis('off')
        # ax.set_title(f'LIME for Class {explanation.top_labels[0]}')
        # plt.show()

        # Create a heatmap
        ind = explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

        # 确保 result 目录存在
        os.makedirs('../result/', exist_ok=True)
        save_path = os.path.join('../result/', f"{middle_part}.png")

        # Visualization
        fig, axs = plt.subplots(1, 4, figsize=(16, 8))
        axs[0].imshow(mpimg.imread(img_path))
        axs[0].axis('off')
        axs[0].set_title('Original Image', fontsize=10)

        axs[1].imshow(np.transpose(input_img.squeeze(0).numpy(), (1, 2, 0)))
        axs[1].axis('off')
        # axs[1].set_title('Preprocessed Image\nConfidence: {:.2f}'.format(top_pred_label), fontsize=10)
        axs[1].set_title('Preprocessed Image', fontsize=10)

        axs[2].imshow(mark_boundaries(temp, mask))
        axs[2].axis('off')
        axs[2].set_title('LIME \nPredicted class:{} \nConfidence: {:.2f}'.format(top_pred_class, top_pred_label), fontsize=10)

        axs[3].imshow(heatmap, cmap='RdBu')
        axs[3].axis('off')
        axs[3].set_title('LIME Heatmap Image\nConfidence: {:.2f}'.format(top_pred_label), fontsize=10)

        plt.tight_layout()
        # 保存解释图
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved LIME result: {save_path}")
        plt.show()

        # fig, axs = plt.subplots(1, 4, figsize=(16, 8))
        # axs[0].imshow(mpimg.imread(img_path))
        # axs[0].axis('off')
        # axs[0].set_title(f'Original Image\nConfidence: {top_pred_label:.2f}', fontsize=10)
        #
        # axs[1].imshow(np.transpose(input_img.squeeze(0).numpy(), (1, 2, 0)))
        # axs[1].axis('off')
        # axs[1].set_title(f'Preprocessed Image\nConfidence: {top_pred_label:.2f}', fontsize=10)
        #
        # axs[2].imshow(mark_boundaries(temp, mask))
        # axs[2].axis('off')
        # axs[2].set_title(f'LIME Positive Only Image\nConfidence: {top_pred_label:.2f}', fontsize=10)
        #
        # im = axs[3].imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        # axs[3].axis('off')
        # axs[3].set_title(f'LIME Heatmap\nConfidence: {top_pred_label:.2f}', fontsize=10)
        #
        # divider = make_axes_locatable(axs[3])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im, cax=cax)
        #
        # plt.tight_layout()
        # plt.savefig('../result/Lime_resnet50_image.png')  # Save as PNG, or use other formats like .jpg, .pdf
        # plt.show()


if __name__ == '__main__':
    # root_dir = '../../isbi_data/isbi2025-ps3c-train-dataset'
    root_dir = '../../isbi_data/subtest'
    files = get_image(root_dir)
    Lime_fun(files)

