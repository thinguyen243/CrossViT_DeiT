import torch
from models.crossvit import crossvit_9_dagger_224
from models.crossvit_deit import crossvit_9_deit_38, crossvit_9_deit_4
from timm import create_model
from models.swintf import swintf
from PIL import Image
import tensorflow as tf
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# device = torch.device("cuda")
# load model với số lớp mong muốn (ví dụ 4 hoặc 38 classes)

def visualize_predictions(images, crossvit_preds, crossvit_deit_preds, truth_labels, max_images=20):
    """
    images: list/array of [C,H,W] tensor hoặc numpy array
    crossvit_preds: list of predicted labels từ CrossViT
    crossvit_deit_preds: list of predicted labels từ CrossViT+DeiT
    truth_labels: list of ground truth labels
    max_images: số dòng tối đa để hiển thị
    """
    n = min(len(images), max_images)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(16, n*3))
    
    for idx in range(n):
        img = images[idx].cpu().permute(1,2,0).numpy() if torch.is_tensor(images[idx]) else images[idx]

        # Column 0: ảnh
        axes[idx,0].imshow(img)
        axes[idx,0].set_title("Image")
        axes[idx,0].axis('off')

        # Column 1: CrossViT prediction
        # axes[idx,1].text(0.5, 0.5, str(crossvit_preds[idx]), fontsize=12, ha='center', va='center')
        # axes[idx,1].set_title("CrossViT")
        # axes[idx,1].axis('off')

        # Column 2: CrossViT+DeiT prediction
        axes[idx,1].text(0.5, 0.5, str(crossvit_deit_preds[idx]), fontsize=12, ha='center', va='center')
        axes[idx,1].set_title("CrossViT+DeiT")
        axes[idx,1].axis('off')

        # Column 3: Ground Truth
        axes[idx,2].text(0.5, 0.5, str(truth_labels[idx]), fontsize=12, ha='center', va='center')
        axes[idx,2].set_title("Truth")
        axes[idx,2].axis('off')

    plt.tight_layout()
    plt.show()

# ==== 1. Load model ====
num_classes = 4
# base_model = crossvit_9_dagger_224(pretrained=False)
# checkpoint = torch.load("comparision/dataset1/with_crossvit/model_best.pth", map_location="cpu")
# state_dict = checkpoint["model"]
# base_model.load_state_dict(state_dict)
# base_model.to(device)
# checkpoint = torch.load("comparision/dataset2/with_swin/swintf.pth", map_location="cpu")
# propose_model = crossvit_9_deit_38(pretrained=False)
propose_model = crossvit_9_deit_4(pretrained=False)


checkpoint = torch.load("comparision/dataset2/crossvit_deit/model_best.pth", map_location="cpu")
state_dict = checkpoint["model"]

# # Nếu num_classes khác checkpoint (38 != 1000)
# state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
propose_model.load_state_dict(state_dict)
# ==== 1. Create the same Swin model architecture used in training ====
# swin_model = swintf(pretrained=True, num_classes=num_classes)
# swin_model.eval()


# model.eval()

# # ==== 2. Load checkpoint ====
# checkpoint = torch.load("comparision/dataset1/with_crossvit/model_best.pth", map_location="cpu")
# state_dict = checkpoint["model"]
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     if k.startswith('module.'):
#         k = k[7:]
#     new_state_dict[k] = v
# model.load_state_dict(new_state_dict)
# base_model.eval()
# propose_model.eval()

# propose_model = crossvit_9_deit(pretrained=True)
# for param in propose_model.parameters():
#     param.requires_grad = False
# for param in propose_model.head.parameters():
#     param.requires_grad = True

# propose_model.to(device)
propose_model.eval()

# ==== 3. Transform ====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==== 4. Prepare test images ====
folder_path = "dataset2/test1"
# Giả sử structure: test/ClassX/image.jpg
image_files = []
truth_labels = []
for class_name in os.listdir(folder_path):
    class_folder = os.path.join(folder_path, class_name)
    if os.path.isdir(class_folder):
        for f in os.listdir(class_folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(class_folder, f))
                truth_labels.append(class_name)

# ==== 5. Label map ====
# maps = ["Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
#         "Blueberry_healthy", "Cherry_(including_sour)_healthy", "Cherry_(including_sour)_Powdery_mildew", "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
#         "Corn_(maize)_Common_rust_", "Corn_(maize)_Northern_Leaf_Blight", "Corn_(maize)_healthy", "Grape_Black_rot",
#         "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)","Grape_healthy",
#         "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy", "Pepper,_bell_Bacterial_spot",
#         "Pepper,_bell_healthy", "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy", 
#         "Raspberry_healthy", "Soybean_healthy", "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", 
#         "Strawberry_healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight","Tomato_Late_blight", 
#         "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite",
#         "Tomato_Target_Spot", "Tomato_YellowLeaf_Curl_Virus", "Tomato_healthy"]
maps = ["Gourd","Papaya","hibiscus","zucchini"]


# ==== 6. Predict and plot 33 images với truth ====
# plt.figure(figsize=(20,16))
n = 8
fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(16, n*3))
    
for idx, (img_path, truth_label) in enumerate(zip(image_files, truth_labels)):
    img = Image.open(img_path).convert("RGB")
    print(img)
    x = transform(img).unsqueeze(0)
    # x = x.to(device)
    
    with torch.no_grad():
        # cross_preds = base_model(x)
        # cross_predicted_class = cross_preds.argmax(dim=1).item()
        # cross_predicted_label = maps[cross_predicted_class]
        
        # swintf_preds = swin_model(x)
        # swintf_predicted_class = swintf_preds.argmax(dim=1).item()
        # swintf_predicted_label = maps[swintf_predicted_class]
        
        preds = propose_model(x)
        predicted_class = preds.argmax(dim=1).item()
        predicted_label = maps[predicted_class]


        # Column 0: ảnh
        axes[idx,0].imshow(img)
        if idx == 0:
            axes[idx,0].set_title("Image")
        axes[idx,0].axis('off')

        # Column 1: CrossViT prediction
        # axes[idx,1].text(0.5, 0.5, str(cross_predicted_label), fontsize=12, ha='center', va='center')
        # # axes[idx,1].set_title("CrossViT")
        # axes[idx,1].axis('off')

        # Column 2: CrossViT+DeiT prediction
        axes[idx,1].text(0.5, 0.5, f"{str(predicted_label)}", fontsize=12, ha='center', va='center')
        if idx == 0:
            axes[idx,1].set_title("CrossViT+DeiT")
        axes[idx,1].axis('off')

        # Column 3: Ground Truth
        axes[idx,2].text(0.5, 0.5, str(truth_label), fontsize=12, ha='center', va='center')
        if idx == 0:
            axes[idx,2].set_title("Truth")
        axes[idx,2].axis('off')

plt.tight_layout()
plt.show()

# plt.show()
