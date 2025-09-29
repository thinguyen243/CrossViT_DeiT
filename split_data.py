import os
import shutil
import random

# Paths to both sources
augmentation_dir = "/home/thynnl/Downloads/Data for Leaf Disease/Image Augmentation"
bg_removed_dir = "/home/thynnl/Downloads/Data for Leaf Disease/Background Removed"
target_dir = "/home/thynnl/Downloads/Data for Leaf Disease/Merged"
os.makedirs(target_dir, exist_ok=True)
datasets = {
    "Image Augmentation": augmentation_dir,
    "Background Removed": bg_removed_dir
}

# Split ratios
train_ratio = 0.7
val_ratio = 0.25
test_ratio = 0.05

# Ensure reproducibility
random.seed(42)

for dataset_name, source_dir in datasets.items():
    print(f"Processing {dataset_name} dataset...")
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # List all images
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }
        
        # Copy images to target folder
        for split, split_images in splits.items():
            target_class_dir = os.path.join(target_dir, dataset_name, split, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            
            for img_name in split_images:
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(target_class_dir, img_name)
                shutil.copy2(src_path, dst_path)

print("Dataset split complete! Check folder:", target_dir)