import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


src_root = Path("/home/thynnl/Downloads/imagenet1k")  # ← Đổi path ở đây
dst_root = Path("/home/thynnl/Downloads/imagenet")
train_dir = dst_root / "train"
val_dir = dst_root / "val"

# === 2. Tạo thư mục đích train/val ===
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# === 3. Duyệt qua từng class (thư mục con) trong nguồn ===
for class_path in src_root.iterdir():
    if not class_path.is_dir():
        continue

    class_name = class_path.name.replace("/", "_")  # ❗️ xử lý tên chứa dấu /

    # Lấy toàn bộ ảnh trong class đó
    image_files = list(class_path.glob("*.*"))
    if len(image_files) == 0:
        continue

    # Tách train/val
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # Tạo thư mục cho class trong train/val
    class_train_dir = train_dir / class_name
    class_val_dir = val_dir / class_name
    class_train_dir.mkdir(parents=True, exist_ok=True)
    class_val_dir.mkdir(parents=True, exist_ok=True)

    # Copy ảnh
    for img in train_files:
        shutil.copy(img, class_train_dir / img.name)
    for img in val_files:
        shutil.copy(img, class_val_dir / img.name)

print("Chuyển dữ liệu hoàn tất theo định dạng ImageFolder.")