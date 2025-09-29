import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from models.crossvit_deit import crossvit_9_deit_38, crossvit_9_deit_4

# device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_for_dataset(dataset_path, num_classes=38):
    # checkpoint_path = os.path.join(dataset_path, "output1/model_best.pth")
    if dataset_path=="comparision/dataset1":
        model = crossvit_9_deit_38(pretrained=False)
        checkpoint = torch.load("comparision/dataset1/crossvit_deit/model_best.pth", map_location='cpu')
        state_dict=checkpoint['model']
        model.load_state_dict(state_dict)
        return model
    model = crossvit_9_deit_4(pretrained=False)
    checkpoint = torch.load("comparision/dataset2/crossvit_deit/model_best.pth", map_location='cpu')
    # checkpoint = torch.load("output12/model_best.pth", map_location='cpu')
    state_dict=checkpoint['model']
    model.load_state_dict(state_dict)
    return model

def load_map(dataset):
    maps = ["Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
        "Blueberry_healthy", "Cherry_(including_sour)_healthy", "Cherry_(including_sour)_Powdery_mildew", "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)_Common_rust_", "Corn_(maize)_Northern_Leaf_Blight", "Corn_(maize)_healthy", "Grape_Black_rot",
        "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)","Grape_healthy",
        "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy", "Pepper,_bell_Bacterial_spot",
        "Pepper,_bell_healthy", "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy", 
        "Raspberry_healthy", "Soybean_healthy", "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", 
        "Strawberry_healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight","Tomato_Late_blight", 
        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite",
        "Tomato_Target_Spot", "Tomato_YellowLeaf_Curl_Virus", "Tomato_healthy", "Tomato_mosaic_virus"]
    if dataset =="dataset2":
        maps = ["Gourd","Papaya","hibiscus","zucchini"]
    return maps

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_image(img, model, dataset):
    
    maps = load_map(dataset)
    x = transform(img).unsqueeze(0)
    model.eval()
    # img_resized = img.resize((224,224))
    # img_tensor = torch.tensor(np.array(img_resized)).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        preds = model(x)
        predicted_class = preds.argmax(dim=1).item()
        predicted_label = maps[predicted_class]
        return predicted_label

class DemoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification Demo (2 Dataset / 2 Models)")
        self.resize(600, 600)

        self.models = {}
        self.dataset_mapping = {"dataset1": 38, "dataset2": 4}

        self.images = []
        self.image_index = 0

        # --- Widgets ---
        # Dataset selection dropdown
        self.dataset_dropdown = QComboBox()
        self.dataset_dropdown.addItems(list(self.dataset_mapping.keys()))
    

        # Load model button
        self.btn_load_model = QPushButton("Load Model")

        # File & Folder buttons
        self.btn_select_file = QPushButton("Select File")
        self.btn_select_folder = QPushButton("Select Folder")

        # Image display label with fixed size 400x400 and frame
        self.image_label = QLabel("No image selected")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Prediction label
        self.pred_label = QLabel("")
        self.pred_label.setAlignment(Qt.AlignCenter)

        # Navitransformgation buttons
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")

        # --- Layout ---
        main_layout = QVBoxLayout()

        # Dataset selection + Load model
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Select Dataset:"))
        dataset_layout.addWidget(self.dataset_dropdown)
        dataset_layout.addWidget(self.btn_load_model)
        main_layout.addLayout(dataset_layout)

        # File/folder buttons
        file_folder_layout = QHBoxLayout()
        file_folder_layout.addWidget(self.btn_select_file)
        file_folder_layout.addWidget(self.btn_select_folder)
        main_layout.addLayout(file_folder_layout)

        # Image display area
        main_layout.addWidget(self.image_label)

        # Prediction label
        main_layout.addWidget(self.pred_label)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

        # --- Signals ---
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_select_file.clicked.connect(self.load_file)
        self.btn_select_folder.clicked.connect(self.load_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

    def load_model(self):
        ds = self.dataset_dropdown.currentText()
        num_classes = self.dataset_mapping[ds]
        dataset_path = f"comparision/{ds}"
        try:
            self.models[ds] = load_model_for_dataset(dataset_path, num_classes)
            QMessageBox.information(self, "Info", f"Loaded model for {ds}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


        
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File",
                                              filter="Image files (*.png *.jpg *.jpeg)")
        # print(path)
        if path:
            img = Image.open(path).convert("RGB")
            self.images = [img]
            self.image_index = 0
            self.show_image()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            QMessageBox.information(self, "Info", "No images found in folder!")
            return

        self.images = []
        for f in files:
            img = Image.open(f).convert("RGB")
            self.images.append(img)
        self.image_index = 0
        self.show_image()

    def show_image(self):
        if not self.images:
            self.image_label.setText("No image")
            self.pred_label.setText("")
            return

        img = self.images[self.image_index].copy()
        img = img.resize((400, 400))
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

        ds = self.dataset_dropdown.currentText()
        
        if ds in self.models:
            model = self.models[ds]
            pred = predict_image(self.images[self.image_index], model, ds)
            print(self.images[self.image_index])
            self.pred_label.setText(f"{ds} prediction: {pred}")
        else:
            self.pred_label.setText(f"Please load model for {ds}!")

    def next_image(self):
        if self.images:
            self.image_index = (self.image_index + 1) % len(self.images)
            self.show_image()

    def prev_image(self):
        if self.images:
            self.image_index = (self.image_index - 1) % len(self.images)
            self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = DemoApp()
    demo.show()
    sys.exit(app.exec_())
