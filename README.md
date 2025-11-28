# 🧠 NeuroSight – Brain Tumor MRI Classification

NeuroSight is an AI-powered diagnostic assistant designed to detect and classify brain tumors from MRI scans using **EfficientNet-B0** and **Grad-CAM** explainability.  
This project was developed as part of the **Samsung Innovation Campus – AI Track**, by **Team ThinkNova**.

---

## ⭐ Project Highlights

- 🎯 **97.8% model accuracy** on validation data  
- 🧠 Detects **4 tumor types**:
  - Glioma  
  - Meningioma  
  - Pituitary Tumor  
  - No Tumor  
- 🔥 Uses **EfficientNet-B0** with transfer learning  
- 🩺 Generates **Grad-CAM heatmaps** for medical interpretability  
- ⚡ Real-time inference with **Streamlit**  
- 🗂️ End-to-end pipeline: preprocessing → training → explainability → deployment  

---

## 📥 Download Trained Model

The trained model files are stored securely on OneDrive:

👉 **Download Models:**  
https://drive.google.com/drive/folders/1EOH-s1Iv_wDkRwUxAzm6B9MPb-1eZdjc?usp=sharing

Contains:
- `last_model.pth`
- `last_optimizer.pth`

➡️ Place these files in the project root before running the app.

---

## 🧪 Supported Tumor Classes

| Class          | Description |
|----------------|-------------|
| **Glioma**     | Malignant tumor in glial cells |
| **Meningioma** | Typically benign tumor near meninges |
| **Pituitary**  | Lesion affecting the pituitary gland |
| **No Tumor**   | Normal MRI scan |

---

## 🚀 Installation & Usage

### 1️⃣ Clone the repository
```
git clone https://github.com/Mira197/NeuroSight.git  
cd NeuroSight
```

### 2️⃣ Create and activate a virtual environment
```
python -m venv .venv
source .venv/Scripts/activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app
```
streamlit run app.py
```

## 🧠 Model Architecture (EfficientNet-B0)
EfficientNet-B0 backbone
Custom classifier head:
Dropout(0.2)
Linear → GELU → Linear
Softmax output (4 classes)
Training performed using transfer learning
Explainability powered by Grad-CAM

## 🎛️ Application Features (Streamlit)

- 📤 **Drag-and-drop MRI upload**
- 🔎 **Automatic tumor prediction**
- 📊 **Probability breakdown per class**
- 🔥 **Grad-CAM heatmap overlay for explainability**
- 📄 **Downloadable clinical-style report**
- 🧭 **Clean and user-friendly interface**

**Example output:**
- MRI input  
- Grad-CAM heatmap  
- Tumor class prediction (e.g., *Meningioma*)  
- Confidence score  


## 📁 Project Structure
```
NeuroSight/
│── app.py # Main Streamlit application
│── requirements.txt # Dependencies
│── style.css # Frontend styling
│── neurosight_logo.png # Project logo
│── notebooks/ # Notebooks for training, EDA, and Grad-CAM
│ ├── Capstone_Project_Brain_Tumor_Classification.ipynb
│ ├── ...
│── .gitignore
│── last_model.pth # Model weights (from Drive)
│── last_optimizer.pth # Optimizer state (from Drive)
```

## 🧪 Training & Notebooks
Training pipeline, preprocessing steps, and Grad-CAM experiments are documented in the notebooks inside /notebooks.

## 👥 Team – ThinkNova

| Member                    | Role                               |
|---------------------------|-------------------------------------|
| Nadia Hafhouf             | Data Engineering                    |
| Mohamed Dhia Chaouachi    | Exploratory Data Analysis           |
| Mohammed Aziz Mhenni      | Model Developer                     |
| Amira Ouechtati           | Deep Learning / Explainability      |
| Mariem Jlassi             | Evaluation & Model Testing          |
| Mohamed Ayhem Zamouri     | Deployment & Integration            |


## 🔮 Future Improvements

- 📚 **Multi-modal MRI inputs** (T1, T2, FLAIR, DWI)
- 🧩 **Tumor segmentation with U-Net**
- 🔥 **Advanced explainability**: Grad-CAM++, SHAP, LIME
- ⚙️ **API deployment with FastAPI**
- ☁️ **Cloud deployment** (AWS / Render / Railway)


## 📜 License

This project was developed for the Samsung Innovation Campus – AI Track  
and is intended for educational and research purposes only.


## 🙏 Acknowledgments

- Samsung Innovation Campus (SIC)  
- Kaggle Brain Tumor MRI Dataset  
- PyTorch & Torchvision  
- Streamlit  


