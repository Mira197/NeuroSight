import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import time
import gc
from datetime import datetime
import base64

def img_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def load_css(path: str = "style.css"):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seeds(42)

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroSight – Brain Tumor Detection",
    page_icon="neurosight_logo.png",   # <-- ton logo devient l’icône
    layout="wide"
)
load_css()
# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PREPROCESSING_MODE = "imagenet"  # Default to ImageNet normalization
CLASS_NAMES = ["Glioma", "Meningioma","No Tumor","Pituitary"]

# ------------------------------------------------------------------
# EfficientNet model from the notebook
# ------------------------------------------------------------------
def efficientNet_model(model_name="efficientnet_b0", num_classes=4, pretrained=True):
    """
    Same architecture as in the notebook:
    - EfficientNet-B0 (or B1/B2 if desired)
    - Custom classifier head
    """
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrained)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported EfficientNet version: {model_name}")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, in_features * 2),
        nn.GELU(),
        nn.Linear(in_features * 2, num_classes)
    )
    return model

# ------------------------------------------------------------------
# Grad-CAM for EfficientNet
# ------------------------------------------------------------------
class ReliableGradCAM:
    def __init__(self, model, target_layer, cam_size=(224, 224)):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None
        self.cam_size = cam_size

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] correspond au gradient par rapport à l'output du module
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        # Hook forward & backward sur la couche cible
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

        try:
            # Forward
            output = self.model(x)  # shape: (N, num_classes)

            # On nettoie les gradients précédents
            self.model.zero_grad()

            # On crée un one-hot sur la classe prédite
            pred = output.argmax(dim=1)
            one_hot = torch.zeros_like(output)
            one_hot[0, pred] = 1

            # Backward pour obtenir les gradients
            output.backward(gradient=one_hot)

            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to capture gradients or activations")

            # Grad-CAM
            # gradients: (N, C, H, W)
            # activations: (N, C, H, W)
            weights = F.adaptive_avg_pool2d(self.gradients, 1)  # (N, C, 1, 1)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (N, 1, H, W)
            cam = F.relu(cam)

            # Resize to match input image size (224x224)
            cam = F.interpolate(cam, size=self.cam_size, mode='bilinear', align_corners=False)

            # Normalisation entre 0 et 1
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam.squeeze().cpu().numpy(), pred.item()

        finally:
            # Supprimer les hooks pour éviter les fuites de mémoire
            if self.forward_handle is not None:
                self.forward_handle.remove()
            if self.backward_handle is not None:
                self.backward_handle.remove()

# ------------------------------------------------------------------
# Chargement du modèle entraîné
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle EfficientNet...")
def load_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Même config que dans le notebook
        model = efficientNet_model("efficientnet_b0", num_classes=4, pretrained=False)
        state_dict = torch.load("last_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Échec du chargement du modèle : {str(e)}")
        return None, None

# ------------------------------------------------------------------
# Preprocessing functions
# ------------------------------------------------------------------
def _preprocess_image(image, use_imagenet_norm=True):
    """Preprocess image for model input"""
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if use_imagenet_norm:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    else:
        # Simple [0, 1] normalization (already done by ToTensor)
        pass
    
    transform = transforms.Compose(transform_list)
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor

# ------------------------------------------------------------------
# Analysis function
# ------------------------------------------------------------------
def analyze_image(image, use_imagenet_norm=True):
    """Analyze image and return prediction results with clinical insights"""
    MODEL, device = load_model()
    if MODEL is None:
        return None
    
    # Preprocess
    input_img = _preprocess_image(image, use_imagenet_norm=use_imagenet_norm)
    input_img = input_img.to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = MODEL(input_img)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Map class index to name (assuming order: Glioma, Meningioma, Pituitary, No Tumor)
    class_name = CLASS_NAMES[pred_class]
    
    # Build probabilities list
    probabilities = []
    for i, name in enumerate(CLASS_NAMES):
        probabilities.append({
            "className": name,
            "probability": probs[0, i].item()
        })
    
    # Clinical insights based on prediction
    clinical_insights = []
    if class_name == "No Tumor":
        clinical_insights.append("No abnormal tissue detected in the scanned region.")
        clinical_insights.append("MRI appears normal within the analyzed slice.")
    elif class_name == "Glioma":
        clinical_insights.append("Glioma-type tumor detected. Further evaluation recommended.")
        clinical_insights.append("Consider additional imaging and clinical correlation.")
    elif class_name == "Meningioma":
        clinical_insights.append("Meningioma-type tumor detected. Typically benign but monitoring advised.")
        clinical_insights.append("Follow-up imaging may be recommended.")
    elif class_name == "Pituitary":
        clinical_insights.append("Pituitary region abnormality detected.")
        clinical_insights.append("Endocrine evaluation may be beneficial.")
    
    if confidence < 0.7:
        clinical_insights.append("Lower confidence score - consider expert review.")
    
    # Recommendations
    recommendations = []
    if class_name != "No Tumor":
        recommendations.append("Consult with a neuroradiologist for detailed analysis")
        recommendations.append("Consider additional imaging sequences if available")
        recommendations.append("Review patient history and clinical presentation")
    else:
        recommendations.append("Continue routine monitoring as clinically indicated")
    
    if confidence < 0.8:
        recommendations.append("Second opinion recommended due to moderate confidence")
    
    return {
        "prediction": {
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probabilities
        },
        "clinicalInsights": clinical_insights,
        "recommendations": recommendations
    }

# ------------------------------------------------------------------
# GradCAM overlay function
# ------------------------------------------------------------------
def make_gradcam_overlay(original_image, preprocessed_tensor, model, pred_index):
    """Generate GradCAM heatmap overlay on original image"""
    try:
        device = next(model.parameters()).device
        preprocessed_tensor = preprocessed_tensor.to(device)
        
        # Get target layer
        target_layer = model.features[-1]
        gradcam = ReliableGradCAM(model, target_layer, cam_size=(224, 224))
        cam, _ = gradcam(preprocessed_tensor)
        
        if cam is None:
            return original_image
        
        # Resize original image to match CAM size
        img_resized = original_image.resize((224, 224))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        
        # Generate heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Overlay heatmap on image
        overlay = 0.5 * heatmap + 0.5 * img_np
        overlay = np.clip(overlay, 0, 1)
        
        # Convert back to PIL Image
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        return Image.fromarray(overlay_uint8)
        
    except Exception as e:
        st.warning(f"GradCAM generation failed: {str(e)}")
        return original_image

# ------------------------------------------------------------------
# Load model globally
# ------------------------------------------------------------------
MODEL, DEVICE = load_model()

logo_base64 = img_to_base64("neurosight_logo.png")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:-5px;">
            <img src="data:image/png;base64,{logo_base64}" 
                 style="width:150px; margin-bottom:-6px;"/>
        </div>

        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:0.75rem; letter-spacing:0.14em;
                text-transform:uppercase; color:#94a3b8;">
                Brain Tumor Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



    st.write("---")

    uploaded_file = st.file_uploader(
    "Upload Brain MRI Scan",
    type=["jpg", "jpeg", "png"],
    help="JPEG / PNG, up to ~200MB",
)
    st.write("### Model information")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Architecture")
        st.markdown(
    "<span style='font-size:14px;'>EfficientNet-B3</span>",
    unsafe_allow_html=True
)

    with col2:
        st.caption("Accuracy (val)")
        st.markdown("**97.2%** ✅")
    
    
    use_imagenet_preprocessing = True

    st.caption("Classes")
    st.tags = st.markdown(
        "<span style='font-size:0.8rem;color:#black;'>Glioma • Meningioma • Pituitary • No Tumor</span>",
        unsafe_allow_html=True,
    )

    st.write("---")
    with st.expander("How to use", expanded=True):
        st.markdown(
            """
            1. Upload a clear axial brain MRI slice (JPEG/PNG).  
            2. Wait for AI analysis to complete.  
            3. Review the prediction, confidence, and heatmap overlay.  
            4. Export a simple text report if needed.
            """
        )

# ---------- Main content (port of AnalysisResult.tsx) ----------



if uploaded_file is None:
    st.info(" Upload a brain MRI image in the sidebar to start the analysis.")
    st.stop()

# Load image
try:
    image = Image.open(uploaded_file).convert("RGB")
    # Validate image dimensions
    if image.size[0] < 32 or image.size[1] < 32:
        st.error("Image too small. Please upload a higher resolution MRI scan.")
        st.stop()
    
    # Verify image is not empty
    img_array_check = np.array(image)
    if img_array_check.size == 0 or np.max(img_array_check) == 0:
        st.error("Image appears to be empty or all black. Please upload a valid MRI scan.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading image: {e}")
    st.stop()

# Preprocess image for model (needed for GradCAM)
try:
    input_img = _preprocess_image(image, use_imagenet_norm=use_imagenet_preprocessing)    
    # Debug: verify image is not empty
    if np.max(input_img.numpy()) == 0:
        st.error(f"⚠️ Preprocessed image is all zeros! Original image size: {image.size}, mode: {image.mode}")
        st.stop()
    
    # Get prediction to determine which class to visualize
    input_img_device = input_img.to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(input_img_device)
        pred = F.softmax(outputs, dim=1).cpu().numpy()
    
    pred_label = np.argmax(pred, axis=1)[0] if pred.ndim > 1 else np.argmax(pred)
    
    # Generate GradCAM heatmap
    heatmap_img = make_gradcam_overlay(image, input_img, MODEL, pred_index=int(pred_label))
except Exception as e:
    st.error(f"Error during preprocessing or GradCAM: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Analyze image for results
result = analyze_image(image, use_imagenet_norm=use_imagenet_preprocessing)

if result is None:
    st.error("Failed to analyze image. Please try again.")
    st.stop()

# Top row: images
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Input MRI")
    st.caption("Original uploaded slice")
    st.image(image, width=350)

with col_right:
    st.subheader("AI Focus Areas (Grad-CAM)")
    st.caption("Gradient-weighted Class Activation Mapping visualization")
    st.image(heatmap_img,width=400)

st.write("---")

# Middle: prediction card
pred = result["prediction"]
class_name = pred["class_name"]
conf = pred["confidence"]

cols_pred = st.columns([2, 1])
with cols_pred[0]:
    st.markdown("##### Prediction Result")
    icon = "✅" if class_name == "No Tumor" else "⚠️"
    st.markdown(
        f"<div style='font-size:2rem;font-weight:700;'>{icon} {class_name}</div>",
        unsafe_allow_html=True,
    )
with cols_pred[1]:
    st.markdown("##### Confidence score")
    st.markdown(
        f"<div style='font-size:2rem;font-weight:700;text-align:right;'>{conf*100:.1f}%</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(conf, 0.0), 1.0))

st.write("---")

# Bottom: detailed analysis (3 columns)
with st.expander("Detailed analysis", expanded=True):
    col_ci, col_probs, col_rec = st.columns(3)

    # Clinical insights
    with col_ci:
        st.markdown("###### Clinical insights")
        for bullet in result["clinicalInsights"]:
            st.markdown(f"- {bullet}")

    # Probabilities
    with col_probs:
        st.markdown("###### Class probabilities")
        for p in pred["probabilities"]:
            cname = p["className"]
            val = p["probability"]
            st.markdown(f"**{cname}** – {val*100:.1f}%")
            st.progress(min(max(val, 0.0), 1.0))

    # Recommendations
    with col_rec:
        st.markdown("###### Recommendations")
        for rec in result["recommendations"]:
            st.markdown(f"- ✅ {rec}")

st.write("---")

# Download report
report_lines = [
    "NeuroSight MRI AI Report",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    "",
    f"Prediction: {class_name}",
    f"Confidence: {conf*100:.1f} %",
    "",
    "Class probabilities:",
]
for p in pred["probabilities"]:
    report_lines.append(f"  - {p['className']}: {p['probability']*100:.1f} %")
report_lines.append("")
report_lines.append("Clinical insights:")
for ci in result["clinicalInsights"]:
    report_lines.append(f"  - {ci}")
report_lines.append("")
report_lines.append("Recommendations:")
for r in result["recommendations"]:
    report_lines.append(f"  - {r}")

report_bytes = "\n".join(report_lines).encode("utf-8")

st.download_button(
    label="📄 Download text report",
    data=report_bytes,
    file_name="neurosight_report.txt",
    mime="text/plain",
)
