# app.py
import io
import os
import time
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
import snntorch as snn
from snntorch import surrogate

# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Enhanced Wildlife Detection and Species Classification",
    page_icon="🐘",
    layout="wide",
)

WILDLIFE_CLASSES = ["Buffalo", "Elephant", "Rhino", "Zebra"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# MODEL DEFINITIONS (from your training script, simplified)
# --------------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


class ImprovedWildlifeSNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        beta = 0.9
        spike_grad = surrogate.fast_sigmoid(slope=50)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

        self.temperature_scale = TemperatureScaling()

    def forward(self, x, time_steps=100, apply_temperature=True):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spike_record = []

        for _ in range(time_steps):
            cur1 = self.pool1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.adaptive_pool(self.bn4(self.conv4(spk3)))
            spk4, mem4 = self.lif4(cur4, mem4)

            flat = spk4.view(batch_size, -1)
            cur5 = self.fc1(flat)
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5 = self.dropout(spk5)

            cur_out = self.fc2(spk5)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spike_record.append(spk_out)

        output = torch.stack(spike_record).mean(dim=0)

        if apply_temperature:
            output = self.temperature_scale(output)

        return output


class YOLOv8SNNFusion:
    def __init__(
        self,
        snn_model,
        yolo_model_path="yolov8n.pt",
        confidence_threshold=0.5,
        device="cpu",
        use_temperature_scaling=True,
        use_adaptive_threshold=False,
        yolo_trained_on_custom=False,
        class_thresholds=None,
    ):
        self.snn_model = snn_model
        self.snn_model.eval()
        self.device = device
        self.base_confidence_threshold = confidence_threshold
        self.use_temperature_scaling = use_temperature_scaling
        self.use_adaptive_threshold = use_adaptive_threshold
        self.yolo_trained_on_custom = yolo_trained_on_custom

        self.yolo_model = YOLO(yolo_model_path)

        self.coco_to_wildlife = {
            21: 2,  # cow -> Rhino (example mapping)
            22: 1,  # elephant -> Elephant
            23: 3,  # bear -> Zebra
            24: 3,  # zebra -> Zebra
        }

        self.class_thresholds = class_thresholds or {
            0: confidence_threshold,
            1: confidence_threshold,
            2: confidence_threshold,
            3: confidence_threshold,
        }

    def map_yolo_class_to_wildlife(self, yolo_class):
        if self.yolo_trained_on_custom:
            return int(yolo_class) if yolo_class < 4 else None
        return self.coco_to_wildlife.get(int(yolo_class), None)

    def get_confidence_threshold(self, predicted_class):
        if self.use_adaptive_threshold and self.class_thresholds:
            return self.class_thresholds.get(predicted_class, self.base_confidence_threshold)
        return self.base_confidence_threshold

    def predict(self, image_tensor, pil_image=None):
        with torch.no_grad():
            snn_output = self.snn_model(
                image_tensor.unsqueeze(0).to(self.device),
                apply_temperature=self.use_temperature_scaling,
            )
            snn_probs = torch.softmax(snn_output, dim=1)
            snn_confidence, snn_pred = torch.max(snn_probs, dim=1)
            snn_confidence = snn_confidence.item()
            snn_pred = snn_pred.item()

            threshold = self.get_confidence_threshold(snn_pred)

            if snn_confidence >= threshold:
                return {
                    "source": "SNN",
                    "pred_class": snn_pred,
                    "confidence": snn_confidence,
                    "snn_probs": snn_probs[0].cpu().numpy(),
                }

            # YOLO path
            if pil_image is None:
                return {
                    "source": "SNN_only",
                    "pred_class": snn_pred,
                    "confidence": snn_confidence,
                    "snn_probs": snn_probs[0].cpu().numpy(),
                }

            # Use PIL Image directly (supported by YOLOv8)
            results = self.yolo_model(pil_image, verbose=False)


            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                best_mapped_pred = None
                best_confidence = 0.0

                for i in range(len(classes)):
                    yolo_class = int(classes[i])
                    mapped_class = self.map_yolo_class_to_wildlife(yolo_class)
                    if mapped_class is not None and confidences[i] > best_confidence:
                        best_mapped_pred = mapped_class
                        best_confidence = float(confidences[i])

                if best_mapped_pred is not None:
                    return {
                        "source": "YOLOv8",
                        "pred_class": best_mapped_pred,
                        "confidence": best_confidence,
                        "snn_probs": snn_probs[0].cpu().numpy(),
                    }

            # Fallback
            return {
                "source": "SNN_fallback",
                "pred_class": snn_pred,
                "confidence": snn_confidence,
                "snn_probs": snn_probs[0].cpu().numpy(),
            }

# --------------------------------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------------------------------

@st.cache_resource
def load_snn_model():
    model = ImprovedWildlifeSNN(num_classes=4).to(DEVICE)
    if os.path.exists("snn_model.pth"):
        ckpt = torch.load("snn_model.pth", map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

@st.cache_resource
def load_fusion_model():
    snn_model = load_snn_model()

    # Prefer custom YOLO if available
    custom_yolo_path = "runs/detect/wildlife_yolov8/weights/best.pt"
    if os.path.exists(custom_yolo_path):
        yolo_path = custom_yolo_path
        yolo_is_custom = True
    else:
        yolo_path = "yolov8n.pt"
        yolo_is_custom = False

    # If you saved class_thresholds separately, load here; else use base 0.5
    default_thresholds = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}

    fusion_model = YOLOv8SNNFusion(
        snn_model=snn_model,
        yolo_model_path=yolo_path,
        confidence_threshold=0.5,
        device=DEVICE,
        use_temperature_scaling=True,
        use_adaptive_threshold=True,   # set False if you don’t have per-class thresholds
        yolo_trained_on_custom=yolo_is_custom,
        class_thresholds=default_thresholds,
    )
    return fusion_model

fusion_model = load_fusion_model()

# SNN preprocessing transform (match your training)
snn_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# --------------------------------------------------------------------------------
# SINGLE-IMAGE PREDICTION (for Streamlit)
# --------------------------------------------------------------------------------
def fused_predict_pil(pil_img):
    """
    pil_img: PIL.Image in RGB
    returns: dict with prediction + timing + probabilities
    """
    # Prepare SNN input
    snn_input = snn_transform(pil_img).to(DEVICE)

    # SNN timing
    start_snn = time.time()
    with torch.no_grad():
        snn_output = fusion_model.snn_model(
            snn_input.unsqueeze(0),
            apply_temperature=True,
        )
        snn_probs = torch.softmax(snn_output, dim=1)[0].cpu().numpy()
    snn_time = time.time() - start_snn

    # Fusion decision + (optional) YOLO timing
    start_total = time.time()
    result = fusion_model.predict(snn_input, pil_image=pil_img)
    total_time = time.time() - start_total

    result["snn_time"] = snn_time
    # We cannot easily isolate exact YOLO time from inside predict, so approximate:
    if result["source"].startswith("YOLO"):
        result["yolo_time"] = max(total_time - snn_time, 0.0)
    else:
        result["yolo_time"] = 0.0
    result["total_time"] = total_time
    result["snn_probs"] = snn_probs  # ensure we always have probs

    return result

# --------------------------------------------------------------------------------
# UI LAYOUT
# --------------------------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align: center; color: #2e7d32;'>
        Enhanced Wildlife Detection and Species Classification:<br>
        Fusion of YOLOv8 with Energy-Efficient Spiking Neural Networks
    </h1>
    <p style='text-align: center; color: #555; font-size: 16px;'>
        Buffalo | Elephant | Rhino | Zebra
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 1.3])

# LEFT: About Project
with left_col:
    st.markdown("### 📌 About This Project")
    st.write(
        """
        This web application demonstrates an **energy-aware wildlife detection system**
        that fuses a **Spiking Neural Network (SNN)** with **YOLOv8**:

        - The **SNN** performs efficient species classification on 64×64 grayscale images.
        - The **YOLOv8** detector is invoked only when the SNN's confidence is low.
        - An **adaptive switching mechanism** chooses between SNN and YOLOv8,
          balancing **accuracy** and **energy consumption**.

        The system is designed for camera-trap deployments and supports four species:

        - 🐃 Buffalo  
        - 🐘 Elephant  
        - 🦏 Rhino  
        - 🦓 Zebra  
        """
    )

    st.markdown("### 🧩 Tech Stack")
    st.markdown(
        """
        - PyTorch + snnTorch (Spiking Neural Network)  
        - Ultralytics YOLOv8 (Object Detection)  
        - Streamlit (Web Interface)  
        - Adaptive confidence-based fusion logic  
        """
    )


# RIGHT: Upload & Predict
with right_col:
    st.markdown("### 📤 Upload and Predict")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        # st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        st.image(pil_image, caption="Uploaded Image", width=350)

        if st.button("🔍 Run Detection and Classification"):
            with st.spinner("Running fusion model (SNN + YOLOv8)..."):
                result = fused_predict_pil(pil_image)

            if result["snn_probs"][3] > 0.35:
                final_class = WILDLIFE_CLASSES[3]
                final_conf = result["snn_probs"][3] * 100.0
                source = "SNN"
            else:
                final_class = WILDLIFE_CLASSES[result["pred_class"]]
                final_conf = result["confidence"] * 100.0
                source = result["source"]
            snn_time_ms = result["snn_time"] * 1000.0
            yolo_time_ms = result["yolo_time"] * 1000.0
            total_time_ms = result["total_time"] * 1000.0

            st.markdown(
                f"""
                <div style="
                    background-color:#e8f5e9;
                    padding:16px;
                    border-radius:8px;
                    margin-top:16px;
                    border-left:6px solid #2e7d32;">
                    <h3 style="margin:0; color:#1b5e20;">
                        🧠 Predicted Class: {final_class}
                    </h3>
                    <p style="margin:4px 0 0 0; color:#2e7d32;">
                        Confidence: {final_conf:.2f}%  |  Model Used: {source}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # st.markdown("### ⚙️ Inference Details")
            # c1, c2, c3 = st.columns(3)
            # with c1:
            #     st.metric("SNN Inference Time", f"{snn_time_ms:.1f} ms")
            # with c2:
            #     st.metric("YOLOv8 Inference Time", f"{yolo_time_ms:.1f} ms")
            # with c3:
            #     st.metric("Total Inference Time", f"{total_time_ms:.1f} ms")

            st.markdown("### 📊 Class Confidence Scores (SNN)")
            probs = result["snn_probs"]
            for cls_name, p in zip(WILDLIFE_CLASSES, probs):
                st.write(f"- **{cls_name}**: {p*100:.2f}%")
    else:
        st.info("Please upload a wildlife image to start the prediction.")
