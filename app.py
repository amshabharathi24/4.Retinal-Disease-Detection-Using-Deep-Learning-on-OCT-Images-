import os
import json
import glob
import torch
import pandas as pd
import streamlit as st
from PIL import Image

from src.preprocessing import get_transforms
from src.model import build_model
from src.step6_gradcam import GradCAM, overlay_heatmap

st.set_page_config(
    page_title="Retinal OCT Disease Classifier",
    page_icon="👁️",
    layout="wide",
)

CLINICAL_INFO = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "summary": "Abnormal blood vessel growth beneath the retina causing fluid leakage and potential central vision distortion.",
        "visual_cue": "Subretinal hyperreflective material, fluid pockets, and RPE elevation.",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "summary": "Fluid accumulation and retinal thickening in the macula caused by leaking diabetic retinal capillaries.",
        "visual_cue": "Intraretinal cystoid spaces (dark fluid pockets) and increased foveal thickness.",
    },
    "DRUSEN": {
        "full_name": "Drusen (Early/Intermediate AMD)",
        "summary": "Yellow extracellular lipid/protein deposits accumulating between the retinal pigment epithelium (RPE) and Bruch's membrane.",
        "visual_cue": "Saw-tooth or dome-shaped bumps along the RPE layer without subretinal fluid.",
    },
    "NORMAL": {
        "full_name": "Normal Retina",
        "summary": "Healthy retinal architecture with intact foveal contour and well-defined, uninterrupted retinal layers.",
        "visual_cue": "Smooth foveal depression and continuous, parallel photoreceptor/RPE bands.",
    },
}


@st.cache_resource
def load_trained_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "mobilenet_v2")
    class_names = ckpt.get("class_names", ["CNV", "DME", "DRUSEN", "NORMAL"])

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, model_name, class_names, device


def load_evaluation_metrics():
    records = []
    for json_path in sorted(glob.glob(os.path.join("models", "eval_*.json"))):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = data.get("classification_report", {})
        macro = report.get("macro avg", {})
        records.append(
            {
                "Model": data.get("model_name", os.path.basename(json_path)),
                "Test Accuracy": round(data.get("test_accuracy", 0.0), 4),
                "Test Macro-F1": round(data.get("test_macro_f1", 0.0), 4),
                "Macro Precision": round(macro.get("precision", 0.0), 4),
                "Macro Recall": round(macro.get("recall", 0.0), 4),
            }
        )
    return pd.DataFrame(records)


st.title("👁️ Retinal OCT Disease Detection & Explainability")
st.caption("Deep Learning Clinical Decision-Support Prototype | Dataset: Kermany2017 OCT (CNV, DME, DRUSEN, NORMAL)")

st.warning(
    "⚠️ **Educational & Research Prototype Disclaimer:** This application is built strictly for academic "
    "and educational demonstration. It is **not** a certified medical device and must not be used for "
    "clinical diagnosis or patient treatment decisions."
)

# Sidebar: Model Selection & Pipeline Summary
st.sidebar.header("Model Configuration")
checkpoints = sorted(glob.glob(os.path.join("models", "best_*.pth")))

if not checkpoints:
    st.error("No trained checkpoints found in `models/`. Run `python -m src.step4_train` first.")
    st.stop()

selected_ckpt = st.sidebar.selectbox(
    "Select Trained Checkpoint",
    checkpoints,
    format_func=lambda p: os.path.basename(p),
)

model, model_name, class_names, device = load_trained_model(selected_ckpt)
st.sidebar.success(f"Loaded: `{model_name}` on `{device}`")
st.sidebar.markdown(
    """
    **Pipeline Highlights**
    - **Splits:** 86,647 Train / 21,662 Val / 1,000 Test
    - **Input Size:** 224 × 224 RGB (ImageNet Norm)
    - **Imbalance Fix:** Inverse-frequency class weights
    - **Explainability:** Grad-CAM on final conv block
    """
)

tab1, tab2 = st.tabs(["🔬 Single OCT Scan Diagnosis & Grad-CAM", "📊 Model Benchmark Comparison"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a Retinal OCT Image (PNG, JPG, JPEG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        tfms = get_transforms(img_size=224)["eval"]
        tensor_in = tfms(pil_img).unsqueeze(0).to(device)

        gradcam = GradCAM(model, model_name)
        heatmap, probs, pred_idx = gradcam.generate(tensor_in)
        overlay_img = overlay_heatmap(pil_img, heatmap, alpha=0.45)

        pred_label = class_names[pred_idx]
        confidence = probs[pred_idx] * 100.0

        col1, col2, col3 = st.columns([1, 1, 1.1])
        with col1:
            st.subheader("Original OCT Scan")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Attention Map")
            st.image(overlay_img, use_container_width=True)
            st.caption("Warm (red/yellow) regions highlight retinal layers driving the prediction.")
        with col3:
            st.subheader("Prediction Summary")
            st.metric(label="Predicted Condition", value=pred_label, delta=f"{confidence:.2f}% Confidence")

            prob_df = pd.DataFrame(
                {"Condition": class_names, "Probability (%)": [round(float(p) * 100, 2) for p in probs]}
            ).sort_values("Probability (%)", ascending=False)
            st.bar_chart(prob_df.set_index("Condition"))

            info = CLINICAL_INFO.get(pred_label, {})
            if info:
                st.info(
                    f"**{info['full_name']}**\n\n"
                    f"- **Overview:** {info['summary']}\n"
                    f"- **Key OCT Biomarker:** {info['visual_cue']}"
                )
    else:
        st.info("Upload an OCT retinal scan above (or test with `oct_samples.png` / any Kermany OCT image) to view class probabilities and Grad-CAM heatmaps.")

with tab2:
    st.subheader("Test Set Performance Across Trained Architectures (1,000 Images)")
    metrics_df = load_evaluation_metrics()
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.bar_chart(metrics_df.set_index("Model")[["Test Accuracy", "Test Macro-F1"]])
    else:
        st.warning("No evaluation JSON files (`models/eval_*.json`) found yet.")

    cm_files = sorted(glob.glob(os.path.join("models", "confusion_matrix_*.png")))
    if cm_files:
        st.subheader("Confusion Matrices")
        cols = st.columns(len(cm_files))
        for col, cm_path in zip(cols, cm_files):
            with col:
                st.image(cm_path, caption=os.path.basename(cm_path), use_container_width=True)