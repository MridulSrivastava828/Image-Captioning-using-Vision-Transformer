import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

st.set_page_config(page_title="Image Captioning Studio", page_icon="🖼️", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    }
    .main-header {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .sub-title {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    .info-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    .result-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(16, 185, 129, 0.16));
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 14px;
        padding: 1rem;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='main-header'><div class='main-title'>🧠 Vision Transformer Caption Studio</div><div class='sub-title'>Upload an image and generate a concise, AI-powered caption in seconds.</div></div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, feature_extractor, tokenizer, device


def generate_caption(image, model, feature_extractor, tokenizer, device):
    image = image.convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()


model, feature_extractor, tokenizer, device = load_model()
model_name = "nlpconnect/vit-gpt2-image-captioning"

with st.sidebar:
    st.markdown("### ⚙️ System Overview")
    st.markdown(f"- Model: {model_name}")
    st.markdown(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.markdown("- Engine: Vision Encoder-Decoder")
    st.markdown("- Status: Ready")

left_column, right_column = st.columns([1.2, 0.9], gap="large")

with left_column:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="visible")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_column:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### ✨ Generated Caption")
    if uploaded_file is not None:
        with st.spinner("Analyzing image and generating caption..."):
            caption = generate_caption(image, model, feature_extractor, tokenizer, device)
        st.markdown(f"<div class='result-box'>{caption}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload an image to begin caption generation.")
    st.markdown("</div>", unsafe_allow_html=True)