import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np

st.set_page_config(page_title="SynapCare AI: Advanced Diagnostics", layout="centered")
st.title("🦴 SynapCare: Advanced Fracture Detection")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- ADVANCED DEBUGGING SETTINGS ---
CONFIDENCE_THRESHOLD = 0.02 # Dropped to 2% for absolute maximum sensitivity
SCALE_FACTOR = 0.2 

uploaded_file = st.file_uploader("Upload Thumb or Bone X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB") # Force RGB to match training
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Run Deep Analysis"):
        with st.spinner('Analyzing bone density and continuity...'):
            # Using 'augment=True' (TTA) - This helps catch 'invisible' thumb fractures
            results = model.predict(image, conf=CONFIDENCE_THRESHOLD, augment=True)
        
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='Deep Analysis Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI found {len(results[0].boxes)} potential area(s).")
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label_name = model.names[class_id]
                score = float(box.conf[0])
                st.write(f"📍 **{label_name.upper()}** - Confidence: {score:.1%}")
        else:
            st.error("Still not detecting. Recommendation: Use a higher-resolution 'Standard AP View' X-ray.")

st.markdown("---")
st.caption("Technical Note: TTA (Test-Time Augmentation) enabled for difficult fracture detection.")
