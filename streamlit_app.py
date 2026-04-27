import streamlit as st
from ultralytics import YOLO
import PIL.Image

# 1. Page Config
st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Precision Fracture Analysis")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- DYNAMIC CONTROLS ---
st.sidebar.header("Analysis Settings")
# This slider lets you fix the "8 fractures" problem instantly!
conf_input = st.sidebar.slider("Sensitivity (Confidence Threshold)", 0.01, 0.50, 0.15)
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Run SynapCare AI Analysis"):
        # We use the slider value here
        results = model.predict(image, conf=conf_input, augment=True)
        
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Detection Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI identified {len(results[0].boxes)} potential area(s):")
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label_name = model.names[class_id]
                score = float(box.conf[0])
                
                coords = box.xyxy[0].tolist() 
                width_mm = (coords[2] - coords[0]) * SCALE_FACTOR
                st.write(f"📍 **{label_name.upper()}** (Conf: {score:.1%}) - Extent: {width_mm:.1f} mm")
        else:
            st.warning("No fractures detected at this sensitivity level.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** SynapCare Development Prototype. Educational use only. Not for clinical diagnosis.")
