import streamlit as st
from ultralytics import YOLO
import PIL.Image

# 1. Page Config
st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Fracture Analysis & Measurement")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 0.08 
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Run SynapCare Analysis"):
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Plot visual boxes WITH labels and confidence directly on the image
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Detection Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI detected {len(results[0].boxes)} potential fracture(s):")
            
            # This loop prints the details for EVERY box detected
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist() 
                width_mm = (coords[2] - coords[0]) * SCALE_FACTOR
                height_mm = (coords[3] - coords[1]) * SCALE_FACTOR
                
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                
                # --- SHOWING THE INFO CLEARLY ---
                st.markdown(f"### 📍 Detection: {name.upper()}")
                st.write(f"**Confidence Level:** {score:.1%}")
                st.write(f"**Extent of Fracture:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                st.markdown("---")
        else:
            st.warning("No fractures detected. Try cropping the image closer if you see a missed fracture.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** SynapCare development prototype. Measurements are estimates for research and educational purposes.")
