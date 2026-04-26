import streamlit as st
from ultralytics import YOLO
import PIL.Image

# 1. Page Config
st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Fracture Analysis & Measurement")

@st.cache_resource
def load_model():
    # Loading the trained brain
    return YOLO('best.pt')

model = load_model()

# --- SETTINGS ---
# Lowered to 0.08 to catch "missed" fractures
CONFIDENCE_THRESHOLD = 0.08 
# Standard pixel-to-mm ratio
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Run SynapCare Analysis"):
        # Running detection with higher sensitivity
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Plot visual boxes with labels and confidence %
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Detection Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI detected {len(results[0].boxes)} potential fracture area(s):")
            
            for box in results[0].boxes:
                # Calculate mm dimensions
                coords = box.xyxy[0].tolist() 
                width_mm = (coords[2] - coords[0]) * SCALE_FACTOR
                height_mm = (coords[3] - coords[1]) * SCALE_FACTOR
                
                # Get label and score
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                
                # Display individual box details
                with st.expander(f"Analysis: {name.upper()} ({score:.1%})"):
                    st.write(f"📏 **Estimated Extent:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                    st.progress(score)
        else:
            st.warning("No fractures detected. If you see one, try cropping the image closer to the bone.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** This application is a SynapCare development prototype. Measurements are estimates for research and educational purposes. Not for clinical diagnosis.")
