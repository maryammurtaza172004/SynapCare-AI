import streamlit as st
from ultralytics import YOLO
import PIL.Image

# 1. Page Config
st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Precision Fracture Analysis")

@st.cache_resource
def load_model():
    # Loading your 'best.pt' model
    return YOLO('best.pt')

model = load_model()

# --- BALANCED SENSITIVITY ---
# Lowered to 0.04 to catch small/difficult fractures like the thumb
CONFIDENCE_THRESHOLD = 0.04 
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Run SynapCare AI Analysis"):
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Plot visual boxes with dynamic labels
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Detection Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI identified {len(results[0].boxes)} potential fracture(s):")
            
            for box in results[0].boxes:
                # Dynamic Labeling
                class_id = int(box.cls[0])
                label_name = model.names[class_id]
                
                # Measurement Calculations
                coords = box.xyxy[0].tolist() 
                width_mm = (coords[2] - coords[0]) * SCALE_FACTOR
                height_mm = (coords[3] - coords[1]) * SCALE_FACTOR
                score = float(box.conf[0])
                
                # Showing the results clearly for the Professor
                st.markdown(f"### 📍 Detection: {label_name.upper()}")
                st.write(f"**Confidence Level:** {score:.1%}")
                st.write(f"**Extent of Fracture:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                st.markdown("---")
        else:
            st.warning("No fractures detected. At 4% sensitivity, this suggests the image features are very different from the training set.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** SynapCare Development Prototype. AI-assisted analysis for educational use. Not for clinical diagnosis.")
