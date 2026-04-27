import streamlit as st
from ultralytics import YOLO
import PIL.Image

# 1. Page Config
st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Multi-Bone Fracture Analysis")

@st.cache_resource
def load_model():
    # This loads your trained 'brain'
    return YOLO('best.pt')

model = load_model()

# --- BALANCED SETTINGS ---
CONFIDENCE_THRESHOLD = 0.12 
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload X-ray Image...", type=["jpg", "jpeg", "png"])

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
            st.success(f"AI detected {len(results[0].boxes)} fracture area(s):")
            
            for box in results[0].boxes:
                # DYNAMIC LABELING: This stops the 'everything is humerus' glitch
                class_id = int(box.cls[0])
                label_name = model.names[class_id] # Pulls the real name from your model
                
                # Measurement Logic
                coords = box.xyxy[0].tolist() 
                width_mm = (coords[2] - coords[0]) * SCALE_FACTOR
                height_mm = (coords[3] - coords[1]) * SCALE_FACTOR
                score = float(box.conf[0])
                
                # Display Results
                st.markdown(f"### 📍 Type: {label_name.upper()}")
                st.write(f"**Confidence:** {score:.1%}")
                st.write(f"**Extent of Fracture:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                st.markdown("---")
        else:
            st.warning("No fractures detected. If one is visible, try adjusting image brightness.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** SynapCare AI. Results are for research and educational purposes only.")
