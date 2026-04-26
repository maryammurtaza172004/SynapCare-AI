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

# --- THE SWEET SPOT SETTINGS ---
CONFIDENCE_THRESHOLD = 0.12  # Balanced to avoid finger "ghost" fractures
SCALE_FACTOR = 0.2 

# 2. Upload Section
uploaded_file = st.file_uploader("Upload X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Analyze with SynapCare"):
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Plot visual boxes on image
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Detection Result', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI identified {len(results[0].boxes)} areas of interest:")
            
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist() 
                width_px = coords[2] - coords[0]
                height_px = coords[3] - coords[1]
                
                # Simple filter: Ignore very tiny 'noise' boxes
                if width_px < 5 and height_px < 5:
                    continue

                width_mm = width_px * SCALE_FACTOR
                height_mm = height_px * SCALE_FACTOR
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                
                # Bold details for the pitch
                st.markdown(f"### 📍 Detection: {name.upper()}")
                st.write(f"**Confidence:** {score:.1%}")
                st.write(f"**Extent:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                st.markdown("---")
        else:
            st.warning("No fractures detected. For faint fractures, try increasing image contrast.")

# 3. PROTOTYPE NOTICE
st.markdown("---")
st.caption("⚠️ **PROTOTYPE NOTICE:** SynapCare development prototype. All AI detections must be verified by a medical professional.")
