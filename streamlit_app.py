import streamlit as st
from ultralytics import YOLO
import PIL.Image

st.set_page_config(page_title="SynapCare AI", layout="centered")
st.title("🦴 SynapCare: Fracture Analysis (with Measurement)")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- SCALE FACTOR ---
# On a standard X-ray, 1 pixel is roughly 0.2mm. 
# You can adjust this number to be more or less precise!
SCALE_FACTOR = 0.2 

uploaded_file = st.file_uploader("Upload X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button("Analyze Extent & Size"):
        results = model(image, conf=0.15)
        
        # Plot visual results
        res_plotted = results[0].plot(labels=True, conf=True)
        res_image = PIL.Image.fromarray(res_plotted[:, :, ::-1])
        st.image(res_image, caption='AI Analysis', use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"AI detected potential fracture(s):")
            
            for box in results[0].boxes:
                # 1. Get dimensions in pixels
                # x1, y1 = top left | x2, y2 = bottom right
                coords = box.xyxy[0].tolist() 
                width_px = coords[2] - coords[0]
                height_px = coords[3] - coords[1]
                
                # 2. Convert to mm
                width_mm = width_px * SCALE_FACTOR
                height_mm = height_px * SCALE_FACTOR
                
                # 3. Get labels and confidence
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                
                # 4. Display the Detailed Analysis
                st.info(f"📍 **Type:** {name.upper()}")
                st.write(f"📈 **Confidence:** {score:.1%}")
                st.write(f"📏 **Estimated Extent:** {width_mm:.1f} mm x {height_mm:.1f} mm")
                st.markdown("---")
        else:
            st.warning("No fractures detected.")

st.caption("⚠️ **PROTOTYPE NOTICE:** Measurements are estimated based on standard scale factors. SynapCare development prototype only.")