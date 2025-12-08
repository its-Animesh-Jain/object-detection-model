import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import time
from collections import Counter
from pathlib import Path
from advanced_ensemble import AdvancedEnsemble

# --- UI/UX IMPROVEMENT: Automatically find the project's root directory ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- UI/UX IMPROVEMENT: Page Configuration for a professional look ---
st.set_page_config(
    page_title="VisionAI Dashboard",
    page_icon="🤖", # This is the robot icon in the browser tab!
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI/UX IMPROVEMENT: Inject Custom CSS for styling ---
def load_css():
    css = """
    <style>
        /* Main title */
        .st-emotion-cache-183lzff {
            padding-bottom: 0.5rem;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #000000;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #1f77b4;
        }
        
        /* --- Metric Card Styling --- */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        
        /* This forces the text inside the metric card to be dark */
        [data-testid="stMetric"] div, [data-testid="stMetric"] span {
             color: #212529 !important; /* Dark text color */
        }
        
        /* Expander styling */
        .st-emotion-cache-p5msec {
             border-color: #1f77b4;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Use Streamlit's cache to load the model only once.
@st.cache_resource
def initialize_ensemble():
    """Initialize the ensemble model using corrected paths"""
    MODELS_DIR = PROJECT_ROOT / "models"
    configs = {
        'yolov3': (str(MODELS_DIR / 'yolov3.cfg'), str(MODELS_DIR / 'yolov3.weights')),
        'yolov4-tiny': (str(MODELS_DIR / 'yolov4-tiny.cfg'), str(MODELS_DIR / 'yolov4-tiny.weights'))
    }
    classes_path = str(MODELS_DIR / 'coco.names')
    try:
        ensemble = AdvancedEnsemble(configs, classes_path=classes_path)
        return ensemble
    except Exception as e:
        st.error(f"Fatal Error Loading Models: {e}.")
        return None

# --- UI/UX IMPROVEMENT: New function for real-time analytics display ---
def display_current_analytics(boxes, scores, labels, processing_time, ensemble):
    """Displays a real-time analytics summary for the current detection."""
    if not boxes:
        st.info("No objects were detected, so no analytics are available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Objects Detected", len(boxes))
        avg_confidence = np.mean(scores) if scores else 0
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
        st.metric("Processing Time", f"{processing_time:.3f} s")
    with col2:
        class_ids = [label for label in labels]
        object_counts = Counter(ensemble.classes[id] for id in class_ids)
        if object_counts:
            df = pd.DataFrame(object_counts.items(), columns=['Object', 'Count'])
            df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
            st.write("**Object Breakdown:**")
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No objects to break down.")

def main():
    # --- UI/UX IMPROVEMENT: Load custom styling and professional header ---
    load_css()
    
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        # --- THE CHANGE IS HERE ---
        # Removed the Google logo and replaced with a generic robot emoji
        st.markdown("<p style='font-size: 70px; text-align: center;'>🤖</p>", unsafe_allow_html=True) 
    with col_title:
        st.title("VisionAI Detection Dashboard")
        st.markdown("*An advanced real-time object analysis platform*")
    st.markdown("---")
    
    # Initialize model
    if 'ensemble' not in st.session_state:
        with st.spinner("Initializing VisionAI models... Please wait."):
            st.session_state.ensemble = initialize_ensemble()
    
    if st.session_state.ensemble is None:
        st.error("Model loading failed. The application cannot proceed.")
        return
    
    # --- UI/UX IMPROVEMENT: Restructured Sidebar ---
    st.sidebar.header("📁 Input Source")
    input_option = st.sidebar.radio(
        "Choose an input method:",
        ["Select Image", "Upload Image", "Webcam"],
        captions=["From test data", "From your device", "From live camera"]
    )
    
    st.sidebar.header("⚙️ Detection Controls")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence", 0.1, 0.9, 0.5, 0.05,
        help="The minimum confidence required for an object to be detected."
    )
    
    # --- UI/UX IMPROVEMENT: Historical Analytics in an Expander ---
    with st.sidebar.expander("📊 View Historical Analytics"):
        st.info("This shows analytics for all detections in this session.")
        if st.button("Generate Session Report"):
            show_analytics()
        if st.button("Clear Session History"):
            if hasattr(st.session_state.ensemble, 'analytics'):
                st.session_state.ensemble.analytics.detection_history.clear()
            st.success("History cleared!")
    
    # --- Main content area ---
    input_col, output_col = st.columns([2, 3])
    
    image_to_process = None
    with input_col:
        st.subheader("Input Image")
        if input_option == "Select Image":
            image_to_process = handle_image_select()
        elif input_option == "Upload Image":
            image_to_process = handle_image_upload()
        elif input_option == "Webcam":
            image_to_process = handle_webcam()

    if image_to_process is not None:
        process_and_display(image_to_process, output_col, confidence_threshold)

def handle_image_select():
    image_dir = PROJECT_ROOT / "data"
    if not image_dir.is_dir():
        st.error(f"The 'data' directory was not found.")
        return None
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        st.warning("No images found in the 'data' folder.")
        return None

    selected_image_name = st.selectbox("Choose an image:", ["-"] + sorted(image_files))
    if selected_image_name != "-":
        image_path = image_dir / selected_image_name
        image = cv2.imread(str(image_path))
        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Selected: {selected_image_name}", use_container_width=True)
            return image
    return None

def handle_image_upload():
    uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            return image
    return None

def handle_webcam():
    camera_image = st.camera_input("Take a picture with your webcam")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Webcam Capture", use_container_width=True)
            return image
    return None

def process_and_display(image, output_col, confidence_threshold):
    with output_col:
        st.subheader("Output Analysis")
        
        # --- UI/UX IMPROVEMENT: Run detection with a low threshold to get all potential objects ---
        with st.spinner("🤖 Running AI inference..."):
            start_time = time.time()
            raw_boxes, raw_scores, raw_labels = st.session_state.ensemble.ensemble_detect(image, 0.1) # Low internal threshold
            processing_time = time.time() - start_time
        
        # --- UI/UX IMPROVEMENT: Interactive Filtering Slider ---
        if raw_boxes:
            filter_threshold = st.slider(
                "Filter displayed boxes by confidence", 0.1, 1.0, confidence_threshold, 0.05
            )
            
            # Filter the raw results based on the new slider
            filtered_indices = [i for i, score in enumerate(raw_scores) if score >= filter_threshold]
            boxes = [raw_boxes[i] for i in filtered_indices]
            scores = [raw_scores[i] for i in filtered_indices]
            labels = [raw_labels[i] for i in filtered_indices]
        else:
            boxes, scores, labels = [], [], []

        # --- UI/UX IMPROVEMENT: Using Tabs for Cleaner Output ---
        tab1, tab2, tab3 = st.tabs(["🖼️ Detection Result", "📋 Detection Summary", "📊 Live Analytics"])

        with tab1:
            result_image = st.session_state.ensemble.visualize_detections(image.copy(), boxes, scores, labels)
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption=f"Processed in {processing_time:.2f}s", use_container_width=True)

        with tab2:
            if boxes:
                st.success(f"✅ Detected {len(boxes)} objects with confidence ≥ {filter_threshold:.0%}")
                results_data = []
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    class_name = st.session_state.ensemble.classes[label]
                    x, y, w, h = box
                    results_data.append({"Object": class_name, "Confidence": f"{score:.2%}", "Position": f"[{x}, {y}, {w}, {h}]"})
                
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            else:
                st.warning("⚠️ No objects detected with current confidence settings.")

        with tab3:
            display_current_analytics(boxes, scores, labels, processing_time, st.session_state.ensemble)

def show_analytics():
    try:
        if not hasattr(st.session_state.ensemble, 'analytics'):
            st.error("Analytics component not found.")
            return
        report = st.session_state.ensemble.analytics.generate_performance_report()
        
        if not report or report.get('status') == 'No data available':
            st.info("No detection data recorded yet in this session.")
            return
        
        st.subheader("Session Performance Summary")
        col1, col2 = st.columns(2)
        col1.metric("Total Objects Detected", report.get('total_detections', 0))
        col2.metric("Avg. Processing Time", f"{report.get('avg_processing_time', 0):.2f}s")
        
        common_objects = report.get('most_common_objects', {})
        if common_objects:
            st.write("**Most Common Objects in Session:**")
            common_df = pd.DataFrame(list(common_objects.items()), columns=['Object', 'Count'])
            st.dataframe(common_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

if __name__ == "__main__":
    main()