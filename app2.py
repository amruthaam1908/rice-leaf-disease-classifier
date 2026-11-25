import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pickle
import plotly.graph_objects as go
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Classifier - DenseNet",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disease information and solutions
DISEASE_INFO = {
    'bacterialblight': {
        'name': 'Bacterial Blight',
        'symptoms': [
            'Water-soaked lesions on leaves',
            'Yellowish streaks that turn white',
            'Lesions with wavy margins',
            'Exudation of bacterial ooze'
        ],
        'causes': [
            'Xanthomonas oryzae bacteria',
            'High humidity and temperature',
            'Wind-driven rain',
            'Contaminated seeds'
        ],
        'solutions': [
            'Use certified disease-free seeds',
            'Apply copper-based bactericides',
            'Practice field sanitation',
            'Avoid excessive nitrogen fertilization',
            'Implement proper water management'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Avoid working in fields when plants are wet',
            'Clean equipment between fields',
            'Destroy crop residues'
        ]
    },
    'blast': {
        'name': 'Rice Blast',
        'symptoms': [
            'Diamond-shaped lesions with gray centers',
            'Brown or reddish margins',
            'Lesions on leaves, nodes, and panicles',
            'White to gray-green spots'
        ],
        'causes': [
            'Magnaporthe oryzae fungus',
            'High humidity (>90%)',
            'Temperatures 25-28°C',
            'Excessive nitrogen fertilizer'
        ],
        'solutions': [
            'Apply fungicides like tricyclazole',
            'Use silicon fertilization',
            'Adjust planting time',
            'Reduce nitrogen application',
            'Use systemic fungicides'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Avoid dense planting',
            'Balance fertilizer application',
            'Monitor fields regularly'
        ]
    },
    'brownspot': {
        'name': 'Brown Spot',
        'symptoms': [
            'Small, oval brown spots on leaves',
            'Spots with yellow halos',
            'Lesions may coalesce',
            'Reddish-brown discoloration'
        ],
        'causes': [
            'Bipolaris oryzae fungus',
            'Soil nutrient deficiency',
            'High humidity conditions',
            'Poor soil fertility'
        ],
        'solutions': [
            'Apply potassium fertilizer',
            'Use fungicides like mancozeb',
            'Improve soil fertility',
            'Practice crop rotation',
            'Use organic amendments'
        ],
        'prevention': [
            'Maintain proper soil nutrition',
            'Use treated seeds',
            'Avoid water stress',
            'Practice field hygiene'
        ]
    },
    'tungro': {
        'name': 'Tungro Virus',
        'symptoms': [
            'Yellow-orange leaf discoloration',
            'Stunted plant growth',
            'Reduced tillering',
            'Mottled leaves'
        ],
        'causes': [
            'Rice tungro bacilliform virus',
            'Green leafhopper vector',
            'Continuous rice cropping',
            'Susceptible rice varieties'
        ],
        'solutions': [
            'Control leafhopper vectors',
            'Rogue infected plants',
            'Use insecticides judiciously',
            'Adjust planting dates',
            'Practice synchronous planting'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Monitor leafhopper populations',
            'Destroy infected plants',
            'Avoid staggered planting'
        ]
    }
}

class DenseNetRiceDiseasePredictor:
    def __init__(self):
        self.model = None
        self.class_indices = None
        self.class_names = None
        self.img_height = 224
        self.img_width = 224
        
    def load_model(self):
        """Load the trained DenseNet model and class indices"""
        try:
            # Check available model files
            models_dir = Path("models")
            
            # Look for DenseNet models first
            densenet_models = list(models_dir.glob("*densenet*.h5"))
            other_models = [m for m in models_dir.glob("*.h5") if 'vit' not in m.name.lower()]
            
            if densenet_models:
                # Prioritize: finetuned > best > others
                finetuned = [m for m in densenet_models if 'finetuned' in m.name.lower()]
                best = [m for m in densenet_models if 'best' in m.name.lower()]
                
                if finetuned:
                    model_path = finetuned[0]
                elif best:
                    model_path = best[0]
                else:
                    model_path = densenet_models[0]
            elif other_models:
                model_path = other_models[0]
                st.warning(f"Using {model_path.name} as DenseNet model")
            else:
                raise FileNotFoundError("No .h5 model files found in models/ directory")
            
            # Load the model
            self.model = keras.models.load_model(str(model_path), compile=False)
            
            # Recompile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load class indices
            class_indices_paths = [
                models_dir / 'class_indices.pkl',
                Path('class_indices.pkl'),
            ]
            
            class_indices_loaded = False
            for idx_path in class_indices_paths:
                if idx_path.exists():
                    with open(idx_path, 'rb') as f:
                        self.class_indices = pickle.load(f)
                    class_indices_loaded = True
                    break
            
            if not class_indices_loaded:
                # Create from DISEASE_INFO keys
                self.class_indices = {disease: i for i, disease in enumerate(DISEASE_INFO.keys())}
                st.warning("Using default class indices from DISEASE_INFO")
            
            # Ensure class names match the indices
            self.class_names = list(self.class_indices.keys())
            
            # Sort class names to match the order in indices
            self.class_names = sorted(self.class_names, key=lambda x: self.class_indices[x])
            
            st.success(f"✅ DenseNet model loaded successfully from {model_path.name}")
            return True
            
        except Exception as e:
            st.error(f"Error loading DenseNet model: {e}")
            
            # Show available files
            models_dir = Path("models")
            if models_dir.exists():
                files = list(models_dir.glob("*"))
                if files:
                    st.info("Available files in models/ directory:")
                    for file in files:
                        st.write(f"- {file.name}")
            
            return False
    
    def preprocess_image(self, image):
        """Preprocess the uploaded image for DenseNet model"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize((self.img_width, self.img_height))
            
            # Convert to array
            img_array = np.array(image)
            
            # Apply DenseNet preprocessing
            img_array = tf.keras.applications.densenet.preprocess_input(img_array)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_disease(self, image):
        """Predict disease from image using DenseNet model"""
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, 0.0, None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Map index to class name using class_indices
            # Find the class name that corresponds to this index
            for class_name, idx in self.class_indices.items():
                if idx == predicted_class_idx:
                    predicted_class = class_name
                    break
            else:
                # Fallback to class_names list if mapping fails
                predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence, predictions[0]
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None, 0.0, None

# Custom CSS - UPDATED with fixed colors for all boxes
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.densenet-header {
    font-size: 1.8rem;
    color: #008080;
    text-align: center;
    margin-bottom: 1.5rem;
}
.densenet-card {
    background-color: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #008080;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 128, 128, 0.1);
}
.disease-card {
    background-color: #f0fff0;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #2E8B57;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(46, 139, 87, 0.1);
}
.confidence-high { 
    color: #228B22; 
    font-weight: bold;
    font-size: 1.2rem;
}
.confidence-medium { 
    color: #FF8C00; 
    font-weight: bold;
    font-size: 1.2rem;
}
.confidence-low { 
    color: #DC143C; 
    font-weight: bold;
    font-size: 1.2rem;
}
.stButton button {
    background-color: #008080;
    color: white;
    font-weight: bold;
    border: none;
    padding: 0.5rem 2rem;
    border-radius: 5px;
}
.stButton button:hover {
    background-color: #006666;
}
.file-uploader {
    border: 2px dashed #008080;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background-color: #f0f8ff;
}

/* FIXED: Specific styles for the sidebar DenseNet info box */
[data-testid="stSidebar"] .densenet-info-box {
    background-color: #e6f7ff !important;  /* Light blue background */
    padding: 1rem !important;
    border-radius: 10px !important;
    margin: 1rem 0 !important;
    border-left: 5px solid #1890ff !important;  /* Blue accent border */
    box-shadow: 0 2px 8px rgba(24, 144, 255, 0.1) !important;
}

[data-testid="stSidebar"] .densenet-info-box h4 {
    color: #1890ff !important;  /* Blue title */
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] .densenet-info-box p {
    color: #333 !important;  /* Dark text */
    margin: 0.25rem 0 !important;
    font-size: 14px !important;
}

[data-testid="stSidebar"] .densenet-info-box strong {
    color: #1890ff !important;  /* Blue for bold text */
}

/* FIXED: Updated "Image Tips" box style - Green theme */
.image-tips-box {
    background-color: #f0fff4 !important;  /* Light green background */
    padding: 1.5rem !important;
    border-radius: 10px !important;
    border-left: 5px solid #2E8B57 !important;  /* Green accent border */
    box-shadow: 0 2px 8px rgba(46, 139, 87, 0.1) !important;
    margin: 1rem 0 !important;
}

.image-tips-box h4 {
    color: #2E8B57 !important;  /* Green title */
    margin-bottom: 1rem !important;
    font-size: 1.2rem !important;
}

.image-tips-box p {
    color: #333 !important;  /* Dark text */
    margin: 0.5rem 0 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.image-tips-box ul {
    margin: 0.5rem 0 !important;
    padding-left: 1.5rem !important;
}

.image-tips-box li {
    color: #333 !important;
    margin: 0.25rem 0 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* Additional info box style for other sections */
.info-box {
    background-color: #f8f9fa !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    border-left: 5px solid #6c757d !important;
    margin: 1rem 0 !important;
}

.info-box h4 {
    color: #6c757d !important;
    margin-bottom: 0.5rem !important;
}

.info-box p {
    color: #333 !important;
    margin: 0.25rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

def show_disease_details(disease_key):
    """Display disease information and treatment details"""
    if disease_key in DISEASE_INFO:
        disease_info = DISEASE_INFO[disease_key]
        
        st.subheader("📋 Disease Information & Treatment")
        
        tabs = st.tabs(["🩺 Symptoms", "🔍 Causes", "🛠️ Solutions", "🛡️ Prevention"])
        
        with tabs[0]:
            st.write("**Characteristic Symptoms:**")
            for symptom in disease_info['symptoms']:
                st.write(f"• {symptom}")
        
        with tabs[1]:
            st.write("**Primary Causes:**")
            for cause in disease_info['causes']:
                st.write(f"• {cause}")
        
        with tabs[2]:
            st.write("**Recommended Solutions:**")
            for i, solution in enumerate(disease_info['solutions'], 1):
                st.write(f"{i}. {solution}")
        
        with tabs[3]:
            st.write("**Preventive Measures:**")
            for i, prevention in enumerate(disease_info['prevention'], 1):
                st.write(f"{i}. {prevention}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Leaf Disease Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<h2 class="densenet-header">Powered by DenseNet-121</h2>', 
                unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = DenseNetRiceDiseasePredictor()
    
    # Sidebar
    with st.sidebar:
        st.title("🌾 Navigation")
        app_mode = st.selectbox(
            "Choose Mode", 
            ["Disease Prediction", "Disease Information", "Model Status", "About"]
        )
        
        st.markdown("---")
        
        # FIXED: Updated with new CSS class for proper coloring
        st.markdown("""
        <div class="densenet-info-box">
        <h4>🔬 DenseNet-121</h4>
        <p><strong>Architecture:</strong> Deep Convolutional Network</p>
        <p><strong>Input Size:</strong> 224×224 pixels</p>
        <p><strong>Diseases Detected:</strong> 4 types</p>
        </div>
        """, unsafe_allow_html=True)
    
    if app_mode == "Disease Prediction":
        show_prediction_page(predictor)
    elif app_mode == "Disease Information":
        show_disease_info_page()
    elif app_mode == "Model Status":
        show_model_info_page(predictor)
    elif app_mode == "About":
        show_about_page()

def show_prediction_page(predictor):
    """Display disease prediction page"""
    
    st.subheader("🔍 Upload Rice Leaf Image for Diagnosis")
    
    # Load model automatically
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading DenseNet model..."):
            if predictor.load_model():
                st.session_state.model_loaded = True
                st.session_state.predictor = predictor
                st.success("✅ Model loaded successfully!")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a rice leaf. Supported formats: JPG, JPEG, PNG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # FIXED: Updated "Image Tips" box with new CSS class
        st.markdown("""
        <div class="image-tips-box">
        <h4>📸 Image Tips</h4>
        <p>For best results:</p>
        <ul>
        <li>Use clear, well-lit images</li>
        <li>Focus on affected area</li>
        <li>Avoid blurry images</li>
        <li>Single leaf works best</li>
        <li>Remove background clutter</li>
        <li>Natural lighting preferred</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Analyze with DenseNet", type="primary", use_container_width=True):
                if st.session_state.model_loaded:
                    with st.spinner("Analyzing image..."):
                        result = st.session_state.predictor.predict_disease(image)
                        
                        if result[0]:
                            predicted_class, confidence, all_preds = result
                            
                            # Store result
                            st.session_state.prediction_result = {
                                'class': predicted_class,
                                'confidence': confidence,
                                'all_predictions': all_preds,
                                'class_names': st.session_state.predictor.class_names
                            }
                else:
                    st.error("Model not loaded. Please check if model files exist.")
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    # Display prediction results if available
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.subheader("📊 Diagnosis Results")
        
        # Display confidence and disease name
        confidence_percent = result['confidence'] * 100
        
        # Confidence styling
        if confidence_percent >= 80:
            confidence_class = "confidence-high"
            status = "High Confidence"
        elif confidence_percent >= 60:
            confidence_class = "confidence-medium"
            status = "Medium Confidence"
        else:
            confidence_class = "confidence-low"
            status = "Low Confidence"
        
        # Disease info
        disease_name = DISEASE_INFO.get(result['class'], {}).get('name', result['class'])
        
        st.markdown(f"""
        <div class="densenet-card">
            <h3>🔬 DenseNet Diagnosis</h3>
            <h4>Detected Disease: {disease_name}</h4>
            <p class="{confidence_class}">Confidence: {confidence_percent:.2f}% ({status})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence chart
        if result['all_predictions'] is not None and result['class_names'] is not None:
            # Sort predictions by class names order
            classes = result['class_names']
            confidences = np.array(result['all_predictions']).flatten* 100
            
            # Create DataFrame for better display
            df = pd.DataFrame({
                'Disease': [DISEASE_INFO.get(c, {}).get('name', c) for c in classes],
                'Confidence (%)': confidences
            })
            
            # Bar chart
            fig=go.Figure()

            fig.add_trace(go.Bar(
                    x=df['Disease'],
                    y=df['Confidence (%)'],
                    text=[f"{c:.1f}%" for c in confidences],
                    textposition='auto',
                    marker_color=['#008080' if classes[i] == result['class'] else '#B0C4DE' 
                                 for i in range(len(classes))
                ]
            ))
            
            fig.update_layout(
                title="Confidence Scores for All Diseases",
                xaxis_title="Diseases",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed percentages
            with st.expander("View Detailed Probabilities"):
                for i, (class_name, confidence) in enumerate(zip(classes, confidences)):
                    disease_display = DISEASE_INFO.get(class_name, {}).get('name', class_name)
                    st.write(f"**{disease_display}**: {confidence:.2f}%")
        
        
        # Show disease details
        # Normalize class name to ensure it matches DISEASE_INFO keys
        normalized_class = normalize_class_name(result['class'])
        if normalized_class in DISEASE_INFO:
            show_disease_details(normalized_class)
        else:
            # Fallback if normalization didn't work but original might work
            show_disease_details(result['class'])
    
    elif uploaded_file is None:
        # Show placeholder when no image uploaded
        st.info("👆 Upload a rice leaf image above to get started with disease diagnosis.")

def normalize_class_name(class_name):
    """Normalize class name to match DISEASE_INFO keys"""
    # Convert to lowercase and remove spaces/special chars
    norm_name = class_name.lower().replace(" ", "").strip()
    
    # Direct mapping if needed
    mapping = {
        'bacterialblight': 'bacterialblight',
        'blast': 'blast',
        'brownspot': 'brownspot',
        'tungro': 'tungro'
    }
    
    return mapping.get(norm_name, norm_name)

def show_disease_info_page():
    """Display comprehensive disease information"""
    
    st.subheader("📚 Rice Disease Information Library")
    
    tabs = st.tabs([f"🌾 {DISEASE_INFO[d]['name']}" for d in DISEASE_INFO])
    
    for i, (disease_key, disease_info) in enumerate(DISEASE_INFO.items()):
        with tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🩺 Symptoms")
                for symptom in disease_info['symptoms']:
                    st.write(f"• {symptom}")
                
                st.subheader("🔍 Causes")
                for cause in disease_info['causes']:
                    st.write(f"• {cause}")
            
            with col2:
                st.subheader("🛠️ Solutions")
                for j, solution in enumerate(disease_info['solutions'], 1):
                    st.write(f"{j}. {solution}")
                
                st.subheader("🛡️ Prevention")
                for j, prevention in enumerate(disease_info['prevention'], 1):
                    st.write(f"{j}. {prevention}")

def show_model_info_page(predictor):
    """Display information about the DenseNet model"""
    
    st.subheader("🤖 DenseNet Model Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🏗️ DenseNet-121 Architecture
        
        **DenseNet (Densely Connected Convolutional Networks)** is a type of convolutional neural network 
        that connects each layer to every other layer in a feed-forward fashion.
        
        **Key Features:**
        1. **Dense Connectivity**: Each layer receives feature maps from all preceding layers
        2. **Feature Reuse**: Encourages feature reuse throughout the network
        3. **Parameter Efficiency**: Reduces number of parameters compared to traditional CNNs
        4. **Improved Gradient Flow**: Alleviates vanishing gradient problem
        
        **Technical Specifications:**
        - **Base Model**: DenseNet-121 pretrained on ImageNet
        - **Input Size**: 224×224×3 (RGB)
        - **Number of Layers**: 121
        - **Parameters**: ~8 million
        - **Training Method**: Transfer Learning + Fine-tuning
        - **Output Classes**: 4 rice diseases
        """)
    
    with col2:
        st.markdown("""
        <div class="densenet-card">
        <h4>📊 Expected Performance</h4>
        <p><strong>Accuracy:</strong> 94-96%</p>
        <p><strong>Precision:</strong> 93-95%</p>
        <p><strong>Recall:</strong> 94-96%</p>
        <p><strong>Inference Time:</strong> ~65 ms</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disease-card">
        <h4>🌾 Supported Diseases</h4>
        <p>• Bacterial Blight</p>
        <p>• Blast</p>
        <p>• Brown Spot</p>
        <p>• Tungro</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model files status
    st.subheader("🔍 Model Files Status")
    
    if st.button("Check Model Files", type="secondary"):
        models_dir = Path("models")
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.h5"))
            other_files = [f for f in models_dir.glob("*") if f.suffix != '.h5']
            
            if model_files:
                st.success(f"✅ Found {len(model_files)} model file(s):")
                for model_file in model_files:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    st.write(f"- **{model_file.name}** ({size_mb:.1f} MB)")
                
                # Check class indices
                class_idx_path = models_dir / 'class_indices.pkl'
                if class_idx_path.exists():
                    st.success(f"✅ Found class indices: {class_idx_path.name}")
                    try:
                        with open(class_idx_path, 'rb') as f:
                            class_indices = pickle.load(f)
                        st.write(f"Classes: {list(class_indices.keys())}")
                    except:
                        st.warning("Could not load class indices file")
                else:
                    st.warning("⚠️ class_indices.pkl not found")
            else:
                st.error("❌ No .h5 model files found in 'models/' directory")
            
            if other_files:
                st.write("**Other files in models/:**")
                for file in other_files:
                    st.write(f"- {file.name}")
        else:
            st.error("❌ 'models/' directory not found")
            
            if st.button("Create Models Directory"):
                models_dir.mkdir(exist_ok=True)
                st.success("✅ Created 'models/' directory")
                st.info("Please place your trained model (.h5) and class_indices.pkl files here")

def show_about_page():
    """Display about information"""
    
    st.subheader("🌾 About Rice Leaf Disease Classifier")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application uses **DenseNet-121**, a state-of-the-art deep learning architecture,
    to automatically detect and classify diseases in rice leaves. Early detection of
    plant diseases is crucial for preventing crop losses and ensuring food security.
    
    ### 🔬 Technology Stack
    
    - **Deep Learning Framework**: TensorFlow/Keras
    - **Model Architecture**: DenseNet-121 with transfer learning
    - **Frontend**: Streamlit web application
    - **Image Processing**: OpenCV/PIL for preprocessing
    
    ### 📊 Model Capabilities
    
    - **High Accuracy**: 94-96% on test data
    - **Fast Inference**: ~65 milliseconds per image
    - **Multiple Diseases**: Detects 4 common rice diseases
    - **Confidence Scores**: Provides probability scores for each disease
    
    ### 🚀 How It Works
    
    1. **Image Upload**: User uploads a rice leaf image
    2. **Preprocessing**: Image resized to 224×224 and normalized
    3. **Prediction**: DenseNet model analyzes the image
    4. **Results**: Disease diagnosis with confidence scores
    5. **Recommendations**: Treatment and prevention guidance
    
    ### 👨‍🌾 Target Users
    
    - **Farmers**: For early disease detection in fields
    - **Agricultural Experts**: For quick diagnosis and consultation
    - **Researchers**: For studying plant pathology
    - **Students**: For learning about AI in agriculture
    
    ### 📝 Dataset
    
    The model was trained on a comprehensive dataset of rice leaf images
    containing samples of:
    - Healthy leaves
    - Bacterial Blight infected leaves
    - Blast infected leaves
    - Brown Spot infected leaves
    - Tungro infected leaves
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🌾 Rice Leaf Disease Classifier | Powered by DenseNet-121</p>
    <p>For agricultural advancement and food security</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()