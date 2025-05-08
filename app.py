import streamlit as st
import os
import json
import pandas as pd
from models import predict_disease, load_model
from pharmacy_scraper import (
    search_medication_1mg,
    search_medication_pharmeasy,
    search_medication_netmeds,
    search_medication_cultfit
)

# Set page configuration
st.set_page_config(
    page_title="AI Disease Prediction & Treatment Recommendation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load model at startup to avoid reloading
@st.cache_resource
def get_model():
    return load_model()

# Load disease and symptom data
@st.cache_data
def load_data():
    with open("data/symptoms.json", "r") as f:
        symptoms_data = json.load(f)
    with open("data/diseases.json", "r") as f:
        diseases_data = json.load(f)
    return symptoms_data, diseases_data

# Initialize model and data
model = get_model()
symptoms_data, diseases_data = load_data()

# Custom CSS for dark theme
st.markdown("""
<style>
    .disclaimer-box {
        background-color: rgba(255, 87, 51, 0.1);
        border-left: 5px solid #ff5733;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .medication-box {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .confidence-high {
        color: #00cc96;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffa15a;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef553b;
        font-weight: bold;
    }
    .prevention-tips-box {
        background-color: rgba(0, 204, 150, 0.1);
        border-left: 5px solid #00cc96;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .lifestyle-box {
        background-color: rgba(100, 149, 237, 0.1);
        border-left: 5px solid #6495ED;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn.jsdelivr.net/npm/feather-icons/dist/icons/activity.svg", width=50)
st.sidebar.title("AI Health Assistant")

# Display disclaimer in sidebar
st.sidebar.markdown("""
<div class="disclaimer-box">
<p><strong>MEDICAL DISCLAIMER</strong></p>
<p>This application is intended for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
<p>Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

# Main content
st.title("🧠 AI-Powered Disease Prediction & Treatment Recommendation")
st.markdown("### Enter your symptoms for disease prediction and treatment recommendations")

# Symptom selection section
st.subheader("Symptom Selection")
col1, col2 = st.columns([2, 1])

with col1:
    # Dropdown for symptom selection
    selected_symptoms = st.multiselect(
        "Select your symptoms from the list:",
        options=symptoms_data["common_symptoms"],
        help="Choose all symptoms that apply to you"
    )

with col2:
    # Custom symptom input
    custom_symptom = st.text_input("Or enter a custom symptom:", 
                                    placeholder="e.g., skin rash, fatigue, etc.")
    
    if st.button("Add Custom Symptom") and custom_symptom and custom_symptom not in selected_symptoms:
        selected_symptoms.append(custom_symptom)

# Only show submit button if symptoms are selected
if selected_symptoms:
    st.write("Selected symptoms:", ", ".join(selected_symptoms))
    predict_btn = st.button("Predict Disease", type="primary")
else:
    predict_btn = False
    st.info("Please select at least one symptom to continue")

# Disease prediction section
if predict_btn and selected_symptoms:
    with st.spinner("Analyzing symptoms..."):
        # Join symptoms for prediction input
        symptoms_text = ", ".join(selected_symptoms)
        
        # Make prediction
        predictions = predict_disease(model, symptoms_text)
        
        st.subheader("Disease Prediction Results")
        
        # Display top 3 disease predictions with confidence
        for i, (disease, confidence) in enumerate(predictions[:3]):
            confidence_pct = confidence * 100
            if confidence_pct >= 70:
                confidence_class = "confidence-high"
            elif confidence_pct >= 40:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            # Create expander for each disease
            with st.expander(f"**{i+1}. {disease}** - Confidence: <span class='{confidence_class}'>{confidence_pct:.1f}%</span>", expanded=(i==0)):
                if disease in diseases_data:
                    disease_info = diseases_data[disease]
                    
                    # Display disease information
                    st.markdown("#### Disease Information")
                    st.markdown(f"**Description**: {disease_info.get('description', 'No description available')}")
                    
                    # Symptoms
                    st.markdown("#### Common Symptoms")
                    symptoms_list = disease_info.get('symptoms', [])
                    if symptoms_list:
                        for symptom in symptoms_list:
                            st.markdown(f"- {symptom}")
                    else:
                        st.write("No symptom information available")
                    
                    # Causes
                    st.markdown("#### Causes")
                    causes_list = disease_info.get('causes', [])
                    if causes_list:
                        for cause in causes_list:
                            st.markdown(f"- {cause}")
                    else:
                        st.write("No cause information available")
                    
                    # Prevention
                    st.markdown("#### Prevention")
                    prevention_list = disease_info.get('prevention', [])
                    if prevention_list:
                        for prevention in prevention_list:
                            st.markdown(f"- {prevention}")
                    else:
                        st.write("No prevention information available")
                    
                    # Detailed Prevention Tips
                    st.markdown("#### Detailed Prevention Tips")
                    prevention_tips = disease_info.get('prevention_tips', [])
                    if prevention_tips:
                        st.markdown('<div class="prevention-tips-box">', unsafe_allow_html=True)
                        for tip in prevention_tips:
                            st.markdown(f"- {tip}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write("No detailed prevention tips available")
                    
                    # Lifestyle Recommendations
                    st.markdown("#### Lifestyle Recommendations")
                    lifestyle_recommendations = disease_info.get('lifestyle_recommendations', [])
                    if lifestyle_recommendations:
                        st.markdown('<div class="lifestyle-box">', unsafe_allow_html=True)
                        for recommendation in lifestyle_recommendations:
                            st.markdown(f"- {recommendation}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write("No lifestyle recommendations available")
                    
                    # Treatment recommendations
                    st.markdown("#### Treatment Recommendations")
                    treatment_list = disease_info.get('treatments', [])
                    if treatment_list:
                        for treatment in treatment_list:
                            st.markdown(f"- {treatment}")
                    else:
                        st.write("No treatment recommendations available")
                    
                    # Medications section
                    st.markdown("#### Recommended Medications")
                    medications_list = disease_info.get('medications', [])
                    
                    if medications_list:
                        # Fetch medication purchase links for each medication
                        for medication in medications_list:
                            with st.spinner(f"Fetching information for {medication}..."):
                                # Create a box for each medication
                                st.markdown(f"""
                                <div class="medication-box">
                                <h5>{medication}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create columns for different pharmacy options
                                pharmacy_col1, pharmacy_col2 = st.columns(2)
                                
                                with pharmacy_col1:
                                    # 1mg
                                    link_1mg = search_medication_1mg(medication)
                                    if link_1mg:
                                        st.markdown(f"[Buy on 1mg]({link_1mg})")
                                    else:
                                        st.write("Not available on 1mg")
                                    
                                    # PharmEasy
                                    link_pharmeasy = search_medication_pharmeasy(medication)
                                    if link_pharmeasy:
                                        st.markdown(f"[Buy on PharmEasy]({link_pharmeasy})")
                                    else:
                                        st.write("Not available on PharmEasy")
                                
                                with pharmacy_col2:
                                    # Netmeds
                                    link_netmeds = search_medication_netmeds(medication)
                                    if link_netmeds:
                                        st.markdown(f"[Buy on Netmeds]({link_netmeds})")
                                    else:
                                        st.write("Not available on Netmeds")
                                    
                                    # CultFit
                                    link_cultfit = search_medication_cultfit(medication)
                                    if link_cultfit:
                                        st.markdown(f"[Buy on Cult.fit]({link_cultfit})")
                                    else:
                                        st.write("Not available on Cult.fit")
                    else:
                        st.write("No medication information available")
                else:
                    st.write("Detailed information for this disease is not available.")
        
        # Disclaimer reminder
        st.markdown("""
        <div class="disclaimer-box">
        <p><strong>IMPORTANT:</strong> The predictions provided are based on the symptoms you entered and should not be considered as a definitive diagnosis. 
        Please consult with a healthcare professional for proper diagnosis and treatment.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("#### 🧠 AI-Powered Disease Prediction & Treatment Recommendation Application")
st.markdown("This application uses machine learning to predict diseases based on symptoms and provide treatment recommendations.")
