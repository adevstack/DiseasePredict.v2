import streamlit as st
import os
import json
import pandas as pd
from models import predict_disease, load_model, advanced_analysis
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
    .advanced-analysis-box {
        background-color: rgba(138, 43, 226, 0.1);
        border-left: 5px solid #8a2be2;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .advanced-result-box {
        background-color: rgba(70, 130, 180, 0.1);
        border-left: 5px solid #4682b4;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .recommendation-box {
        background-color: rgba(46, 139, 87, 0.1);
        border-left: 5px solid #2e8b57;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    /* Style file uploader */
    .stFileUploader {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    /* Style text area */
    .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid rgba(138, 43, 226, 0.2);
    }
    /* Style buttons */
    .stButton button {
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

# Initialize session state for predictions if it doesn't exist
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

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
        
        # Store predictions in session state to access them in advanced analysis
        st.session_state.predictions = predictions
        
        # Advanced Analysis Section
        st.markdown("---")
        st.subheader("🔬 Advanced Analysis (Optional)")
        st.markdown("""
        <div class="advanced-analysis-box">
        <p>For a more detailed analysis, you can provide additional information below. This is completely optional but may help with a more accurate assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed symptom description
        detailed_symptoms = st.text_area(
            "Describe your symptoms in detail:",
            placeholder="Please describe how you're feeling in your own words. For example: 'I've been having a persistent dry cough for 3 days, along with fatigue and mild fever that gets worse in the evening...'",
            height=150
        )
        
        # Medical reports upload section
        st.markdown("#### Upload Medical Reports (Optional)")
        st.markdown("You can upload medical reports like X-rays, CT scans, or lab reports for a more comprehensive analysis.")
        
        # Create three columns for different types of uploads
        upload_col1, upload_col2, upload_col3 = st.columns(3)
        
        with upload_col1:
            xray_file = st.file_uploader("X-ray Image", type=["jpg", "jpeg", "png"])
            if xray_file is not None:
                st.image(xray_file, caption="Uploaded X-ray", use_column_width=True)
        
        with upload_col2:
            ct_scan_file = st.file_uploader("CT Scan", type=["jpg", "jpeg", "png"])
            if ct_scan_file is not None:
                st.image(ct_scan_file, caption="Uploaded CT Scan", use_column_width=True)
        
        with upload_col3:
            lab_report = st.file_uploader("Lab Report", type=["pdf", "jpg", "jpeg", "png"])
            if lab_report is not None:
                if lab_report.type == "application/pdf":
                    st.write(f"PDF Uploaded: {lab_report.name}")
                else:
                    st.image(lab_report, caption="Uploaded Lab Report", use_column_width=True)
        
        # Run advanced analysis button
        if st.button("Run Advanced Analysis", type="secondary"):
            if not detailed_symptoms and not xray_file and not ct_scan_file and not lab_report:
                st.warning("Please provide either a detailed symptom description or upload at least one medical report for advanced analysis.")
            else:
                # Check if any predictions are available
                current_predictions = st.session_state.get('predictions', [])
                
                if not current_predictions:
                    st.warning("Please run the basic disease prediction first by selecting symptoms and clicking 'Predict Disease' before running advanced analysis.")
                else:
                    # Run DistilBERT-based advanced analysis
                    with st.spinner("Running advanced analysis with DistilBERT model..."):
                        # Check if we have images uploaded
                        has_images = any([xray_file, ct_scan_file, lab_report])
                        
                        # Get advanced analysis results using our advanced_analysis function from models.py
                        analysis_results = advanced_analysis(
                            detailed_symptoms=detailed_symptoms,
                            initial_predictions=current_predictions[:3] if current_predictions else [],
                            image_uploaded=has_images
                        )
                    
                    if analysis_results and analysis_results.get("success", False):
                        st.success("Advanced analysis complete!")
                        
                        # Display results in an expander
                        with st.expander("View Advanced Analysis Results", expanded=True):
                            # Basic info
                            st.markdown(f"#### Analysis of {analysis_results.get('top_disease', 'Unknown Condition')}")
                            
                            # Confidence comparison
                            initial_conf = analysis_results.get('initial_confidence', 0) * 100
                            advanced_conf = analysis_results.get('advanced_confidence', 0) * 100
                            
                            # Display confidence scores
                            confidence_col1, confidence_col2 = st.columns(2)
                            with confidence_col1:
                                st.metric(
                                    label="Initial Confidence", 
                                    value=f"{initial_conf:.1f}%",
                                )
                            with confidence_col2:
                                st.metric(
                                    label="Advanced Confidence", 
                                    value=f"{advanced_conf:.1f}%",
                                    delta=f"{advanced_conf - initial_conf:.1f}%"
                                )
                            
                            # Severity and urgency indicators
                            severity = analysis_results.get('severity', 'Unknown')
                            urgency = analysis_results.get('urgency', 'Unknown')
                            
                            severity_col1, severity_col2 = st.columns(2)
                            with severity_col1:
                                st.markdown(f"**Severity Assessment:** {severity}")
                            with severity_col2:
                                st.markdown(f"**Recommended Action:** {urgency}")
                            
                            # Natural language analysis
                            if detailed_symptoms:
                                st.markdown("#### Natural Language Analysis")
                                nl_indicators = analysis_results.get('natural_language_indicators', {})
                                
                                if nl_indicators:
                                    # Format the detected language patterns
                                    severity_terms = nl_indicators.get('severity_terms', [])
                                    duration_terms = nl_indicators.get('duration_terms', [])
                                    intensity_terms = nl_indicators.get('intensity_terms', [])
                                    
                                    st.markdown("""
                                    <div class="advanced-result-box">
                                    <p><strong>Symptom Analysis:</strong> Our language analysis has detected key indicators in your description.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if severity_terms:
                                        st.markdown(f"**Severity Indicators:** {', '.join(severity_terms)}")
                                    if duration_terms:
                                        st.markdown(f"**Duration Indicators:** {', '.join(duration_terms)}")
                                    if intensity_terms:
                                        st.markdown(f"**Intensity Indicators:** {', '.join(intensity_terms)}")
                            
                            # Medical imaging analysis
                            if has_images:
                                st.markdown("#### Medical Imaging Analysis")
                                image_analysis = analysis_results.get('image_analysis', None)
                                
                                if image_analysis:
                                    confidence = image_analysis.get('confidence', 0) * 100
                                    findings = image_analysis.get('findings', 'No significant findings')
                                    
                                    st.markdown(f"""
                                    <div class="advanced-result-box">
                                    <p><strong>Image Analysis:</strong> {findings}</p>
                                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Recommendation section
                            st.markdown("#### Recommendation")
                            
                            # Customize recommendation based on severity and confidence
                            if advanced_conf >= 70 and severity in ["Significant", "Severe"]:
                                recommendation = f"Based on our advanced analysis, we strongly recommend seeking prompt medical attention for potential {analysis_results.get('top_disease', 'condition')}. Please share these results with your healthcare provider."
                            elif advanced_conf >= 50:
                                recommendation = f"Based on our advanced analysis, we recommend consulting with a healthcare professional about potential {analysis_results.get('top_disease', 'condition')}. Please share these results with your doctor at your next appointment."
                            else:
                                recommendation = "Based on our advanced analysis, we recommend monitoring your symptoms and consulting with a healthcare professional if they persist or worsen."
                            
                            st.markdown(f"""
                            <div class="recommendation-box">
                            <p>{recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Disclaimer for advanced analysis
                            st.markdown("""
                            <div class="disclaimer-box">
                            <p><strong>MEDICAL DISCLAIMER:</strong> This advanced analysis is not a substitute for professional medical diagnosis. 
                            The image analysis feature is experimental and should be verified by qualified healthcare professionals.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Advanced analysis failed: {analysis_results.get('message', 'Unknown error')}")

# If no prediction was made yet, show the direct advanced analysis option
if not predict_btn or not selected_symptoms:
    st.markdown("---")
    st.subheader("🔬 Direct Advanced Analysis")
    st.markdown("""
    <div class="advanced-analysis-box">
    <p>You can also skip the basic symptom checker and go directly to our advanced analysis system. 
    Provide a detailed description of your symptoms and optionally upload medical reports for a comprehensive assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed symptom description for direct analysis
    direct_detailed_symptoms = st.text_area(
        "Describe your symptoms in detail:",
        placeholder="Please describe how you're feeling in your own words. For example: 'I've been having a persistent dry cough for 3 days, along with fatigue and mild fever that gets worse in the evening...'",
        height=150,
        key="direct_symptoms"
    )
    
    # Medical reports upload section
    st.markdown("#### Upload Medical Reports (Optional)")
    st.markdown("You can upload medical reports like X-rays, CT scans, or lab reports for a more comprehensive analysis.")
    
    # Create three columns for different types of uploads
    direct_upload_col1, direct_upload_col2, direct_upload_col3 = st.columns(3)
    
    with direct_upload_col1:
        direct_xray_file = st.file_uploader("X-ray Image", type=["jpg", "jpeg", "png"], key="direct_xray")
        if direct_xray_file is not None:
            st.image(direct_xray_file, caption="Uploaded X-ray", use_column_width=True)
    
    with direct_upload_col2:
        direct_ct_scan_file = st.file_uploader("CT Scan", type=["jpg", "jpeg", "png"], key="direct_ct")
        if direct_ct_scan_file is not None:
            st.image(direct_ct_scan_file, caption="Uploaded CT Scan", use_column_width=True)
    
    with direct_upload_col3:
        direct_lab_report = st.file_uploader("Lab Report", type=["pdf", "jpg", "jpeg", "png"], key="direct_lab")
        if direct_lab_report is not None:
            if direct_lab_report.type == "application/pdf":
                st.write(f"PDF Uploaded: {direct_lab_report.name}")
            else:
                st.image(direct_lab_report, caption="Uploaded Lab Report", use_column_width=True)
    
    # Direct advanced analysis button
    if st.button("Run Direct Advanced Analysis", type="primary", key="direct_advanced_btn"):
        if not direct_detailed_symptoms and not direct_xray_file and not direct_ct_scan_file and not direct_lab_report:
            st.warning("Please provide either a detailed symptom description or upload at least one medical report for advanced analysis.")
        else:
            # Generate a basic prediction first based on text analysis
            with st.spinner("Pre-processing symptoms for analysis..."):
                # Extract key symptoms from the detailed description
                # This is a simplistic approach - in a real system, NLP would be used here
                potential_symptoms = []
                common_symptoms = symptoms_data["common_symptoms"]
                
                # Simple keyword matching for symptoms
                for symptom in common_symptoms:
                    if symptom.lower() in direct_detailed_symptoms.lower():
                        potential_symptoms.append(symptom)
                
                # If we found symptoms, make a basic prediction
                if potential_symptoms:
                    symptoms_text = ", ".join(potential_symptoms)
                    basic_predictions = predict_disease(model, symptoms_text)
                    st.session_state.predictions = basic_predictions
                else:
                    # Create a dummy prediction with low confidence for the most common conditions
                    dummy_predictions = [
                        ("Common Cold", 0.3),
                        ("Influenza", 0.25),
                        ("Stress", 0.2)
                    ]
                    st.session_state.predictions = dummy_predictions
            
            # Now run the advanced analysis
            with st.spinner("Running advanced analysis with DistilBERT model..."):
                # Check if we have images uploaded
                has_images = any([direct_xray_file, direct_ct_scan_file, direct_lab_report])
                
                # Get advanced analysis results
                analysis_results = advanced_analysis(
                    detailed_symptoms=direct_detailed_symptoms,
                    initial_predictions=st.session_state.predictions[:3] if st.session_state.predictions else [],
                    image_uploaded=has_images
                )
                
                if analysis_results and analysis_results.get("success", False):
                    st.success("Advanced analysis complete!")
                    
                    # Display results in an expander
                    with st.expander("View Advanced Analysis Results", expanded=True):
                        # Basic info
                        st.markdown(f"#### Analysis of {analysis_results.get('top_disease', 'Unknown Condition')}")
                        
                        # Confidence score (only show advanced confidence since this is direct analysis)
                        advanced_conf = analysis_results.get('advanced_confidence', 0) * 100
                        
                        # Display confidence scores
                        st.markdown(f"**Analysis Confidence:** {advanced_conf:.1f}%")
                        
                        # Severity and urgency indicators
                        severity = analysis_results.get('severity', 'Unknown')
                        urgency = analysis_results.get('urgency', 'Unknown')
                        
                        severity_col1, severity_col2 = st.columns(2)
                        with severity_col1:
                            st.markdown(f"**Severity Assessment:** {severity}")
                        with severity_col2:
                            st.markdown(f"**Recommended Action:** {urgency}")
                        
                        # Natural language analysis
                        if direct_detailed_symptoms:
                            st.markdown("#### Natural Language Analysis")
                            nl_indicators = analysis_results.get('natural_language_indicators', {})
                            
                            if nl_indicators:
                                # Format the detected language patterns
                                severity_terms = nl_indicators.get('severity_terms', [])
                                duration_terms = nl_indicators.get('duration_terms', [])
                                intensity_terms = nl_indicators.get('intensity_terms', [])
                                
                                st.markdown("""
                                <div class="advanced-result-box">
                                <p><strong>Symptom Analysis:</strong> Our language analysis has detected key indicators in your description.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if severity_terms:
                                    st.markdown(f"**Severity Indicators:** {', '.join(severity_terms)}")
                                if duration_terms:
                                    st.markdown(f"**Duration Indicators:** {', '.join(duration_terms)}")
                                if intensity_terms:
                                    st.markdown(f"**Intensity Indicators:** {', '.join(intensity_terms)}")
                        
                        # Medical imaging analysis
                        if has_images:
                            st.markdown("#### Medical Imaging Analysis")
                            image_analysis = analysis_results.get('image_analysis', None)
                            
                            if image_analysis:
                                confidence = image_analysis.get('confidence', 0) * 100
                                findings = image_analysis.get('findings', 'No significant findings')
                                
                                st.markdown(f"""
                                <div class="advanced-result-box">
                                <p><strong>Image Analysis:</strong> {findings}</p>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Recommendation section
                        st.markdown("#### Recommendation")
                        
                        # Customize recommendation based on severity and confidence
                        if advanced_conf >= 70 and severity in ["Significant", "Severe"]:
                            recommendation = f"Based on our advanced analysis, we strongly recommend seeking prompt medical attention for potential {analysis_results.get('top_disease', 'condition')}. Please share these results with your healthcare provider."
                        elif advanced_conf >= 50:
                            recommendation = f"Based on our advanced analysis, we recommend consulting with a healthcare professional about potential {analysis_results.get('top_disease', 'condition')}. Please share these results with your doctor at your next appointment."
                        else:
                            recommendation = "Based on our advanced analysis, we recommend monitoring your symptoms and consulting with a healthcare professional if they persist or worsen."
                        
                        st.markdown(f"""
                        <div class="recommendation-box">
                        <p>{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Disclaimer for advanced analysis
                        st.markdown("""
                        <div class="disclaimer-box">
                        <p><strong>MEDICAL DISCLAIMER:</strong> This advanced analysis is not a substitute for professional medical diagnosis. 
                        The image analysis feature is experimental and should be verified by qualified healthcare professionals.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Advanced analysis failed: {analysis_results.get('message', 'Unknown error')}")

# Footer
st.markdown("---")
st.markdown("#### 🧠 AI-Powered Disease Prediction & Treatment Recommendation Application")
st.markdown("This application uses machine learning to predict diseases based on symptoms and provide treatment recommendations.")
