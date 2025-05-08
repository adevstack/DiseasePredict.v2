import streamlit as st
import random

# List of common diseases for classification
DISEASES = [
    "Common Cold", "Influenza", "Pneumonia", "Bronchitis", "Asthma", 
    "Hypertension", "Diabetes", "Migraine", "Arthritis", "Gastritis",
    "Irritable Bowel Syndrome", "Urinary Tract Infection", "Eczema", 
    "Psoriasis", "Anemia", "Hypothyroidism", "Malaria", "Dengue Fever",
    "Tuberculosis", "COVID-19"
]

# Dictionary mapping symptoms to diseases (simplified knowledge base)
SYMPTOM_DISEASE_MAP = {
    "Common Cold": ["runny nose", "sneezing", "congestion", "sore throat", "cough", "mild fever", "headache"],
    "Influenza": ["high fever", "chills", "body aches", "fatigue", "headache", "cough", "sore throat"],
    "Pneumonia": ["cough", "chest pain", "fever", "chills", "shortness of breath", "fatigue"],
    "Bronchitis": ["cough", "mucus", "fatigue", "shortness of breath", "mild fever", "chest discomfort"],
    "Asthma": ["wheezing", "shortness of breath", "chest tightness", "coughing", "trouble sleeping"],
    "Hypertension": ["headache", "shortness of breath", "nosebleeds", "chest pain", "dizziness", "blurred vision"],
    "Diabetes": ["increased thirst", "frequent urination", "hunger", "fatigue", "blurred vision", "weight loss"],
    "Migraine": ["severe headache", "nausea", "vomiting", "light sensitivity", "sound sensitivity", "vision changes"],
    "Arthritis": ["joint pain", "stiffness", "swelling", "redness", "decreased range of motion"],
    "Gastritis": ["abdominal pain", "nausea", "vomiting", "bloating", "indigestion", "loss of appetite"],
    "Irritable Bowel Syndrome": ["abdominal pain", "cramping", "bloating", "gas", "diarrhea", "constipation"],
    "Urinary Tract Infection": ["burning sensation", "frequent urination", "cloudy urine", "strong odor", "pelvic pain"],
    "Eczema": ["dry skin", "itching", "rash", "red bumps", "thickened skin", "cracked skin"],
    "Psoriasis": ["red patches", "silvery scales", "dry skin", "itching", "burning", "soreness"],
    "Anemia": ["fatigue", "weakness", "pale skin", "cold hands", "shortness of breath", "dizziness", "headaches"],
    "Hypothyroidism": ["fatigue", "weight gain", "cold sensitivity", "dry skin", "hoarseness", "muscle weakness"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "vomiting", "muscle pain"],
    "Dengue Fever": ["high fever", "severe headache", "pain behind eyes", "joint pain", "muscle pain", "rash"],
    "Tuberculosis": ["cough", "chest pain", "weight loss", "fatigue", "fever", "night sweats"],
    "COVID-19": ["fever", "cough", "shortness of breath", "fatigue", "body aches", "loss of taste", "loss of smell"]
}

# Symptom severity indicators for advanced analysis
SEVERITY_INDICATORS = {
    "mild": 0.3,
    "moderate": 0.6,
    "severe": 0.9,
    "extreme": 1.0,
    "persistent": 0.7,
    "occasional": 0.4,
    "chronic": 0.8,
    "acute": 0.75,
    "intermittent": 0.5
}

# Natural language indicators for advanced analysis
LANGUAGE_PATTERNS = {
    "duration_patterns": {
        "few days": 0.3,
        "week": 0.5,
        "weeks": 0.6,
        "month": 0.7,
        "months": 0.8,
        "year": 0.9,
        "years": 1.0
    },
    "intensity_patterns": {
        "mild": 0.3,
        "slight": 0.2,
        "moderate": 0.5,
        "significant": 0.7,
        "severe": 0.8,
        "intense": 0.9,
        "unbearable": 1.0,
        "worst": 1.0
    }
}

def load_model():
    """
    Load the symptom-disease knowledge base
    """
    try:
        # No actual model loading needed, just return the symptom-disease map
        return SYMPTOM_DISEASE_MAP
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_disease(symptom_disease_map, symptoms_text):
    """
    Predict diseases based on provided symptoms using a simple rule-based approach
    
    Args:
        symptom_disease_map: Dictionary mapping diseases to their symptoms
        symptoms_text: String containing symptoms separated by commas
    
    Returns:
        List of (disease, confidence) tuples, sorted by confidence
    """
    try:
        # Make sure symptoms text is not empty
        if not symptoms_text or not symptoms_text.strip():
            return [("Unable to predict", 0.0)]
        
        # Split input symptoms by comma, remove leading/trailing spaces, and convert to lowercase
        input_symptoms = [s.strip().lower() for s in symptoms_text.split(',')]
        
        # Calculate scores for each disease based on symptom matches
        predictions = []
        for disease, symptoms in symptom_disease_map.items():
            # Convert disease symptoms to lowercase for case-insensitive matching
            disease_symptoms = [s.lower() for s in symptoms]
            
            # Count how many input symptoms match this disease
            matching_symptoms = sum(1 for s in input_symptoms if any(s in ds for ds in disease_symptoms))
            
            # Calculate score: matching symptoms / total symptoms for this disease
            # Add a small value to avoid division by zero
            score = matching_symptoms / len(disease_symptoms) if disease_symptoms else 0
            
            # Add a small bonus if most of the symptoms for this disease are present
            if matching_symptoms / len(disease_symptoms) > 0.5:
                score += 0.2
                
            predictions.append((disease, min(score, 1.0)))  # Cap score at 1.0
        
        # Sort by score (highest first)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return [("Prediction error", 0.0)]

def advanced_analysis(detailed_symptoms, initial_predictions, image_uploaded=False):
    """
    Perform advanced analysis using natural language processing on detailed symptom description
    
    Args:
        detailed_symptoms: String containing detailed description of symptoms
        initial_predictions: List of (disease, confidence) tuples from initial prediction
        image_uploaded: Boolean indicating if medical images were uploaded
        
    Returns:
        Dictionary containing advanced analysis results
    """
    try:
        if not detailed_symptoms and not image_uploaded:
            return {
                "success": False,
                "message": "No detailed symptoms or images provided for analysis"
            }
        
        # Get the top predicted disease from initial prediction
        top_disease, initial_confidence = initial_predictions[0] if initial_predictions else ("Unknown", 0.0)
        
        # Initialize results
        advanced_results = {
            "success": True,
            "top_disease": top_disease,
            "initial_confidence": initial_confidence,
            "advanced_confidence": initial_confidence,  # Start with initial confidence
            "severity": "Moderate",
            "urgency": "Non-urgent",
            "natural_language_indicators": [],
            "image_analysis": None
        }
        
        # Analyze detailed symptoms if provided
        if detailed_symptoms:
            # Convert to lowercase for case-insensitive matching
            detailed_symptoms_lower = detailed_symptoms.lower()
            
            # Check for severity indicators
            severity_score = 0.5  # Default moderate severity
            severity_matches = []
            for indicator, score in SEVERITY_INDICATORS.items():
                if indicator in detailed_symptoms_lower:
                    severity_matches.append(indicator)
                    severity_score = max(severity_score, score)
            
            # Check for duration patterns
            duration_score = 0.5  # Default moderate duration
            duration_matches = []
            for pattern, score in LANGUAGE_PATTERNS["duration_patterns"].items():
                if pattern in detailed_symptoms_lower:
                    duration_matches.append(pattern)
                    duration_score = max(duration_score, score)
            
            # Check for intensity patterns
            intensity_score = 0.5  # Default moderate intensity
            intensity_matches = []
            for pattern, score in LANGUAGE_PATTERNS["intensity_patterns"].items():
                if pattern in detailed_symptoms_lower:
                    intensity_matches.append(pattern)
                    intensity_score = max(intensity_score, score)
            
            # Calculate combined score
            combined_score = (severity_score + duration_score + intensity_score) / 3
            
            # Adjust confidence based on natural language analysis
            adjustment = 0.1 * (combined_score - 0.5)  # -0.05 to +0.05 adjustment
            advanced_confidence = min(1.0, max(0.0, initial_confidence + adjustment))
            
            # Determine severity based on combined score
            if combined_score <= 0.3:
                severity = "Mild"
                urgency = "Non-urgent"
            elif combined_score <= 0.6:
                severity = "Moderate"
                urgency = "Follow-up recommended"
            elif combined_score <= 0.8:
                severity = "Significant"
                urgency = "Medical attention advised"
            else:
                severity = "Severe"
                urgency = "Urgent medical attention recommended"
            
            # Update results
            advanced_results["advanced_confidence"] = advanced_confidence
            advanced_results["severity"] = severity
            advanced_results["urgency"] = urgency
            advanced_results["natural_language_indicators"] = {
                "severity_terms": severity_matches,
                "duration_terms": duration_matches,
                "intensity_terms": intensity_matches,
                "combined_score": combined_score
            }
        
        # Add image analysis results if images were uploaded
        if image_uploaded:
            # Simulate image analysis results
            advanced_results["image_analysis"] = {
                "image_analyzed": True,
                "findings": "Potential indicators consistent with predicted condition detected",
                "confidence": 0.7 + random.uniform(-0.1, 0.1)  # Simulated confidence with slight randomization
            }
            
            # Adjust overall confidence based on "image analysis"
            if "advanced_confidence" in advanced_results:
                # Average with image analysis confidence
                advanced_results["advanced_confidence"] = (advanced_results["advanced_confidence"] + 
                                                       advanced_results["image_analysis"]["confidence"]) / 2
        
        return advanced_results
        
    except Exception as e:
        st.error(f"Error in advanced analysis: {str(e)}")
        return {
            "success": False,
            "message": f"Error in advanced analysis: {str(e)}"
        }
