import streamlit as st

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
