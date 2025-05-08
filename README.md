
# AI Disease Prediction & Treatment Recommendation System

## Overview
An AI-powered application that predicts diseases based on symptoms and provides treatment recommendations. The system uses both basic rule-based prediction and advanced analysis with natural language processing and medical image analysis capabilities.

## Features
- Symptom-based disease prediction
- Confidence scoring system
- Advanced analysis using:
  - Natural Language Processing for detailed symptom descriptions
  - Medical image analysis (X-rays, CT scans)
  - Lab report analysis
- Treatment recommendations
- Medication suggestions with pharmacy links
- Prevention tips and lifestyle recommendations
- Real-time pharmacy availability checking

## Technical Stack
- Frontend: Streamlit
- Backend: Python
- Machine Learning: Transformers (DistilBERT), PyTorch
- Data Storage: JSON-based knowledge base
- External APIs: Pharmacy scrapers for medication availability

## Project Structure
```
├── app.py                    # Main Streamlit application
├── models.py                 # Core prediction models and logic
├── advanced_models.py        # Advanced analysis implementations
├── pharmacy_scraper.py       # Pharmacy data scraping utilities
├── data/
│   ├── diseases.json        # Disease information database
│   └── symptoms.json        # Symptoms database
```

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r render_requirements.txt
```

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Core Components

### 1. Disease Prediction
- Rule-based symptom matching
- Confidence score calculation
- Support for multiple diseases prediction

### 2. Advanced Analysis
- Natural language processing for detailed symptom analysis
- Severity assessment
- Urgency classification
- Medical image analysis support
- Lab report interpretation

### 3. Treatment Recommendations
- Disease-specific treatments
- Medication suggestions
- Prevention tips
- Lifestyle recommendations

### 4. Pharmacy Integration
- Real-time medication availability checking
- Multiple pharmacy source support
- Direct purchase links

## API Reference

### Core Functions

#### `predict_disease(model, symptoms_text)`
Predicts diseases based on input symptoms
- Parameters:
  - model: Loaded disease prediction model
  - symptoms_text: Comma-separated symptom string
- Returns: List of (disease, confidence) tuples

#### `advanced_analysis(detailed_symptoms, initial_predictions, image_uploaded)`
Performs comprehensive analysis of symptoms
- Parameters:
  - detailed_symptoms: Detailed symptom description
  - initial_predictions: Initial disease predictions
  - image_uploaded: Boolean for image analysis
- Returns: Dictionary with analysis results

## Data Structure

### Disease Information
Each disease entry contains:
- Description
- Common symptoms
- Causes
- Prevention measures
- Treatment options
- Recommended medications

### Symptom Categories
- Physical symptoms
- Mental health indicators
- Vital signs
- Laboratory findings

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## Medical Disclaimer
This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers for medical conditions.

## License
MIT License - feel free to use and modify for your projects
