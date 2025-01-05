import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import joblib
from collections import Counter

# Download stopwords
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    cleaned_text = ''.join([char.lower() if char.isalpha() or char.isspace() else ' ' for char in text])
    tokens = cleaned_text.split()
    tokens = [word for word in tokens if word not in stop_words]
    abbreviation_mapping = {"CKD": "Cooked", "STMD": "Steamed", "RAW": "Raw"}
    expanded_text = ' '.join([abbreviation_mapping.get(word.upper(), word) for word in tokens])
    return expanded_text

# Load the model
model = joblib.load('healthbuddyapp3.pkl')

# Define nutrients and thresholds
target_names = ['Vitamin_A', 'Vitamin_B12', 'Vitamin_C', 'Vitamin_D', 'Vitamin_E', 'Vitamin_K']
deficiency_thresholds = {
    'Vitamin_A': 15.0,
    'Vitamin_B12': 0.5,
    'Vitamin_C': 10.0,
    'Vitamin_D': 0.5,
    'Vitamin_E': 5.0,
    'Vitamin_K': 2.0,
}

# Define health and disease data
health_disease_data = {
    "Vitamin_A": {"Diseases": ["Night blindness"], "Foods": ["Carrots", "Spinach"]},
    "Vitamin_B12": {"Diseases": ["Anemia"], "Foods": ["Meat", "Dairy"]},
    "Vitamin_C": {"Diseases": ["Scurvy"], "Foods": ["Oranges", "Strawberries"]},
    "Vitamin_D": {"Diseases": ["Rickets"], "Foods": ["Fatty fish", "Mushrooms"]},
    "Vitamin_E": {"Diseases": ["Nerve damage"], "Foods": ["Nuts", "Seeds"]},
    "Vitamin_K": {"Diseases": ["Bleeding disorders"], "Foods": ["Leafy greens", "Broccoli"]},
}

# Streamlit app
st.title("HealthBuddy Vitamin Deficiency Tracker")
st.write("Enter food names to analyze possible vitamin deficiencies.")

# User input
food_input = st.text_input("Enter food names separated by commas:")

if food_input:
    food_names = [preprocess_text(food.strip()) for food in food_input.split(',')]
    predictions = model.predict(food_names)

    deficiencies_for_day = []
    for i, food in enumerate(food_names):
        food_predictions = dict(zip(target_names, predictions[i]))
        deficiencies = [
            nutrient for nutrient, value in food_predictions.items()
            if value < deficiency_thresholds.get(nutrient, float('inf'))
        ]
        deficiencies_for_day.extend(deficiencies)

    deficiency_counts = Counter(deficiencies_for_day)
    
    if deficiency_counts:
        st.write("### Deficiencies Detected:")
        for vitamin, count in deficiency_counts.items():
            st.write(f"**{vitamin}**: {count} occurrences")
            if vitamin in health_disease_data:
                st.write(f"  - Diseases: {', '.join(health_disease_data[vitamin]['Diseases'])}")
                st.write(f"  - Foods to Eat: {', '.join(health_disease_data[vitamin]['Foods'])}")
    else:
        st.write("No deficiencies detected.")
