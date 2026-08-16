import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import gradio as gr

# 1. Read the giant book of sicknesses
flashcards = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.zip")
clues = flashcards.drop('diseases', axis=1) 
answers = flashcards['diseases']

# 2. Train the smart robot!
robot_brain = DecisionTreeClassifier().fit(clues, answers)

# --- MAGIC CLEANUP TRICK ---
# Make the computer words beautiful (e.g., 'skin_rash' becomes 'Skin Rash')
raw_clues = list(clues.columns)
clue_dictionary = {c.replace('_', ' ').title(): c for c in raw_clues}

# 3. Make the guessing function
def guess_sickness(symptoms):
    if not symptoms or len(symptoms) < 3:
        return "⚠️ Please select at least 3 symptoms to get a prediction!"

    # Create the zero list and mark the chosen symptoms with a 1
    my_clues = [0] * len(raw_clues)
    for s in symptoms:
        my_clues[raw_clues.index(clue_dictionary[s])] = 1

    # Get probabilities (how confident the robot is) and find the top 3!
    probs = robot_brain.predict_proba([my_clues])[0]
    top_3 = sorted(zip(robot_brain.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

    # Format the beautiful output message
    final_message = "🩺 Top Predictions:\n\n"
    for disease, confidence in top_3:
        if confidence > 0:
            final_message += f"• {disease.replace('_', ' ').title()} ({confidence * 100:.0f}% confidence)\n"
        
    return final_message

# 4. Build the BEAUTIFUL Website!
with gr.Blocks() as webpage:
    gr.Markdown("# 🩺 Disease Prediction App\nSelect at least **3 symptoms** from the dropdown below and click **Predict** to see possible diseases.")
    
    # Create the dropdown menu, button, and output box
    symptom_dropdown = gr.Dropdown(choices=list(clue_dictionary.keys()), multiselect=True, label="Choose your symptoms:")
    predict_btn = gr.Button("Predict")
    output_box = gr.Textbox(label="Prediction Results", lines=5)
    
    # Connect the button!
    predict_btn.click(fn=guess_sickness, inputs=symptom_dropdown, outputs=output_box)

# Launch it for cloud deployment!
webpage.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
