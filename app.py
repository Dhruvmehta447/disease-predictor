import os
import joblib
import gradio as gr

# 1. Load the pre-trained robot brain and clues directly! 
# (This uses almost NO memory)
robot_brain = joblib.load("robot_brain.joblib")
raw_clues = joblib.load("raw_clues.joblib")

# --- MAGIC CLEANUP TRICK ---
clue_dictionary = {c.replace('_', ' ').title(): c for c in raw_clues}

# 2. Make the guessing function
def guess_sickness(symptoms):
    if not symptoms or len(symptoms) < 3:
        return "⚠️ Please select at least 3 symptoms to get a prediction!"

    my_clues = [0] * len(raw_clues)
    for s in symptoms:
        my_clues[raw_clues.index(clue_dictionary[s])] = 1

    probs = robot_brain.predict_proba([my_clues])[0]
    top_3 = sorted(zip(robot_brain.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

    final_message = "🩺 Top Predictions:\n\n"
    for disease, confidence in top_3:
        if confidence > 0:
            final_message += f"• {disease.replace('_', ' ').title()} ({confidence * 100:.0f}% confidence)\n"
        
    return final_message

# 3. Build the BEAUTIFUL Website!
with gr.Blocks() as webpage:
    gr.Markdown("# 🩺 Disease Prediction App\nSelect at least **3 symptoms** from the dropdown below and click **Predict** to see possible diseases.")
    
    symptom_dropdown = gr.Dropdown(choices=list(clue_dictionary.keys()), multiselect=True, label="Choose your symptoms:")
    predict_btn = gr.Button("Predict")
    output_box = gr.Textbox(label="Prediction Results", lines=5)
    
    predict_btn.click(fn=guess_sickness, inputs=symptom_dropdown, outputs=output_box)

# Launch it for cloud deployment!
webpage.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
