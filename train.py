import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading data...")
flashcards = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
clues = flashcards.drop('diseases', axis=1) 
answers = flashcards['diseases']

print("2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(clues, answers, test_size=0.2, random_state=42)

print("3. Training the NEW lightweight brain (Naive Bayes)...")
# We swapped the heavy Decision Tree for the ultra-light Naive Bayes!
robot_brain = MultinomialNB().fit(X_train, y_train)

print("4. Testing accuracy...")
predictions = robot_brain.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("5. Saving compressed files...")
joblib.dump(robot_brain, "robot_brain.joblib", compress=5)
joblib.dump(list(clues.columns), "raw_clues.joblib", compress=5)

print("Done!")