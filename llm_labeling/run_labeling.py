import pandas as pd
import ollama
from tqdm import tqdm
import time

# =============================
# Helper functions ( correct code )
# =============================

def format_prompt(comment: str) -> str:
    """Constructs the classification prompt for a single comment."""
    prompt = f"""
You are an expert classifier for detecting fraudulent intent.
You must classify the comment into one of the three categories. Even if uncertain, select the closest category.

Categories:
1. Fraud intention → The comment expresses intent to commit fraud, even if small.
2. Solution or prevention intention → The comment suggests fraud prevention or security advice.
3. Out of context → The comment is unrelated to fraud.

Provide the output in the format:
Label: [category]
Explanation: [brief explanation]

Now classify this comment:
Comment: "{comment}"

Your response must be structured as:
Label: [category]
Explanation: [brief reason]
"""
    return prompt


def safe_ollama_chat(prompt, model_name="mistral:instruct"):
    """Calls Ollama with retry logic in case of disconnection."""
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}  # deterministic
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return "Error: Connection failed after retries."


def classify_comments_with_ollama(comments, model_name="mistral:instruct"):
    """Classifies a list of comments using Ollama."""
    results = []

    for comment in tqdm(comments, desc="Classifying comments"):
        prompt = format_prompt(comment)
        output = safe_ollama_chat(prompt, model_name)
        results.append(output)
    
    return results

# =============================
# Main script
# =============================

# Load CSV file
df = pd.read_excel("manual_annotation.xlsx")


# Ensure the column exists
if "comment" not in df.columns:
    raise ValueError("❌ The CSV must contain a column named 'comment'.")

# Get the comments as a list
comments = df["comment"].astype(str).tolist()

# Classify all comments
print("🚀 Starting classification...")
classifications = classify_comments_with_ollama(comments)

# Extract Label and Explanation from the model output
labels, explanations = [], []
for result in classifications:
    if "Label:" in result:
        label_part = result.split("Label:", 1)[1].split("Explanation:", 1)[0].strip()
        explanation_part = result.split("Explanation:", 1)[1].strip() if "Explanation:" in result else ""
    else:
        label_part, explanation_part = "Error", result
    labels.append(label_part)
    explanations.append(explanation_part)

# Add results to DataFrame
df["Label"] = labels
df["Explanation"] = explanations

# Save to a new CSV
output_file = "LLM_annotation_till131.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f" Classification complete! Results saved to '{output_file}'")
