import time
from typing import List

import ollama
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. Prompt definitions (edit the text, keep the variables)
# ============================================================

BASELINE_PROMPT_TEMPLATE = """
You are an expert classifier for detecting fraudulent intent.
You must classify the comment into one of the three categories. Even if uncertain, select the closest category.

Categories:
1. Fraud intention → The comment expresses intent to commit fraud, even if small.
2. Solution or prevention intention → The comment suggests fraud prevention or security advice.
3. Out of context → The comment is unrelated to fraud.

Provide the output in the following format:
Label: [category]
Explanation: [brief explanation]

Now classify this comment:
Comment: "{comment}"

Your response must be structured as:
Label: [category]
Explanation: [brief reason]
""".strip()

DOMAIN_SPECIFIC_PROMPT_TEMPLATE = """
You are an expert classifier for detecting fraudulent intent in the cryptocurrency
and cryptography domain.
You must classify the comment into one of the three categories. Even if uncertain, select the closest category.

Categories:
1. Fraud intention → The comment expresses intent to commit fraud, even if small.
2. Solution or prevention intention → The comment suggests fraud prevention or security advice.
3. Out of context → The comment is unrelated to fraud.

Provide the output in the following format:
Label: [category]
Explanation: [brief explanation]

Now classify this comment:
Comment: "{comment}"

Your response must be structured as:
Label: [category]
Explanation: [brief reason]
""".strip()

INTENT_FOCUSED_PROMPT_TEMPLATE = """
You are an expert classifier for detecting fraudulent intent.
You must classify the comment into one of the three categories. Even if uncertain, select the closest category.
As you process these comments, learn from the language, patterns, and terminology present in the set of comments you are classifying.
Adjust your understanding and interpretation as you encounter new domain-specific phrases or patterns.

Categories:
1. Fraud intention → The comment expresses intent to commit fraud, even if small.
2. Solution or prevention intention → The comment suggests fraud prevention or security advice.
3. Out of context → The comment is unrelated to fraud.

Focus on the intent rather than isolated keywords. When security terminology appears
(encrypt, hide, anonymize, etc.), check the overall intent.

Provide the output in the following format:
Label: [category]
Explanation: [brief explanation]

Now classify this comment:
Comment: "{comment}"

Your response must be structured as:
Label: [category]
Explanation: [brief reason]
""".strip()


# ============================================================
# 2. Helper functions
# ============================================================

def format_prompt(comment: str, template: str) -> str:
    """Fill the {comment} placeholder in the chosen prompt template."""
    return template.format(comment=comment)


def safe_ollama_chat(prompt: str, model_name: str = "mistral:instruct") -> str:
    """
    Call an Ollama model with simple retry logic.
    Users can replace this with their own LLM backend if desired.
    """
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "Error: Connection failed after retries."


def classify_comments(
    comments: List[str],
    prompt_template: str,
    model_name: str = "mistral:instruct",
) -> List[str]:
    """Classify a list of comments using the chosen prompt and LLM model."""
    results = []
    for comment in tqdm(comments, desc="Classifying comments"):
        prompt = format_prompt(comment, prompt_template)
        output = safe_ollama_chat(prompt, model_name=model_name)
        results.append(output)
    return results


def extract_label_and_explanation(raw_output: str):
    """
    Extract 'Label' and 'Explanation' fields from the LLM output.

    Expects a format similar to:
        Label: 1. Fraud intention
        Explanation: some text...
    """
    label = "Error"
    explanation = raw_output

    if "Label:" in raw_output:
        try:
            after_label = raw_output.split("Label:", 1)[1]
            if "Explanation:" in after_label:
                label_part, explanation_part = after_label.split("Explanation:", 1)
                label = label_part.strip()
                explanation = explanation_part.strip()
            else:
                label = after_label.strip()
                explanation = ""
        except Exception:
            pass

    return label, explanation


# ============================================================
# 3. Main script
# ============================================================

if __name__ == "__main__":
    # User parameters (edit these before running)
    INPUT_FILE = "manual_annotation.xlsx"      # can also be a CSV
    INPUT_SHEET = None                        # e.g. "Sheet1" for Excel, or None
    TEXT_COLUMN = "comment"                   # name of the text column
    OUTPUT_FILE = "LLM_annotations_intent_focused.csv"
    MODEL_NAME = "mistral:instruct"           # replace with your model if needed

    # Choose which prompt to use: "baseline", "domain_specific", or "intent_focused"
    PROMPT_CHOICE = "intent_focused"

    if PROMPT_CHOICE == "baseline":
        PROMPT_TEMPLATE = BASELINE_PROMPT_TEMPLATE
    elif PROMPT_CHOICE == "domain_specific":
        PROMPT_TEMPLATE = DOMAIN_SPECIFIC_PROMPT_TEMPLATE
    elif PROMPT_CHOICE == "intent_focused":
        PROMPT_TEMPLATE = INTENT_FOCUSED_PROMPT_TEMPLATE
    else:
        raise ValueError(f"Unknown PROMPT_CHOICE: {PROMPT_CHOICE}")

    # Load data
    if INPUT_FILE.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
    else:
        df = pd.read_csv(INPUT_FILE)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input file must contain a column named '{TEXT_COLUMN}'.")

    comments = df[TEXT_COLUMN].astype(str).tolist()

    # Classification
    print("Starting classification...")
    raw_outputs = classify_comments(
        comments,
        prompt_template=PROMPT_TEMPLATE,
        model_name=MODEL_NAME,
    )

    labels, explanations = [], []
    for out in raw_outputs:
        label, explanation = extract_label_and_explanation(out)
        labels.append(label)
        explanations.append(explanation)

    df["Label"] = labels
    df["Explanation"] = explanations

    # Save results
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Classification complete. Results saved to '{OUTPUT_FILE}'.")
