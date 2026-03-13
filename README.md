# LLM-Assisted Fraud Intent Annotation

This repository contains a simple implementation of the annotation pipeline
described in our paper:

"Detecting Fraud Intent in Cryptocurrency Discussions with LLM-Assisted Annotation"

The script uses a local large language model via [Ollama](https://ollama.com/)
to classify text comments into three categories:

- Fraud intention
- Solution or prevention intention
- Out of context

Datasets are **not** included for privacy and licensing reasons.
Users must provide their own CSV or Excel file with a text column.

## LLM-assisted labeling script

The file `llm_labeling/run_labeling.py` implements our annotation pipeline.
To use it, install the dependencies (`pandas`, `tqdm`, `ollama`) and configure
a local model in Ollama (for example `mistral:instruct`).

Then edit the parameters in the `__main__` block of `run_labeling.py`:

- `INPUT_FILE`: path to your CSV/Excel file
- `INPUT_SHEET`: sheet name for Excel files (or `None`)
- `TEXT_COLUMN`: name of the text column to classify
- `OUTPUT_FILE`: path for the output CSV with labels
- `MODEL_NAME`: Ollama model to use
- `PROMPT_CHOICE`: `baseline`, `domain_specific`, or `intent_focused`


### Model Selection and Extensions

The script trains three baseline machine learning classifiers:
- Multinomial Naive Bayes
- Logistic Regression
- Random Forest

After evaluating their performance on the test set, the pipeline automatically selects the best-performing model based on the F1-score and saves it together with the TF-IDF vectorizer.

This implementation is intended as a **baseline example**. Users are encouraged to further improve performance by:

- tuning model hyperparameters (e.g., using grid search or cross-validation),
- experimenting with additional machine learning models,
- modifying the TF-IDF configuration or feature extraction strategy.

Depending on the specific classification task and dataset characteristics, other classifiers (e.g., SVM, gradient boosting, or neural models) may yield better results.
