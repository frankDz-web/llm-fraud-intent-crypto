"""
Text classification pipeline for fraud-intent datasets.

Input:
    CSV file (e.g. cleaned_dataset.csv) with:
      - text  : text to classify
      - label : target class (e.g. fraud, prevention, out_of_context)

The script:
    - splits the data into train/test sets
    - vectorizes text with TF-IDF
    - trains multiple classifiers (Naive Bayes, Logistic Regression, Random Forest)
    - evaluates them with standard metrics
    - saves the best-performing model and the TF-IDF vectorizer

Users can modify or extend the `train_models` method to plug in other
classifiers if desired.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TextClassifier:
    def __init__(self, csv_path, test_size=0.2, random_state=42):
        """
        Parameters
        ----------
        csv_path : str
            Path to the cleaned dataset CSV.
        test_size : float
            Proportion of data used for the test set.
        random_state : int
            Random seed for reproducibility.
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state

        self.vectorizer = None
        self.models = {}
        self.results = {}

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vec = None
        self.X_test_vec = None

    # ------------------------------------------------------------
    # Data loading and splitting
    # ------------------------------------------------------------

    def load_data(self):
        """Load and check the cleaned CSV."""
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} samples")
            print(f"Columns: {df.columns.tolist()}")

            if {"text", "label"} - set(df.columns):
                raise ValueError("Input file must contain 'text' and 'label' columns.")

            # Remove rows with missing values
            if df.isnull().any().any():
                print("\nMissing values detected: dropping rows with NaN.")
                df = df.dropna()
                print(f"After cleanup: {len(df)} samples")

            print("\nLabel distribution:")
            print(df["label"].value_counts())
            print(f"Total unique labels: {df['label'].nunique()}\n")

            self.X = df["text"].astype(str).values
            self.y = df["label"].values

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def split_data(self):
        """Split data into training and test sets."""
        print("=" * 60)
        print("STEP 2: SPLITTING DATA")
        print("=" * 60)

        unique_labels, label_counts = np.unique(self.y, return_counts=True)
        min_samples = int(label_counts.min())

        print(f"Minimum samples in any class: {min_samples}")

        can_stratify = min_samples >= 2

        if can_stratify:
            print("Using stratified split to preserve class distribution.\n")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.y,
            )
        else:
            print("Not enough samples per class for stratified split; using random split.\n")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set:     {len(self.X_test)} samples")
        print(f"Train/Test split: {100 * (1 - self.test_size):.0f}% / {100 * self.test_size:.0f}%\n")

        print("Training set label distribution:")
        for label, count in zip(*np.unique(self.y_train, return_counts=True)):
            print(f"  {label}: {count} samples")
        print()

    # ------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------

    def vectorize_text(self):
        """Convert text to TF-IDF feature vectors."""
        print("=" * 60)
        print("STEP 3: VECTORIZING TEXT (TF-IDF)")
        print("=" * 60)

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            stop_words="english",
        )

        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

        print(f"Feature matrix (train): {self.X_train_vec.shape}")
        print(f"Feature matrix (test):  {self.X_test_vec.shape}")
        print(f"Number of features:     {len(self.vectorizer.get_feature_names_out())}\n")

    # ------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------

    def train_models(self):
        """
        Train several baseline classifiers.

        Users can extend this method to include other models,
        or comment out models they do not need.
        """
        print("=" * 60)
        print("STEP 4: TRAINING MODELS")
        print("=" * 60)

        # Model 1: Multinomial Naive Bayes
        print("\n[1/3] Training Multinomial Naive Bayes...")
        try:
            nb_model = MultinomialNB()
            nb_model.fit(self.X_train_vec, self.y_train)
            self.models["Naive Bayes"] = nb_model
            print("Naive Bayes trained.")
        except Exception as e:
            print(f"Naive Bayes training failed: {e}")

        # Model 2: Logistic Regression
        print("\n[2/3] Training Logistic Regression...")
        try:
            lr_model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                random_state=self.random_state,
            )
            lr_model.fit(self.X_train_vec, self.y_train)
            self.models["Logistic Regression"] = lr_model
            print("Logistic Regression trained.")
        except Exception as e:
            print(f"Logistic Regression training failed: {e}")

        # Model 3: Random Forest
        print("\n[3/3] Training Random Forest...")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            rf_model.fit(self.X_train_vec, self.y_train)
            self.models["Random Forest"] = rf_model
            print("Random Forest trained.")
        except Exception as e:
            print(f"Random Forest training failed: {e}")

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def evaluate_models(self):
        """Evaluate all trained models on the test set."""
        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 60)

        if not self.models:
            print("No models were trained.")
            return False

        for model_name, model in self.models.items():
            print("\n" + "-" * 60)
            print(f"Model: {model_name}")
            print("-" * 60)

            y_pred = model.predict(self.X_test_vec)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

            print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")

            self.results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            print("\nClassification report:")
            print(classification_report(self.y_test, y_pred, zero_division=0))

        return True

    # ------------------------------------------------------------
    # Saving and inference
    # ------------------------------------------------------------

    def save_best_model(self):
        """Save the best model (by F1 score) and the TF-IDF vectorizer."""
        print("=" * 60)
        print("STEP 6: SAVING BEST MODEL")
        print("=" * 60)

        if not self.results:
            print("No evaluation results available.")
            return None

        best_model_name = max(self.results, key=lambda name: self.results[name]["f1"])
        best_model = self.models[best_model_name]
        best_f1 = self.results[best_model_name]["f1"]
        best_accuracy = self.results[best_model_name]["accuracy"]

        model_path = f"best_classifier_{best_model_name.replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        vectorizer_path = "tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"\nBest model: {best_model_name}")
        print(f"Accuracy:   {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
        print(f"F1-score:   {best_f1:.4f}")
        print(f"Model saved to:     {model_path}")
        print(f"Vectorizer saved to:{vectorizer_path}\n")

        return best_model_name, model_path, vectorizer_path

    def predict_new_text(self, texts):
        """Predict labels for new text samples using the best model."""
        if self.vectorizer is None or not self.models or not self.results:
            print("Models are not trained or evaluated yet.")
            return None, None

        best_model_name = max(self.results, key=lambda name: self.results[name]["f1"])
        best_model = self.models[best_model_name]

        texts_vec = self.vectorizer.transform(texts)
        predictions = best_model.predict(texts_vec)

        return predictions, best_model_name

    # ------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------

    def run_pipeline(self):
        """Run the full ML pipeline."""
        print()
        print("=" * 60)
        print("ML TEXT CLASSIFICATION PIPELINE")
        print("=" * 60)

        if not self.load_data():
            return False

        self.split_data()
        self.vectorize_text()
        self.train_models()

        if not self.evaluate_models():
            return False

        result = self.save_best_model()

        if result is None:
            print("=" * 60)
            print("PIPELINE FAILED (model not saved).")
            print("=" * 60)
            return False

        best_model_name, model_path, vectorizer_path = result

        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nBest model: {best_model_name}")
        print(f"Saved model:      {model_path}")
        print(f"Saved vectorizer: {vectorizer_path}\n")

        return True


if __name__ == "__main__":
    # Example usage:
    # csv_path should point to the cleaned dataset produced by the
    # label normalization script (e.g. 'cleaned_dataset.csv').
    classifier = TextClassifier(
        csv_path="cleaned_dataset.csv",
        test_size=0.2,
        random_state=42,
    )

    success = classifier.run_pipeline()

    if success:
        # Optional example: predict on a few sample texts
        example_texts = [
            "This is a legitimate discussion about blockchain technology.",
            "Click here to get free coins, limited time offer.",
            "Here is how to protect yourself from cryptocurrency scams.",
        ]
        preds, model_name = classifier.predict_new_text(example_texts)

        if preds is not None:
            print("Example predictions (using best model):")
            for text, pred in zip(example_texts, preds):
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"  Text: {preview}")
                print(f"  Predicted label: {pred}")
