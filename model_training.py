import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


class ParkinsonModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(random_state=42)
        }
        self.results = {}
        
    def train_models(self):
        """Train multiple models and evaluate their performance"""
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            print("-" * 50)
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Predict and evaluate
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"\n{name} Performance:")
            print("-" * 50)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy
            }
            
    def compare_models(self):
        """Compare the performance of different models"""
        accuracies = {name: result['accuracy'] for name, result in self.results.items()}
        
        print("\nModel Comparison:")
        print("-" * 50)
        for name, acc in accuracies.items():
            print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    # Load preprocessed data
    import data_preprocessing
    processor = data_preprocessing.ParkinsonDataPreprocessor('parkinson_disease.csv')
    processor.load_data()
    X_train, X_test, y_train, y_test = processor.prepare_data()
    
    # Initialize and train models
    trainer = ParkinsonModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_models()
    trainer.compare_models()
