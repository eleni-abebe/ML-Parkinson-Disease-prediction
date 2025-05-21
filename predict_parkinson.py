import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_training import ParkinsonModelTrainer
from data_preprocessing import ParkinsonDataPreprocessor

class ParkinsonPredictor:
    def __init__(self, model_path=None):
        """Initialize the predictor with trained models"""
        self.models = None
        self.scaler = StandardScaler()
        self.feature_count = None
        self.feature_names = None
        
    def load_models(self):
        """Load the trained models"""
        # Create a temporary processor to load the data
        processor = ParkinsonDataPreprocessor('parkinson_disease.csv')
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Store the number of features and their names
        self.feature_count = X_train.shape[1]
        self.feature_names = processor.X.columns
        
        # Train models
        trainer = ParkinsonModelTrainer(X_train, X_test, y_train, y_test)
        trainer.train_models()
        
        # Store the models
        self.models = trainer.results
        
    def preprocess_input(self, input_data):
        """Preprocess input data to match the model's requirements"""
        # Create DataFrame with proper number of features
        if len(input_data) != self.feature_count:
            print(f"\nWarning: Input data has {len(input_data)} features, but model expects {self.feature_count} features.")
            print("Using random values for missing features.")
            
            # Create DataFrame with random values for all features
            input_data = pd.DataFrame(np.random.rand(1, self.feature_count), columns=self.feature_names)
        else:
            input_data = pd.DataFrame([input_data], columns=self.feature_names)
        
        # Scale the features
        input_scaled = self.scaler.fit_transform(input_data)
        
        return input_scaled
    
    def predict(self, input_data):
        """Make predictions using all trained models"""
        if self.models is None:
            self.load_models()
            
        input_processed = self.preprocess_input(input_data)
        predictions = {}
        
        for model_name, result in self.models.items():
            model = result['model']
            prediction = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)[:, 1]
            
            predictions[model_name] = {
                'prediction': prediction[0],
                'probability': prediction_proba[0],
                'status': 'Parkinson' if prediction[0] == 1 else 'Healthy'
            }
        
        return predictions

def main():
    # Example usage
    predictor = ParkinsonPredictor()
    
    print("\nWelcome to Parkinson's Disease Prediction System")
    print("-" * 50)
    
    # Show the number of features expected by the model
    print("\nModel Information:")
    print("-" * 50)
    if predictor.feature_count is not None:
        print(f"The model expects {predictor.feature_count} features as input")
    else:
        print("The model's expected feature count is not yet determined.")
    print("\nFeature names:")
    print("-" * 50)
    if predictor.feature_names is not None:
        print(predictor.feature_names.tolist())
    else:
        print("The model's feature names are not yet determined.")
    
    # Example input (random values for demonstration)
    sample_input = [0.75, 0.85, 0.90, 100]  # Example feature values
    
    # Make prediction
    predictions = predictor.predict(sample_input)
    
    print("\nPrediction Results:")
    print("-" * 50)
    for model_name, result in predictions.items():
        print(f"\n{model_name} Prediction:")
        print(f"Status: {result['status']}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Raw Prediction: {result['prediction']}")

if __name__ == "__main__":
    main()
