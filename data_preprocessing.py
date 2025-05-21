import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ParkinsonDataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        self.data = pd.read_csv(self.file_path)
        print("\nDataset Information:")
        print("-" * 50)
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")
        print("\nColumn names:", self.data.columns.tolist())
        print("\nMissing values:", self.data.isnull().sum().sum())
        return self.data
        
    def explore_data(self):
        """Explore the dataset and visualize key features"""
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print("-" * 50)
        print(self.data.describe())
        
        # Distribution of target variable
        plt.figure(figsize=(10, 6))
        sns.countplot(x='class', data=self.data)
        plt.title('Distribution of Parkinson Status')
        plt.show()
        
    def prepare_data(self, test_size=0.2, random_state=42, target='class'):
        """Prepare the data for machine learning"""
        if self.data is None:
            self.load_data()  # Load data if not already loaded
            
        # Separate features and target
        self.X = self.data.drop(['id', target], axis=1)
        self.y = self.data[target]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    processor = ParkinsonDataPreprocessor('parkinson_disease.csv')
    processor.explore_data()
    X_train, X_test, y_train, y_test = processor.prepare_data()
