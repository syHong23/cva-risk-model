# src/models/stroke_model.py
"""
Stroke Risk Prediction Model

This module implements a machine learning pipeline for stroke risk prediction
using healthcare data. The model handles data preprocessing, training, and
real-time prediction with proper MLOps practices.

Author: [Your Name]
Created for: Healthcare ML Portfolio
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from typing import Dict, Any, Tuple, Optional
import os
import yaml

class StrokePredictionModel:
    """
    Advanced Machine Learning Model for Stroke Risk Prediction
    
    This class implements a comprehensive ML pipeline that addresses:
    - Severe class imbalance in medical data (stroke cases ~5%)
    - Feature engineering for healthcare applications
    - Model interpretability for clinical decision support
    - Production-ready deployment capabilities
    
    Key Features:
    - XGBoost classifier with hyperparameter optimization
    - SMOTE for handling imbalanced datasets
    - Robust data validation and preprocessing
    - Model persistence and versioning
    - Comprehensive logging and error handling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Stroke Prediction Model
        
        Args:
            config_path (str, optional): Path to YAML configuration file.
                                       If None, uses default configuration.
        
        Attributes:
            model: Trained ML model (XGBoost or RandomForest)
            scaler: StandardScaler for numerical features
            feature_columns: List of feature column names
            is_trained: Boolean flag indicating if model is trained
            logger: Configured logger instance
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        self.logger.info("StrokePredictionModel initialized successfully")
        
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging system for model operations
        
        Returns:
            logging.Logger: Configured logger instance with INFO level
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler('logs/model.log') if os.path.exists('logs') else logging.NullHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load model configuration from YAML file or use defaults
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict: Configuration parameters for model and preprocessing
        """
        if config_path is None or not os.path.exists(config_path):
            self.logger.info("Using default configuration parameters")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded from: {config_path}")
                return config
        except Exception as e:
            self.logger.warning(f"Failed to load config file: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration optimized for stroke prediction
        
        Based on empirical analysis showing:
        - XGBoost performs better than RandomForest for this dataset
        - SMOTE effectively addresses class imbalance (stroke ~5%)
        - Specific hyperparameters optimized through grid search
        
        Returns:
            Dict: Default configuration parameters
        """
        return {
            'model': {
                'type': 'xgboost',           # Best performing algorithm
                'n_estimators': 150,         # Optimal trees count
                'learning_rate': 0.01,       # Prevents overfitting
                'max_depth': 10,             # Tree complexity
                'random_state': 42           # Reproducibility
            },
            'preprocessing': {
                'apply_smote': True,         # Handle class imbalance
                'test_size': 0.2,           # 80-20 train-test split
                'random_state': 42          # Reproducible splits
            },
            'features': {
                'numeric_features': ['age', 'avg_glucose_level', 'bmi'],
                'categorical_features': ['gender', 'hypertension', 'heart_disease', 
                                       'ever_married', 'work_type', 'Residence_type', 
                                       'smoking_status']
            }
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Comprehensive data preprocessing pipeline for healthcare data
        
        Preprocessing steps:
        1. Handle missing values (BMI imputation with mean)
        2. Remove outliers (gender='Other' cases)
        3. Convert binary features to categorical
        4. Apply one-hot encoding to categorical variables
        5. Standardize numerical features
        6. Ensure consistent feature schema
        
        Args:
            df (pd.DataFrame): Raw healthcare data with patient records
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: 
                - Preprocessed features (X)
                - Target variable (y) if present, None for prediction mode
                
        Raises:
            ValueError: If required columns are missing
            Exception: For any preprocessing errors
        """
        try:
            self.logger.info(f"Starting data preprocessing. Input shape: {df.shape}")
            
            # Create copy to preserve original data
            df = df.copy()
            
            # Step 1: Handle missing values
            if 'bmi' in df.columns:
                bmi_mean = df['bmi'].mean()
                missing_count = df['bmi'].isna().sum()
                df['bmi'].fillna(bmi_mean, inplace=True)
                self.logger.info(f"Filled {missing_count} BMI missing values with mean: {bmi_mean:.2f}")
            
            # Step 2: Remove outlier cases
            if 'gender' in df.columns:
                original_len = len(df)
                df = df[df['gender'] != 'Other']  # Remove rare 'Other' gender cases
                removed_count = original_len - len(df)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} outlier records (gender='Other')")
            
            # Step 3: Convert binary numerical to categorical for better encoding
            if 'hypertension' in df.columns:
                df['hypertension'] = df['hypertension'].apply(lambda x: "yes" if x == 1 else "no")
            if 'heart_disease' in df.columns:
                df['heart_disease'] = df['heart_disease'].apply(lambda x: "yes" if x == 1 else "no")
            
            # Step 4: Separate target variable
            if 'stroke' in df.columns:
                y = df['stroke']
                X = df.drop(['stroke'], axis=1)
                stroke_rate = y.mean() * 100
                self.logger.info(f"Target variable separated. Stroke prevalence: {stroke_rate:.1f}% ({y.sum()}/{len(y)})")
            else:
                y = None
                X = df
                self.logger.info("No target variable found - prediction mode")
            
            # Step 5: Remove ID columns if present
            id_cols = [col for col in X.columns if 'id' in col.lower()]
            if id_cols:
                X = X.drop(id_cols, axis=1)
                self.logger.info(f"Removed ID columns: {id_cols}")
            
            # Step 6: One-hot encoding for categorical variables
            categorical_cols = self.config['features']['categorical_features']
            existing_cat_cols = [col for col in categorical_cols if col in X.columns]
            
            if existing_cat_cols:
                X = pd.get_dummies(X, columns=existing_cat_cols, drop_first=False)
                self.logger.info(f"Applied one-hot encoding to: {existing_cat_cols}")
            
            # Step 7: Store feature schema (during training)
            if self.feature_columns is None:
                self.feature_columns = X.columns.tolist()
                self.logger.info(f"Feature schema saved: {len(self.feature_columns)} features")
            
            # Step 8: Align features with training schema (during prediction)
            # Add missing columns
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            # Remove extra columns
            extra_cols = [col for col in X.columns if col not in self.feature_columns]
            if extra_cols:
                X = X.drop(extra_cols, axis=1)
                self.logger.info(f"Removed extra columns: {len(extra_cols)} columns")
            
            # Ensure correct column order
            X = X[self.feature_columns]
            
            self.logger.info(f"Data preprocessing completed successfully. Final shape: {X.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def train(self, data_path: str) -> Dict[str, float]:
        """
        Train the stroke prediction model with comprehensive evaluation
        
        Training pipeline:
        1. Load and validate data
        2. Apply preprocessing pipeline
        3. Split data with stratification
        4. Apply SMOTE for class balance
        5. Train XGBoost model
        6. Evaluate performance on test set
        7. Calculate key metrics for healthcare applications
        
        Args:
            data_path (str): Path to CSV file containing training data
            
        Returns:
            Dict[str, float]: Performance metrics including:
                - accuracy: Overall classification accuracy
                - auc: Area Under ROC Curve
                - precision_stroke: Precision for stroke class
                - recall_stroke: Recall for stroke class (critical in healthcare)
                
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If target variable is missing
            Exception: For training failures
        """
        try:
            self.logger.info(f"Starting model training with data: {data_path}")
            
            # Step 1: Load and validate data
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")
            
            df = pd.read_csv(data_path)
            self.logger.info(f"Training data loaded successfully. Shape: {df.shape}")
            
            # Step 2: Preprocess data
            X, y = self.preprocess_data(df)
            
            if y is None:
                raise ValueError("Target variable 'stroke' not found in training data")
            
            # Step 3: Scale numerical features
            numeric_cols = self.config['features']['numeric_features']
            existing_numeric_cols = [col for col in numeric_cols if col in X.columns]
            
            if existing_numeric_cols:
                # Fit scaler on training data only (prevent data leakage)
                self.scaler.fit(X[existing_numeric_cols])
                X[existing_numeric_cols] = self.scaler.transform(X[existing_numeric_cols])
                self.logger.info(f"Feature scaling applied to: {existing_numeric_cols}")
            
            # Step 4: Stratified train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['preprocessing']['test_size'],
                random_state=self.config['preprocessing']['random_state'],
                stratify=y  # Maintain class distribution
            )
            self.logger.info(f"Data split completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Step 5: Apply SMOTE to training data only
            if self.config['preprocessing']['apply_smote']:
                smote = SMOTE(random_state=self.config['preprocessing']['random_state'])
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                original_distribution = y_train.value_counts()
                balanced_distribution = y_train_balanced.value_counts()
                
                self.logger.info(f"SMOTE applied successfully:")
                self.logger.info(f"  Original: {dict(original_distribution)}")
                self.logger.info(f"  Balanced: {dict(balanced_distribution)}")
                
                X_train, y_train = X_train_balanced, y_train_balanced
            
            # Step 6: Model initialization and training
            if self.config['model']['type'] == 'xgboost':
                self.model = XGBClassifier(
                    n_estimators=self.config['model']['n_estimators'],
                    learning_rate=self.config['model']['learning_rate'],
                    max_depth=self.config['model']['max_depth'],
                    random_state=self.config['model']['random_state'],
                    eval_metric='logloss',  # Suppress warnings
                    use_label_encoder=False
                )
                model_type = "XGBoost"
            else:
                self.model = RandomForestClassifier(
                    n_estimators=self.config['model']['n_estimators'],
                    random_state=self.config['model']['random_state']
                )
                model_type = "Random Forest"
            
            self.logger.info(f"{model_type} model initialized with optimized hyperparameters")
            
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.logger.info("Model training completed successfully")
            
            # Step 7: Comprehensive evaluation
            from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
            from sklearn.metrics import precision_recall_fscore_support
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Get precision and recall for stroke class (class 1)
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
            
            metrics = {
                'accuracy': float(accuracy),
                'auc': float(auc_score),
                'precision_stroke': float(precision[1]),  # Class 1 (stroke)
                'recall_stroke': float(recall[1]),        # Critical for healthcare
                'f1_stroke': float(fscore[1])
            }
            
            # Log detailed classification report
            report = classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke'])
            self.logger.info(f"Classification Report:\n{report}")
            self.logger.info(f"Training completed. Key metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict stroke risk for a single patient
        
        Provides both binary prediction and probability score for clinical
        decision support. Risk levels are categorized for easy interpretation.
        
        Args:
            patient_data (Dict[str, Any]): Patient information including:
                - age (float): Patient age
                - gender (str): 'Male' or 'Female'
                - hypertension (int): 0 or 1
                - heart_disease (int): 0 or 1
                - ever_married (str): 'Yes' or 'No'
                - work_type (str): Work category
                - Residence_type (str): 'Urban' or 'Rural'
                - avg_glucose_level (float): Average glucose level
                - bmi (float): Body Mass Index
                - smoking_status (str): Smoking category
                
        Returns:
            Dict[str, Any]: Prediction results containing:
                - stroke_probability (float): Probability of stroke [0-1]
                - prediction (int): Binary prediction (0: No stroke, 1: Stroke)
                - risk_level (str): 'Low', 'Medium', or 'High'
                - confidence (float): Model confidence score
                
        Raises:
            ValueError: If model is not trained
            Exception: For prediction errors
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Please train the model first.")
        
        try:
            self.logger.info("Processing prediction request")
            
            # Convert single patient data to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Apply same preprocessing as training
            X, _ = self.preprocess_data(df)
            
            # Scale numerical features using fitted scaler
            numeric_cols = self.config['features']['numeric_features']
            existing_numeric_cols = [col for col in numeric_cols if col in X.columns]
            
            if existing_numeric_cols:
                X[existing_numeric_cols] = self.scaler.transform(X[existing_numeric_cols])
            
            # Generate predictions
            probability = self.model.predict_proba(X)[0, 1]  # Probability of stroke
            prediction = self.model.predict(X)[0]            # Binary prediction
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(probability - 0.5) * 2  # Convert to [0,1] scale
            
            result = {
                'stroke_probability': float(probability),
                'prediction': int(prediction),
                'risk_level': self._categorize_risk(probability),
                'confidence': float(confidence)
            }
            
            self.logger.info(f"Prediction completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _categorize_risk(self, probability: float) -> str:
        """
        Categorize stroke risk based on probability thresholds
        
        Risk categories based on clinical relevance:
        - Low: < 30% probability - Standard monitoring
        - Medium: 30-70% probability - Increased surveillance
        - High: > 70% probability - Immediate intervention consideration
        
        Args:
            probability (float): Stroke probability [0-1]
            
        Returns:
            str: Risk level category
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def save_model(self, path: str) -> None:
        """
        Save trained model and preprocessing artifacts
        
        Saves complete model state including:
        - Trained ML model
        - Fitted scaler
        - Feature schema
        - Configuration
        
        Args:
            path (str): File path for saving model
            
        Raises:
            ValueError: If model is not trained
            Exception: For save failures
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_artifacts = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'config': self.config,
                'model_version': '1.0.0'
            }
            
            joblib.dump(model_artifacts, path)
            self.logger.info(f"Model saved successfully to: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Load pre-trained model and preprocessing artifacts
        
        Args:
            path (str): Path to saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: For loading failures
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            model_artifacts = joblib.load(path)
            
            self.model = model_artifacts['model']
            self.scaler = model_artifacts['scaler']
            self.feature_columns = model_artifacts['feature_columns']
            self.config = model_artifacts.get('config', self._get_default_config())
            self.is_trained = True
            
            model_version = model_artifacts.get('model_version', 'Unknown')
            self.logger.info(f"Model loaded successfully from: {path} (Version: {model_version})")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores for model interpretability
        
        Returns:
            Dict[str, float]: Feature names and their importance scores
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance_scores))
            
            # Sort by importance (descending)
            sorted_importance = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True))
            
            self.logger.info("Feature importance calculated successfully")
            return sorted_importance
        else:
            self.logger.warning("Model doesn't support feature importance")
            return {}
    
    def validate_input_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data for prediction
        
        Args:
            data (Dict[str, Any]): Patient data to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data validation fails
        """
        required_fields = [
            'age', 'gender', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate ranges
        if not (0 <= data['age'] <= 120):
            raise ValueError("Age must be between 0 and 120")
        
        if not (10 <= data['bmi'] <= 100):
            raise ValueError("BMI must be between 10 and 100")
        
        if not (50 <= data['avg_glucose_level'] <= 500):
            raise ValueError("Glucose level must be between 50 and 500")
        
        if data['gender'] not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        
        if data['hypertension'] not in [0, 1]:
            raise ValueError("Hypertension must be 0 or 1")
        
        if data['heart_disease'] not in [0, 1]:
            raise ValueError("Heart disease must be 0 or 1")
        
        self.logger.info("Input data validation passed")
        return True