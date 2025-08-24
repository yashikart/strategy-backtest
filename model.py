import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import xgboost as xgb
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available. LSTM model will not be available.")

from utils import DataProcessor

class MLModelManager:
    """
    Manages multiple ML models for spot price prediction.
    Supports both classification and regression tasks.
    """
    
    def __init__(self):
        """
        Initialize the ML model manager.
        """
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Try importing TensorFlow/Keras for LSTM
        self.tf_available = False
        self.keras_available = False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            self.tf = tf
            self.keras = keras
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            self.Adam = Adam
            
            self.tf_available = True
            self.keras_available = True
            print("✅ TensorFlow/Keras available. LSTM model enabled.")
            
        except ImportError:
            try:
                import keras
                from keras.models import Sequential
                from keras.layers import LSTM, Dense, Dropout
                from keras.optimizers import Adam
                
                self.keras = keras
                self.Sequential = Sequential
                self.LSTM = LSTM
                self.Dense = Dense
                self.Dropout = Dropout
                self.Adam = Adam
                
                self.keras_available = True
                print("✅ Keras available. LSTM model enabled.")
                
            except ImportError:
                print("⚠️ TensorFlow/Keras not available. LSTM model will not be available.")
                print("Install with: pip install tensorflow")
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_type: str = 'classification',
                    target_period: int = 1,
                    test_size: float = 0.2) -> Tuple[Any, Any, Any, Any, List[str]]:
        """
        Prepare data for ML training.
        
        Args:
            df: Input DataFrame with features
            target_type: 'classification' or 'regression'
            target_period: Forward looking period for target
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print(f"Preparing data for {target_type} with {target_period}-period forward looking target...")
        
        # Prepare features and targets
        df_features = DataProcessor.prepare_ml_features(df)
        df_with_targets = DataProcessor.create_target_variables(df_features)
        df_clean = DataProcessor.clean_data(df_with_targets)
        
        # Select features (excluding non-predictive columns)
        exclude_cols = ['datetime', 'signal', 'composite_signal', 'ticker', 'closest_expiry']
        target_cols = [col for col in df_clean.columns if col.startswith(('future_return', 'price_up', 'price_class'))]
        exclude_cols.extend(target_cols)
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Handle any remaining non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if df_clean[col].dtype in ['int64', 'float64'] and not df_clean[col].isna().all():
                numeric_features.append(col)
        
        # Prepare features
        X = df_clean[numeric_features]
        
        # Prepare target
        if target_type == 'classification':
            target_col = f'price_class_{target_period}'
            if target_col not in df_clean.columns:
                raise ValueError(f"Target column {target_col} not found")
            y = df_clean[target_col]
        else:  # regression
            target_col = f'future_return_{target_period}'
            if target_col not in df_clean.columns:
                raise ValueError(f"Target column {target_col} not found")
            y = df_clean[target_col]
        
        # Remove rows with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Features: {len(numeric_features)} numeric features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if target_type == 'classification' else None
        )
        
        return X_train, X_test, y_train, y_test, numeric_features
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, model_name: str = 'logistic'):
        """
        Train logistic regression model for classification.
        """
        print(f"Training Logistic Regression ({model_name})...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        self.models[model_name] = best_model
        self.model_performance[model_name] = {
            'type': 'classification',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return best_model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, 
                           task_type: str = 'classification', model_name: str = 'random_forest'):
        """
        Train random forest model.
        """
        print(f"Training Random Forest {task_type} ({model_name})...")
        
        if task_type == 'classification':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'r2'
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Performance metrics
        if task_type == 'classification':
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)
            score_name = 'accuracy'
        else:
            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            score_name = 'r2_score'
        
        # Feature importance
        feature_importance = best_model.feature_importances_
        
        self.models[model_name] = best_model
        self.feature_importance[model_name] = feature_importance
        self.model_performance[model_name] = {
            'type': task_type,
            f'train_{score_name}': train_score,
            f'test_{score_name}': test_score,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        print(f"Train {score_name}: {train_score:.4f}")
        print(f"Test {score_name}: {test_score:.4f}")
        
        return best_model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, 
                     task_type: str = 'classification', model_name: str = 'xgboost'):
        """
        Train XGBoost model.
        """
        print(f"Training XGBoost {task_type} ({model_name})...")
        
        if task_type == 'classification':
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            scoring = 'accuracy'
        else:
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            scoring = 'r2'
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)  # Reduced CV for speed
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Performance metrics
        if task_type == 'classification':
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)
            score_name = 'accuracy'
        else:
            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            score_name = 'r2_score'
        
        # Feature importance
        feature_importance = best_model.feature_importances_
        
        self.models[model_name] = best_model
        self.feature_importance[model_name] = feature_importance
        self.model_performance[model_name] = {
            'type': task_type,
            f'train_{score_name}': train_score,
            f'test_{score_name}': test_score,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        print(f"Train {score_name}: {train_score:.4f}")
        print(f"Test {score_name}: {test_score:.4f}")
        
        return best_model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, 
                  sequence_length: int = 20, model_name: str = 'lstm'):
        """
        Train LSTM model for time series prediction.
        """
        if not (self.tf_available or self.keras_available):
            print("⚠️ LSTM training skipped: TensorFlow/Keras not available")
            return None
        
        print(f"Training LSTM ({model_name})...")
        
        # Prepare sequence data
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X.iloc[i-seq_length:i].values)
                y_seq.append(y.iloc[i])
            return np.array(X_seq), np.array(y_seq)
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
        
        if len(X_train_seq) < 50:  # Need minimum data for LSTM
            print("⚠️ Not enough data for LSTM training (need >50 sequences)")
            return None
        
        # Build LSTM model
        model = self.Sequential([
            self.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            self.Dropout(0.2),
            self.LSTM(50, return_sequences=False),
            self.Dropout(0.2),
            self.Dense(25, activation='relu'),
            self.Dense(1)
        ])
        
        model.compile(optimizer=self.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model with early stopping
        try:
            if self.tf_available:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
                callbacks = [early_stopping]
            else:
                callbacks = []
            
            history = model.fit(X_train_seq, y_train_seq, 
                              batch_size=32, epochs=50, 
                              validation_data=(X_test_seq, y_test_seq),
                              verbose=0, callbacks=callbacks)
        
            # Predictions
            y_pred_train = model.predict(X_train_seq, verbose=0).flatten()
            y_pred_test = model.predict(X_test_seq, verbose=0).flatten()
            
            # Performance metrics
            train_r2 = r2_score(y_train_seq, y_pred_train)
            test_r2 = r2_score(y_test_seq, y_pred_test)
            train_mse = mean_squared_error(y_train_seq, y_pred_train)
            test_mse = mean_squared_error(y_test_seq, y_pred_test)
            
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_performance[model_name] = {
                'type': 'regression',
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'sequence_length': sequence_length
            }
            
            print(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            
            return model
            
        except Exception as e:
            print(f"⚠️ LSTM training failed: {e}")
            return None
            
    def train_ensemble_model(self, X_train, y_train, X_test, y_test, 
                            task_type: str = 'classification', model_name: str = 'ensemble'):
        """
        Train ensemble model combining multiple base models.
        """
        print(f"Training Ensemble {task_type} ({model_name})...")
        
        if task_type == 'classification':
            # Create base models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            lr = LogisticRegression(random_state=42, max_iter=1000)
            
            # Create ensemble
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
                voting='soft'
            )
            scoring_metric = 'accuracy'
        else:
            # Create base models
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            lr = LinearRegression()
            
            # Create ensemble
            ensemble = VotingRegressor(
                estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)]
            )
            scoring_metric = 'r2'
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = ensemble.predict(X_train)
        y_pred_test = ensemble.predict(X_test)
        
        # Performance metrics
        if task_type == 'classification':
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='weighted')
            recall = recall_score(y_test, y_pred_test, average='weighted')
            f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            self.model_performance[model_name] = {
                'type': task_type,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"Train Accuracy: {train_score:.4f}")
            print(f"Test Accuracy: {test_score:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            self.model_performance[model_name] = {
                'type': task_type,
                'train_r2_score': train_score,
                'test_r2_score': test_score,
                'train_mse': train_mse,
                'test_mse': test_mse
            }
            
            print(f"Train R2: {train_score:.4f}")
            print(f"Test R2: {test_score:.4f}")
        
        self.models[model_name] = ensemble
        return ensemble
    
    def train_clustering_model(self, X_train, y_train, X_test, y_test, 
                              n_clusters: int = 5, model_name: str = 'clustering'):
        """
        Train clustering-based model for pattern recognition.
        """
        print(f"Training Clustering Model ({model_name})...")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels_train = kmeans.fit_predict(X_train)
        cluster_labels_test = kmeans.predict(X_test)
        
        # Add cluster features
        X_train_clustered = X_train.copy()
        X_test_clustered = X_test.copy()
        X_train_clustered['cluster'] = cluster_labels_train
        X_test_clustered['cluster'] = cluster_labels_test
        
        # Train model with cluster features
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_clustered, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train_clustered)
        y_pred_test = rf_model.predict(X_test_clustered)
        
        # Performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        self.models[model_name] = {'kmeans': kmeans, 'classifier': rf_model}
        self.model_performance[model_name] = {
            'type': 'classification_with_clustering',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_clusters': n_clusters
        }
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return {'kmeans': kmeans, 'classifier': rf_model}
    
    def optimize_hyperparameters(self, X_train, y_train, model_type: str = 'xgboost', 
                                task_type: str = 'classification', cv_folds: int = 5):
        """
        Advanced hyperparameter optimization using time series aware cross-validation.
        """
        print(f"Optimizing {model_type} hyperparameters...")
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        if model_type == 'xgboost':
            if task_type == 'classification':
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                scoring = 'accuracy'
            else:
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                scoring = 'r2'
        
        elif model_type == 'random_forest':
            if task_type == 'classification':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                scoring = 'r2'
        
        # Perform grid search with time series cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, scoring=scoring, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def find_best_model(self, metric: str = 'test_accuracy') -> Tuple[str, Any]:
        """
        Find the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.model_performance:
            return None, None
        
        best_score = -float('inf')
        best_model_name = None
        
        for model_name, performance in self.model_performance.items():
            if metric in performance:
                score = performance[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        return best_model_name, self.models.get(best_model_name)
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with model performance summary
        """
        summary_data = []
        
        for model_name, performance in self.model_performance.items():
            row = {'model_name': model_name}
            row.update(performance)
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lstm' and model_name in self.scalers:
            # Handle LSTM prediction
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            # Note: LSTM requires sequence data, this is simplified
            return model.predict(X_scaled)
        else:
            return model.predict(X)

def test_ml_models():
    """
    Comprehensive ML model testing and optimization to find the best combination.
    """
    print("=== Comprehensive ML Model Testing & Optimization ===")
    
    # Load data
    df = pd.read_csv('data/spot_with_composite_signals_2023.csv')
    print(f"Loaded {len(df)} rows of data")
    
    # Initialize model manager
    ml_manager = MLModelManager()
    
    # Use subset for faster training but ensure statistical validity
    df_subset = df.iloc[1000:8000].copy()  # Larger subset for better training
    print(f"Using {len(df_subset)} rows for training")
    
    print("\n=== Phase 1: Baseline Model Training ===")
    
    # Prepare data for classification
    X_train, X_test, y_train, y_test, feature_names = ml_manager.prepare_data(
        df_subset, target_type='classification', target_period=1, test_size=0.25
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")
    print(f"Feature names: {len(feature_names)} features")
    
    # Train baseline models
    model_results = {}
    
    print("\n--- Training Logistic Regression ---")
    ml_manager.train_logistic_regression(X_train, y_train, X_test, y_test, 'logistic_baseline')
    
    print("\n--- Training Random Forest ---")
    ml_manager.train_random_forest(X_train, y_train, X_test, y_test, 'classification', 'rf_baseline')
    
    print("\n--- Training XGBoost ---")
    ml_manager.train_xgboost(X_train, y_train, X_test, y_test, 'classification', 'xgb_baseline')
    
    print("\n--- Training Ensemble Model ---")
    ml_manager.train_ensemble_model(X_train, y_train, X_test, y_test, 'classification', 'ensemble_class')
    
    print("\n--- Training Clustering-Enhanced Model ---")
    ml_manager.train_clustering_model(X_train, y_train, X_test, y_test, n_clusters=5, model_name='cluster_enhanced')
    
    print("\n=== Phase 2: Hyperparameter Optimization ===")
    
    # Optimize best performing models
    best_xgb, best_xgb_params = ml_manager.optimize_hyperparameters(
        X_train, y_train, 'xgboost', 'classification', cv_folds=3
    )
    
    # Train optimized model
    y_pred_train_opt = best_xgb.predict(X_train)
    y_pred_test_opt = best_xgb.predict(X_test)
    
    train_acc_opt = accuracy_score(y_train, y_pred_train_opt)
    test_acc_opt = accuracy_score(y_test, y_pred_test_opt)
    
    ml_manager.models['xgb_optimized'] = best_xgb
    ml_manager.model_performance['xgb_optimized'] = {
        'type': 'classification',
        'train_accuracy': train_acc_opt,
        'test_accuracy': test_acc_opt,
        'best_params': best_xgb_params
    }
    
    print(f"Optimized XGBoost - Train Accuracy: {train_acc_opt:.4f}, Test Accuracy: {test_acc_opt:.4f}")
    
    print("\n=== Phase 3: Regression Models ===")
    
    # Prepare data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, _ = ml_manager.prepare_data(
        df_subset, target_type='regression', target_period=1, test_size=0.25
    )
    
    print(f"Regression target stats: Mean={y_train_reg.mean():.4f}, Std={y_train_reg.std():.4f}")
    
    ml_manager.train_random_forest(X_train_reg, y_train_reg, X_test_reg, y_test_reg, 'regression', 'rf_regression')
    ml_manager.train_xgboost(X_train_reg, y_train_reg, X_test_reg, y_test_reg, 'regression', 'xgb_regression')
    ml_manager.train_ensemble_model(X_train_reg, y_train_reg, X_test_reg, y_test_reg, 'regression', 'ensemble_reg')
    
    # Train LSTM if available
    if KERAS_AVAILABLE:
        print("\n--- Training LSTM ---")
        ml_manager.train_lstm(X_train_reg, y_train_reg, X_test_reg, y_test_reg, 
                             sequence_length=15, model_name='lstm_optimized')
    
    print("\n=== Phase 4: Model Selection & Evaluation ===")
    
    # Get comprehensive model summary
    summary_df = ml_manager.get_model_summary()
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Find best models by different criteria
    classification_models = summary_df[summary_df['type'].str.contains('classification', na=False)]
    regression_models = summary_df[summary_df['type'].str.contains('regression', na=False)]
    
    print("\n=== Best Model Selection ===")
    
    if not classification_models.empty:
        # Find best classification model
        if 'test_accuracy' in classification_models.columns:
            best_class_idx = classification_models['test_accuracy'].idxmax()
            best_class_model = classification_models.loc[best_class_idx]
            print(f"Best Classification Model: {best_class_model['model_name']} (Accuracy: {best_class_model['test_accuracy']:.4f})")
        
    if not regression_models.empty:
        # Find best regression model
        if 'test_r2_score' in regression_models.columns:
            best_reg_idx = regression_models['test_r2_score'].idxmax()
            best_reg_model = regression_models.loc[best_reg_idx]
            print(f"Best Regression Model: {best_reg_model['model_name']} (R2: {best_reg_model['test_r2_score']:.4f})")
    
    # Feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    for model_name, importance in ml_manager.feature_importance.items():
        if importance is not None and len(importance) > 0:
            # Get top 10 features
            feature_imp_df = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            print(f"\nTop 10 Features - {model_name}:")
            for _, row in feature_imp_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save comprehensive results
    summary_df.to_csv('results/model_performance_detailed.csv', index=False)
    print("\nDetailed model summary saved to: results/model_performance_detailed.csv")
    
    # Save feature importance
    if ml_manager.feature_importance:
        feature_imp_data = []
        for model_name, importance in ml_manager.feature_importance.items():
            if importance is not None:
                for i, imp in enumerate(importance):
                    if i < len(feature_names):
                        feature_imp_data.append({
                            'model': model_name,
                            'feature': feature_names[i],
                            'importance': imp
                        })
        
        if feature_imp_data:
            feature_imp_df = pd.DataFrame(feature_imp_data)
            feature_imp_df.to_csv('results/feature_importance.csv', index=False)
            print("Feature importance saved to: results/feature_importance.csv")
    
    print("\n=== Model Optimization Complete ===")
    return ml_manager, summary_df

if __name__ == "__main__":
    test_ml_models()