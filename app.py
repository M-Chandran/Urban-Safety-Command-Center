from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import csv
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json

app = Flask(__name__, static_folder='static')
app.secret_key = 'crime-prediction-secret-key-change-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_DATA_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODELS_FOLDER = 'models'
HISTORY_FOLDER = 'history'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# CSV file for storing user data
USERS_CSV = 'users.csv'

# Initialize CSV files if they don't exist
def init_csv_files():
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'username', 'email', 'password', 'created_at'])

# Ensure users.csv exists for both `python app.py` and `flask run`.
init_csv_files()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

# Helper functions
def allowed_data_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATA_EXTENSIONS

def normalize_text(value):
    return (value or '').strip()

def to_user(row):
    return User(row['id'], row['username'], row['email'], row['password'])

def get_user_by_username(username):
    username = normalize_text(username).lower()
    if not username:
        return None
    if not os.path.exists(USERS_CSV):
        return None
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if normalize_text(row.get('username')).lower() == username:
                return to_user(row)
    return None

def get_user_by_email(email):
    email = normalize_text(email).lower()
    if not email:
        return None
    if not os.path.exists(USERS_CSV):
        return None
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if normalize_text(row.get('email')).lower() == email:
                return to_user(row)
    return None

def get_user_by_identifier(identifier):
    identifier = normalize_text(identifier).lower()
    if not identifier:
        return None
    if not os.path.exists(USERS_CSV):
        return None
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_username = normalize_text(row.get('username')).lower()
            row_email = normalize_text(row.get('email')).lower()
            if row_username == identifier or row_email == identifier:
                return to_user(row)
    return None

def get_user_by_id(user_id):
    if not os.path.exists(USERS_CSV):
        return None
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['id'] == str(user_id):
                return to_user(row)
    return None

def user_exists(username=None, email=None):
    username = normalize_text(username).lower()
    email = normalize_text(email).lower()

    if not os.path.exists(USERS_CSV):
        return {'username': False, 'email': False}

    exists = {'username': False, 'email': False}
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if username and normalize_text(row.get('username')).lower() == username:
                exists['username'] = True
            if email and normalize_text(row.get('email')).lower() == email:
                exists['email'] = True
            if exists['username'] and exists['email']:
                break
    return exists

def add_user(username, email, password):
    user_id = str(uuid.uuid4())[:8]
    password_hash = generate_password_hash(password)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(USERS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, normalize_text(username), normalize_text(email), password_hash, created_at])
    
    return user_id

def get_user_history_dir(user_id):
    user_history_dir = os.path.join(HISTORY_FOLDER, str(user_id))
    os.makedirs(user_history_dir, exist_ok=True)
    return user_history_dir

def get_user_history_file(user_id):
    return os.path.join(get_user_history_dir(user_id), 'predictions.csv')

def init_user_history_file(user_id):
    history_file = get_user_history_file(user_id)
    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'model_name',
                'algorithm',
                'prediction_type',
                'prediction',
                'risk_level',
                'risk_score',
                'confidence',
                'feature_values',
                'notes'
            ])
    return history_file

def log_prediction_event(
    user_id,
    model_name,
    algorithm,
    prediction_type,
    prediction_value,
    risk_level='UNKNOWN',
    risk_score=0,
    confidence=None,
    feature_values=None,
    notes=''
):
    try:
        history_file = init_user_history_file(user_id)
        if isinstance(feature_values, (list, tuple, np.ndarray)):
            feature_text = ", ".join([str(v) for v in feature_values])
        else:
            feature_text = normalize_text(feature_values)

        with open(history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                normalize_text(model_name),
                normalize_text(algorithm),
                normalize_text(prediction_type),
                round(float(prediction_value), 4) if prediction_value is not None else '',
                normalize_text(risk_level),
                round(float(risk_score), 2) if risk_score is not None else '',
                round(float(confidence), 2) if confidence is not None else '',
                feature_text,
                normalize_text(notes)
            ])
    except Exception:
        # History logging should never break core prediction flows.
        pass

def get_recent_prediction_history(user_id, limit=20):
    history_file = get_user_history_file(user_id)
    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows = rows[-limit:]
        rows.reverse()
        return rows
    except Exception:
        return []

# Dark color palette for charts - crime theme (red/orange for danger)
CRIME_COLORS = [
    {'bg': 'rgba(220, 53, 69, 0.7)', 'border': 'rgba(220, 53, 69, 1)'},
    {'bg': 'rgba(255, 123, 114, 0.7)', 'border': 'rgba(255, 123, 114, 1)'},
    {'bg': 'rgba(253, 126, 20, 0.7)', 'border': 'rgba(253, 126, 20, 1)'},
    {'bg': 'rgba(255, 193, 7, 0.7)', 'border': 'rgba(255, 193, 7, 1)'},
    {'bg': 'rgba(23, 162, 184, 0.7)', 'border': 'rgba(23, 162, 184, 1)'},
    {'bg': 'rgba(102, 16, 242, 0.7)', 'border': 'rgba(102, 16, 242, 1)'},
    {'bg': 'rgba(232, 62, 140, 0.7)', 'border': 'rgba(232, 62, 140, 1)'},
    {'bg': 'rgba(40, 167, 69, 0.7)', 'border': 'rgba(40, 167, 69, 1)'},
    {'bg': 'rgba(0, 123, 255, 0.7)', 'border': 'rgba(0, 123, 255, 1)'},
    {'bg': 'rgba(108, 117, 125, 0.7)', 'border': 'rgba(108, 117, 125, 1)'},
]

class ContextAwareBayesianTensorDecompositionRegressor:
    """
    Context-aware Bayesian tensor decomposition regressor.

    Steps:
    1) Build a context matrix from standardized features.
    2) Perform low-rank decomposition on feature-context covariance.
    3) Train Bayesian regression on original + tensor-latent components.
    """

    def __init__(self, rank=8, random_state=42):
        self.rank = int(rank)
        self.random_state = int(random_state)

    def _ensure_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("Input features must be 2D.")
        return X

    def _build_context_matrix(self, X_scaled):
        row_mean = np.mean(X_scaled, axis=1, keepdims=True)
        row_std = np.std(X_scaled, axis=1, keepdims=True)
        row_min = np.min(X_scaled, axis=1, keepdims=True)
        row_max = np.max(X_scaled, axis=1, keepdims=True)

        if X_scaled.shape[1] > 1:
            diffs = np.diff(X_scaled, axis=1)
            diff_mean = np.mean(diffs, axis=1, keepdims=True)
            diff_std = np.std(diffs, axis=1, keepdims=True)
        else:
            diff_mean = np.zeros((X_scaled.shape[0], 1), dtype=float)
            diff_std = np.zeros((X_scaled.shape[0], 1), dtype=float)

        return np.concatenate(
            [
                X_scaled,
                X_scaled ** 2,
                row_mean,
                row_std,
                row_min,
                row_max,
                diff_mean,
                diff_std
            ],
            axis=1
        )

    def fit(self, X, y):
        X = self._ensure_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Training data is empty.")

        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        self.feature_mean_ = np.mean(X, axis=0)
        self.feature_scale_ = np.std(X, axis=0)
        self.feature_scale_ = np.where(self.feature_scale_ <= 1e-8, 1.0, self.feature_scale_)
        X_scaled = (X - self.feature_mean_) / self.feature_scale_

        if np.allclose(y, y[0]):
            self.is_constant_model_ = True
            self.constant_prediction_ = float(np.mean(y))
            self.feature_importances_ = np.zeros(n_features, dtype=float)
            self.factor_feature_ = np.zeros((n_features, 1), dtype=float)
            self.factor_context_ = np.zeros((self._build_context_matrix(X_scaled).shape[1], 1), dtype=float)
            self.singular_values_ = np.zeros(1, dtype=float)
            self.bayesian_model_ = None
            return self

        self.is_constant_model_ = False
        context_matrix = self._build_context_matrix(X_scaled)
        covariance = (X_scaled.T @ context_matrix) / max(float(n_samples), 1.0)
        left_u, singular_values, right_vt = np.linalg.svd(covariance, full_matrices=False)

        max_rank = min(left_u.shape[1], right_vt.shape[0], max(self.rank, 1))
        self.rank_ = int(max_rank)

        self.factor_feature_ = left_u[:, :self.rank_]
        self.factor_context_ = right_vt.T[:, :self.rank_]
        self.singular_values_ = singular_values[:self.rank_]

        latent_feature = X_scaled @ self.factor_feature_
        latent_context = context_matrix @ self.factor_context_
        tensor_latent = (latent_feature * latent_context) * self.singular_values_[None, :]

        design_matrix = np.concatenate([X_scaled, tensor_latent], axis=1)
        self.bayesian_model_ = BayesianRidge(compute_score=False)
        self.bayesian_model_.fit(design_matrix, y)

        coefficients = np.asarray(self.bayesian_model_.coef_, dtype=float).reshape(-1)
        linear_coeff = np.abs(coefficients[:n_features])
        tensor_coeff = np.abs(coefficients[n_features:n_features + self.rank_])

        context_strength = np.mean(np.abs(self.factor_context_), axis=0)
        tensor_strength = np.abs(self.factor_feature_) @ (tensor_coeff * context_strength)
        self.feature_importances_ = linear_coeff + tensor_strength

        if not np.all(np.isfinite(self.feature_importances_)) or self.feature_importances_.size != n_features:
            self.feature_importances_ = np.zeros(n_features, dtype=float)

        return self

    def predict(self, X):
        X = self._ensure_2d(X)

        if X.shape[1] != getattr(self, 'n_features_in_', X.shape[1]):
            raise ValueError(
                f"Expected {getattr(self, 'n_features_in_', X.shape[1])} features, got {X.shape[1]}."
            )

        if getattr(self, 'is_constant_model_', False):
            return np.full(X.shape[0], float(self.constant_prediction_), dtype=float)

        X_scaled = (X - self.feature_mean_) / self.feature_scale_
        context_matrix = self._build_context_matrix(X_scaled)

        latent_feature = X_scaled @ self.factor_feature_
        latent_context = context_matrix @ self.factor_context_
        tensor_latent = (latent_feature * latent_context) * self.singular_values_[None, :]

        design_matrix = np.concatenate([X_scaled, tensor_latent], axis=1)
        prediction = self.bayesian_model_.predict(design_matrix)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        prediction = self.predict(X).reshape(-1)

        if y.size < 2 or np.allclose(y, y[0]):
            return 0.0

        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 1e-12:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

def get_available_models():
    """Return available ML models"""
    return [
        {'id': 'linear', 'name': 'Linear Regression', 'desc': 'Simple linear relationship'},
        {'id': 'ridge', 'name': 'Ridge Regression', 'desc': 'Linear with L2 regularization'},
        {'id': 'lasso', 'name': 'Lasso Regression', 'desc': 'Linear with L1 regularization'},
        {'id': 'elastic_net', 'name': 'Elastic Net', 'desc': 'Combines L1 and L2 regularization'},
        {'id': 'decision_tree', 'name': 'Decision Tree', 'desc': 'Tree-based non-linear model'},
        {'id': 'random_forest', 'name': 'Random Forest', 'desc': 'Ensemble of decision trees'},
        {'id': 'extra_trees', 'name': 'Extra Trees', 'desc': 'Randomized tree ensemble'},
        {'id': 'gradient_boosting', 'name': 'Gradient Boosting', 'desc': 'Sequential ensemble method'},
        {'id': 'adaboost', 'name': 'AdaBoost', 'desc': 'Boosting-based ensemble model'},
        {
            'id': 'context_aware_bayesian_tensor',
            'name': 'Context-Aware Bayesian Tensor Decomposition',
            'desc': 'Low-rank Bayesian tensor model with adaptive context features'
        },
    ]

def get_model_display_name(model_type):
    for model in get_available_models():
        if model['id'] == model_type:
            return model['name']
    return model_type.replace('_', ' ').title()

def create_model(model_type):
    if model_type == 'linear':
        return LinearRegression()
    if model_type == 'ridge':
        return Ridge(alpha=1.0)
    if model_type == 'lasso':
        return Lasso(alpha=1.0)
    if model_type == 'elastic_net':
        return ElasticNet(alpha=0.5, l1_ratio=0.5)
    if model_type == 'decision_tree':
        return DecisionTreeRegressor(max_depth=8, random_state=42)
    if model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=150, random_state=42)
    if model_type == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=150, random_state=42)
    if model_type == 'gradient_boosting':
        return GradientBoostingRegressor(n_estimators=150, random_state=42)
    if model_type == 'adaboost':
        return AdaBoostRegressor(n_estimators=150, random_state=42)
    if model_type in ('context_aware_bayesian_tensor', 'cabtd'):
        return ContextAwareBayesianTensorDecompositionRegressor(rank=8, random_state=42)
    return LinearRegression()

def load_tabular_data(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    if filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    raise ValueError('Unsupported file format')

def clean_numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')

    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace(r'[^0-9.\-]', '', regex=True)
        .str.strip()
    )
    cleaned = cleaned.mask(cleaned == '')
    return pd.to_numeric(cleaned, errors='coerce')

def detect_label_column(df):
    location_keywords = (
        'location',
        'city',
        'neighborhood',
        'area',
        'district',
        'region',
        'zone',
        'ward',
        'county',
        'state'
    )

    for column in df.columns:
        normalized = str(column).strip().lower().replace(' ', '_')
        if any(keyword in normalized for keyword in location_keywords):
            return column

    for column in df.columns:
        numeric_col = clean_numeric_series(df[column])
        numeric_ratio = float(numeric_col.notna().sum()) / max(len(df), 1)
        if numeric_ratio < 0.5:
            return column

    return df.columns[0] if len(df.columns) > 0 else None

def get_target_statistics(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0}

    std_value = float(np.std(values))
    if std_value <= 0:
        std_value = 1.0
    return {
        'mean': float(np.mean(values)),
        'std': std_value,
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }

def prepare_training_data(df, target_column):
    if target_column not in df.columns:
        return None, "Target column not found"

    y_series = clean_numeric_series(df[target_column])

    feature_candidates = []
    for column in df.columns:
        if column == target_column:
            continue
        numeric_col = clean_numeric_series(df[column])
        if numeric_col.notna().sum() >= 2:
            feature_candidates.append((column, numeric_col))

    if not feature_candidates:
        return None, "No usable numeric feature columns found in the file."

    feature_df = pd.DataFrame({column: values for column, values in feature_candidates})
    combined = pd.concat([feature_df, y_series.rename(target_column)], axis=1)
    combined = combined.dropna(subset=[target_column])
    combined = combined[combined[feature_df.columns].notna().any(axis=1)]

    if len(combined) < 2:
        return None, "Not enough valid rows after cleaning data."

    feature_columns = feature_df.columns.tolist()
    for column in feature_columns:
        median_value = combined[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        combined[column] = combined[column].fillna(median_value)

    X = combined[feature_columns].to_numpy(dtype=float)
    y = combined[target_column].to_numpy(dtype=float)
    feature_defaults = {column: float(combined[column].median()) for column in feature_columns}
    target_statistics = get_target_statistics(y)
    return (X, y, feature_columns, feature_defaults, target_statistics), None

def safe_model_score(model, X_data, y_data):
    if len(y_data) < 2:
        return 0.0
    if np.allclose(y_data, y_data[0]):
        return 0.0

    score = float(model.score(X_data, y_data))
    if np.isnan(score) or np.isinf(score):
        return 0.0
    return score

def to_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number

def estimate_accuracy_percent(model_info, key='test_accuracy'):
    """
    Return a user-facing accuracy percentage (0-100).

    - Uses score directly when it is in [0, 1].
    - Supports legacy stored percentages in [0, 100].
    - Falls back to MAE normalized by target range/std when score is negative.
    """
    if not isinstance(model_info, dict):
        return 0.0

    raw_score = to_finite_float(model_info.get(key))
    if raw_score is not None:
        if 0.0 <= raw_score <= 1.0:
            return round(raw_score * 100.0, 1)
        if 0.0 <= raw_score <= 100.0:
            return round(raw_score, 1)

    mae = to_finite_float(model_info.get('mae'))
    stats = model_info.get('target_statistics') if isinstance(model_info.get('target_statistics'), dict) else {}

    target_min = to_finite_float(stats.get('min'))
    target_max = to_finite_float(stats.get('max'))
    if mae is not None and target_min is not None and target_max is not None:
        target_range = target_max - target_min
        if target_range > 0:
            normalized = (1.0 - (mae / target_range)) * 100.0
            return round(max(0.0, min(100.0, normalized)), 1)

    target_std = to_finite_float(stats.get('std'))
    if mae is not None and target_std is not None and target_std > 0:
        normalized = (1.0 - (mae / target_std)) * 100.0
        return round(max(0.0, min(100.0, normalized)), 1)

    if raw_score is not None:
        return round(max(0.0, min(100.0, raw_score * 100.0)), 1)
    return 0.0

def extract_feature_importance(model, feature_columns):
    if not feature_columns:
        return []

    raw_scores = None
    if hasattr(model, 'feature_importances_'):
        raw_scores = np.asarray(getattr(model, 'feature_importances_'), dtype=float)
    elif hasattr(model, 'coef_'):
        coefficients = np.asarray(getattr(model, 'coef_'), dtype=float)
        if coefficients.ndim > 1:
            coefficients = np.mean(np.abs(coefficients), axis=0)
        else:
            coefficients = np.abs(coefficients)
        raw_scores = coefficients

    if raw_scores is None:
        return []

    scores = np.abs(np.asarray(raw_scores, dtype=float)).flatten()
    if scores.size != len(feature_columns):
        return []

    total = float(np.sum(scores))
    if total <= 0:
        normalized = np.zeros_like(scores)
    else:
        normalized = scores / total

    feature_importance = []
    for feature_name, value in zip(feature_columns, normalized):
        score = float(value)
        feature_importance.append({
            'feature': feature_name,
            'importance': score,
            'percent': round(score * 100, 2)
        })

    feature_importance.sort(key=lambda item: item['importance'], reverse=True)
    return feature_importance

def train_crime_model(filepath, target_column, model_type='linear', degree=2):
    """Train ML model for crime prediction with multiple algorithms"""
    try:
        df = load_tabular_data(filepath)
        columns = df.columns.tolist()
        
        if target_column not in columns:
            return None, "Target column not found"

        prepared, error = prepare_training_data(df, target_column)
        if error:
            return None, error

        X, y, feature_columns, feature_defaults, target_statistics = prepared

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = create_model(model_type)
        model.fit(X_train, y_train)

        train_score = safe_model_score(model, X_train, y_train)
        test_score = safe_model_score(model, X_test, y_test)

        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        residual_std = float(np.std(y_test - y_pred)) if len(y_test) > 1 else rmse
        if residual_std <= 0:
            residual_std = rmse if rmse > 0 else 1.0

        # Refit on full cleaned dataset for production prediction.
        model.fit(X, y)
        feature_importance = extract_feature_importance(model, feature_columns)
        accuracy_percent = estimate_accuracy_percent(
            {'accuracy': train_score, 'mae': mae, 'target_statistics': target_statistics},
            key='accuracy'
        )
        test_accuracy_percent = estimate_accuracy_percent(
            {'test_accuracy': test_score, 'mae': mae, 'target_statistics': target_statistics},
            key='test_accuracy'
        )

        return model, {
            "accuracy": float(train_score),
            "test_accuracy": float(test_score),
            "accuracy_percent": accuracy_percent,
            "test_accuracy_percent": test_accuracy_percent,
            "rmse": rmse,
            "mae": mae,
            "residual_std": residual_std,
            "model_type": model_type,
            "model_type_name": get_model_display_name(model_type),
            "columns": columns,
            "target_column": target_column,
            "degree": degree,
            "training_samples": len(X),
            "feature_count": X.shape[1],
            "feature_columns": feature_columns,
            "feature_defaults": feature_defaults,
            "target_statistics": target_statistics,
            "feature_importance": feature_importance,
            "top_features": [item['feature'] for item in feature_importance[:3]]
        }
    except Exception as e:
        return None, str(e)

def predict_crime(model, input_data, model_type='linear'):
    """Make predictions using trained model"""
    try:
        prediction = model.predict([input_data])[0]
        return float(max(0, prediction))  # Ensure non-negative
    except Exception as e:
        return None

def multi_step_prediction(model, historical_data, steps, model_type='linear'):
    """Make multi-step predictions (like in the research paper)"""
    predictions = []
    current_features = list(historical_data)
    
    for i in range(steps):
        pred = predict_crime(model, current_features, model_type)
        if pred is not None:
            predictions.append({
                'step': i + 1,
                'prediction': round(pred, 2),
                'confidence': calculate_confidence(pred, historical_data)
            })
            # Shift features for next prediction
            current_features = current_features[1:] + [pred]
        else:
            break
    
    return predictions

def calculate_confidence(prediction, historical_data):
    """Calculate prediction confidence based on historical data"""
    try:
        historical_mean = np.mean(historical_data)
        historical_std = np.std(historical_data)
        
        if historical_std > 0:
            z_score = abs(prediction - historical_mean) / historical_std
            # Convert z-score to confidence (0-100%)
            confidence = max(0, min(100, 100 - (z_score * 15)))
            return round(confidence, 1)
        return 75.0  # Default confidence
    except:
        return 75.0

def calculate_risk_level(prediction, historical_mean, historical_std):
    """Calculate risk level based on prediction vs historical data"""
    try:
        if historical_std > 0:
            z_score = (prediction - historical_mean) / historical_std
            if z_score > 1.5:
                return "CRITICAL", 90, "Immediate attention required"
            elif z_score > 1.0:
                return "HIGH", 75, "Enhanced monitoring recommended"
            elif z_score > 0.5:
                return "MEDIUM", 50, "Standard precautions"
            elif z_score > -0.5:
                return "LOW", 25, "Low risk area"
            else:
                return "VERY LOW", 10, "Minimal risk"
        return "UNKNOWN", 50, "Insufficient data"
    except:
        return "UNKNOWN", 50, "Calculation error"

def parse_feature_list(features_text):
    try:
        values = [float(x.strip()) for x in features_text.split(',') if x.strip()]
        if not values:
            return None, 'Features must be comma-separated numbers!'
        return values, None
    except ValueError:
        return None, 'Features must be comma-separated numbers!'

def validate_feature_count(feature_list, model_info):
    expected = int(model_info.get('feature_count', 0) or 0)
    if expected <= 0:
        return True, None
    if len(feature_list) == expected:
        return True, None
    return False, f'Expected {expected} feature values for this model, but got {len(feature_list)}.'

def get_risk_baseline(model_info):
    stats = model_info.get('target_statistics') or {}
    historical_mean = float(stats.get('mean', 0) or 0)
    historical_std = float(stats.get('std', 1) or 1)
    if historical_std <= 0:
        historical_std = 1.0
    return historical_mean, historical_std

def analyze_crime_trends(filepath):
    """Analyze crime trends from data"""
    try:
        df = load_tabular_data(filepath)
        
        columns = df.columns.tolist()
        analysis = {
            'total_records': len(df),
            'columns': columns,
            'crime_types': columns[1:] if len(columns) > 1 else [],
            'statistics': {},
            'trends': {},
            'context_impact': {}
        }
        
        # Calculate statistics for each crime type
        for col in columns[1:]:
            numeric_col = clean_numeric_series(df[col])
            values = numeric_col.dropna().values
            if len(values) >= 2:
                analysis['statistics'][col] = {
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values)),
                    'std': float(np.std(values)),
                    'total': float(np.sum(values)),
                    'median': float(np.median(values))
                }
                
                # Calculate trend (simple linear trend)
                if len(values) > 1:
                    x = np.arange(len(values))
                    z = np.polyfit(x, values, 1)
                    trend_direction = "increasing" if z[0] > 0 else "decreasing"
                    analysis['trends'][col] = {
                        'direction': trend_direction,
                        'slope': float(z[0]),
                        'change_rate': float(z[0] / np.mean(values) * 100) if np.mean(values) != 0 else 0
                    }
        
        # Context impact analysis (simulated based on patterns)
        # In real implementation, this would use actual weather/holiday data
        analysis['context_impact'] = {
            'weather': {
                'fine': {'impact': 'positive', 'description': 'More outdoor activity leads to higher crime'},
                'rainy': {'impact': 'negative', 'description': 'Indoor activity reduces street crime'}
            },
            'holidays': {
                'working_day': {'impact': 'neutral', 'description': 'Standard crime patterns'},
                'rest_day': {'impact': 'positive', 'description': 'More alcohol-related incidents'}
            }
        }
        
        return analysis
    except Exception as e:
        return None

def read_crime_data_file(filepath, chart_type='bar'):
    """Read CSV or Excel file and return data as dictionary"""
    try:
        df = load_tabular_data(filepath)
        
        columns = df.columns.tolist()
        charts = []
        
        if len(columns) > 1:
            label_column = detect_label_column(df)
            if label_column:
                labels = df[label_column].fillna('Unknown').astype(str).tolist()
            else:
                labels = df[columns[0]].fillna('Unknown').astype(str).tolist()

            def aggregate_locations(raw_labels, raw_values):
                """Aggregate duplicate location labels into a single value (mean)."""
                grouped_sum = {}
                grouped_count = {}
                label_order = []

                for raw_label, raw_value in zip(raw_labels, raw_values):
                    label_text = str(raw_label).strip() or 'Unknown'
                    try:
                        numeric_value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(numeric_value):
                        continue

                    if label_text not in grouped_sum:
                        grouped_sum[label_text] = 0.0
                        grouped_count[label_text] = 0
                        label_order.append(label_text)

                    grouped_sum[label_text] += numeric_value
                    grouped_count[label_text] += 1

                unique_labels = []
                unique_values = []
                for label_text in label_order:
                    count = max(grouped_count.get(label_text, 0), 1)
                    unique_labels.append(label_text)
                    unique_values.append(grouped_sum[label_text] / count)

                return unique_labels, unique_values
            
            for i in range(1, len(columns)):
                if columns[i] == label_column:
                    continue

                numeric_col = clean_numeric_series(df[columns[i]])
                if numeric_col.notna().sum() < 2:
                    continue

                raw_values = numeric_col.fillna(0).astype(float).tolist()
                chart_labels, values = aggregate_locations(labels, raw_values)

                # Build Top-3 area ranking for quick graph insights.
                ranking_pairs = []
                for label, value in zip(chart_labels, values):
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(numeric_value):
                        continue
                    ranking_pairs.append((str(label), numeric_value))

                ranking_pairs.sort(key=lambda item: item[1], reverse=True)
                top_areas = []
                for idx, (area_label, area_value) in enumerate(ranking_pairs[:3], start=1):
                    top_areas.append({
                        'rank': idx,
                        'area': area_label,
                        'value': round(float(area_value), 2)
                    })

                color = CRIME_COLORS[i % len(CRIME_COLORS)]
                chart_data = {
                    'labels': chart_labels,
                    'dataset': {
                        'label': columns[i],
                        'data': values,
                        'backgroundColor': color['bg'],
                        'borderColor': color['border'],
                        'borderWidth': 2,
                        'fill': True if chart_type == 'line' else False,
                        'tension': 0.4 if chart_type == 'line' else 0
                    },
                    'title': columns[i],
                    'chartType': chart_type,
                    'labelAxis': label_column or columns[0],
                    'top_areas': top_areas
                }
                charts.append(chart_data)
        
        return charts
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = normalize_text(request.form.get('username'))
        email = normalize_text(request.form.get('email'))
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))
        
        existing = user_exists(username=username, email=email)
        if existing['username']:
            flash('Username already exists!', 'error')
            return redirect(url_for('register'))
        if existing['email']:
            flash('Email already exists!', 'error')
            return redirect(url_for('register'))
        
        add_user(username, email, password)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        login_input = normalize_text(request.form.get('login') or request.form.get('username'))
        password = request.form.get('password') or ''

        if not login_input or not password:
            flash('Username/email and password are required!', 'error')
            return render_template('login.html')
        
        user = get_user_by_identifier(login_input)
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    crime_files = []
    crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    if os.path.exists(crime_dir):
        for filename in sorted(os.listdir(crime_dir)):
            if allowed_data_file(filename):
                crime_files.append({
                    'name': filename,
                    'path': f"/uploads/{current_user.id}/crime_data/{filename}"
                })
    
    # Get available models
    models_info = []
    user_models_dir = os.path.join(MODELS_FOLDER, current_user.id)
    if os.path.exists(user_models_dir):
        for filename in sorted(os.listdir(user_models_dir)):
            if filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                info_path = os.path.join(user_models_dir, f"{model_name}_info.json")
                info = {'name': model_name, 'file': filename}
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info.update(json.load(f))
                    except:
                        pass
                info['model_type'] = info.get('model_type', 'linear')
                info['model_type_name'] = info.get('model_type_name') or get_model_display_name(info['model_type'])
                info['accuracy_percent'] = estimate_accuracy_percent(info, key='accuracy')
                info['test_accuracy_percent'] = estimate_accuracy_percent(info, key='test_accuracy')
                info['feature_count'] = int(info.get('feature_count', 0) or 0)
                info['feature_columns'] = info.get('feature_columns', [])
                info['feature_importance'] = info.get('feature_importance', [])
                info['feature_defaults'] = info.get('feature_defaults', {}) if isinstance(info.get('feature_defaults'), dict) else {}
                info['default_feature_values'] = []
                for col in info['feature_columns']:
                    raw_default = info['feature_defaults'].get(col, 0)
                    try:
                        info['default_feature_values'].append(round(float(raw_default), 2))
                    except (TypeError, ValueError):
                        info['default_feature_values'].append(0)
                models_info.append(info)

    models_info.sort(key=lambda model: model.get('name', '').lower())
    prediction_history = get_recent_prediction_history(current_user.id, limit=20)
    
    available_models = get_available_models()

    return render_template(
        'dashboard.html',
        user=current_user,
        crime_files=crime_files,
        models=models_info,
        available_models=available_models,
        prediction_history=prediction_history
    )

@app.route('/upload_crime_data', methods=['POST'])
@login_required
def upload_crime_data():
    if 'crime_file' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['crime_file']
    
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    if file and allowed_data_file(file.filename):
        filename = secure_filename(file.filename)
        user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
        os.makedirs(user_crime_dir, exist_ok=True)
        file_path = os.path.join(user_crime_dir, filename)
        file.save(file_path)
        
        try:
            data = read_crime_data_file(file_path)
            analysis = analyze_crime_trends(file_path)
            if data:
                flash(f'Crime data "{filename}" uploaded! Analyze trends, train ML models, and predict future crime.', 'success')
            else:
                flash('Could not read data from file. Please check the format.', 'warning')
        except Exception as e:
            flash(f'Error reading file: {str(e)}', 'error')
    else:
        flash('File type not allowed! Please upload CSV or Excel files.', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/train_model', methods=['POST'])
@login_required
def train_model():
    """Train ML model for crime prediction"""
    filename = normalize_text(request.form.get('filename'))
    target_column = normalize_text(request.form.get('target_column'))
    model_name = secure_filename(normalize_text(request.form.get('model_name')))
    model_type = normalize_text(request.form.get('model_type', 'linear'))
    
    if not filename or not target_column or not model_name:
        flash('All fields are required!', 'error')
        return redirect(url_for('dashboard'))
    
    user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    file_path = os.path.join(user_crime_dir, filename)
    
    if not os.path.exists(file_path):
        flash('File not found!', 'error')
        return redirect(url_for('dashboard'))
    
    # Train the model
    model, result = train_crime_model(file_path, target_column, model_type)
    
    if model:
        # Save the model
        user_models_dir = os.path.join(MODELS_FOLDER, current_user.id)
        os.makedirs(user_models_dir, exist_ok=True)
        model_path = os.path.join(user_models_dir, f"{model_name}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model info
        info_path = os.path.join(user_models_dir, f"{model_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump(result, f)
        
        model_label = result.get('model_type_name') or get_model_display_name(result.get('model_type', 'linear'))
        train_percent = result.get('accuracy_percent', estimate_accuracy_percent(result, key='accuracy'))
        test_percent = result.get('test_accuracy_percent', estimate_accuracy_percent(result, key='test_accuracy'))
        flash(
            f'Model "{model_name}" ({model_label}) trained! '
            f'Train: {train_percent:.1f}%, Test: {test_percent:.1f}%, RMSE: {result["rmse"]:.2f}',
            'success'
        )
    else:
        flash(f'Error training model: {result}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Make single crime prediction"""
    model_name = normalize_text(request.form.get('model_name'))
    features = normalize_text(request.form.get('features'))
    
    if not model_name or not features:
        flash('All fields are required!', 'error')
        return redirect(url_for('dashboard'))
    
    feature_list, parse_error = parse_feature_list(features)
    if parse_error:
        flash(parse_error, 'error')
        return redirect(url_for('dashboard'))
    
    model_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}.pkl")
    info_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}_info.json")
    
    if not os.path.exists(model_path):
        flash('Model not found!', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)

        valid_count, validation_error = validate_feature_count(feature_list, info)
        if not valid_count:
            flash(validation_error, 'error')
            return redirect(url_for('dashboard'))
        
        prediction = predict_crime(model, feature_list, info.get('model_type', 'linear'))
        
        if prediction is not None:
            # Get historical statistics for risk assessment
            historical_mean, historical_std = get_risk_baseline(info)
            
            risk_level, risk_score, risk_description = calculate_risk_level(prediction, historical_mean, historical_std)
            confidence = calculate_confidence(prediction, feature_list)
            residual_std = float(info.get('residual_std', 0) or 0)
            interval_text = ''
            if residual_std > 0:
                margin = 1.96 * residual_std
                lower = max(0, prediction - margin)
                upper = prediction + margin
                interval_text = f' | 95% Range: {lower:.2f} to {upper:.2f}'

            flash(
                f'Prediction: {prediction:.2f}{interval_text} | '
                f'Risk Level: {risk_level} ({risk_score}%) - {risk_description}',
                'success'
            )

            log_prediction_event(
                user_id=current_user.id,
                model_name=model_name,
                algorithm=info.get('model_type_name') or get_model_display_name(info.get('model_type', 'linear')),
                prediction_type='single',
                prediction_value=prediction,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence=confidence,
                feature_values=feature_list,
                notes='Single prediction'
            )
        else:
            flash('Error making prediction!', 'error')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/multi_step_predict', methods=['POST'])
@login_required
def multi_step_predict():
    """Multi-step crime prediction (like in research paper)"""
    model_name = normalize_text(request.form.get('model_name'))
    features = normalize_text(request.form.get('features'))
    try:
        steps = max(1, min(30, int(request.form.get('steps', 7))))
    except (TypeError, ValueError):
        flash('Days to predict must be a valid number.', 'error')
        return redirect(url_for('dashboard'))
    
    if not model_name or not features:
        flash('All fields are required!', 'error')
        return redirect(url_for('dashboard'))
    
    feature_list, parse_error = parse_feature_list(features)
    if parse_error:
        flash(parse_error, 'error')
        return redirect(url_for('dashboard'))
    
    model_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}.pkl")
    info_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}_info.json")
    
    if not os.path.exists(model_path):
        flash('Model not found!', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)

        valid_count, validation_error = validate_feature_count(feature_list, info)
        if not valid_count:
            flash(validation_error, 'error')
            return redirect(url_for('dashboard'))
        
        # Keep the same feature vector width the model was trained with.
        historical_data = feature_list.copy()
        
        predictions = multi_step_prediction(model, historical_data, steps, info.get('model_type', 'linear'))
        
        if predictions:
            historical_mean, historical_std = get_risk_baseline(info)
            # Store predictions in session for display
            predictions_summary = "\n".join([f"Day {p['step']}: {p['prediction']} (Confidence: {p['confidence']}%)" for p in predictions])
            flash(f'Multi-step predictions:\n{predictions_summary}', 'success')

            for p in predictions:
                risk_level, risk_score, _ = calculate_risk_level(p['prediction'], historical_mean, historical_std)
                log_prediction_event(
                    user_id=current_user.id,
                    model_name=model_name,
                    algorithm=info.get('model_type_name') or get_model_display_name(info.get('model_type', 'linear')),
                    prediction_type=f'multi_step_day_{p["step"]}',
                    prediction_value=p['prediction'],
                    risk_level=risk_level,
                    risk_score=risk_score,
                    confidence=p.get('confidence'),
                    feature_values=feature_list,
                    notes=f'Step {p["step"]} of {steps}'
                )
        else:
            flash('Error making predictions!', 'error')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/get_columns/<filename>')
@login_required
def get_columns(filename):
    """Get column names from a crime data file"""
    user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    file_path = os.path.join(user_crime_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        df = load_tabular_data(file_path)
        
        columns = []
        
        # Get statistics for each column
        stats = {}
        for col in df.columns:
            numeric_col = clean_numeric_series(df[col])
            if numeric_col.notna().sum() < 2:
                continue
            columns.append(col)
            stats[col] = {
                'mean': float(numeric_col.mean()),
                'max': float(numeric_col.max()),
                'min': float(numeric_col.min()),
                'std': float(numeric_col.std() if not np.isnan(numeric_col.std()) else 0)
            }
        
        return jsonify({'columns': columns, 'statistics': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_crime_chart/<filename>')
@login_required
def get_crime_chart(filename):
    chart_type = request.args.get('chart_type', 'bar')
    user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    file_path = os.path.join(user_crime_dir, filename)
    
    if not os.path.commonpath([user_crime_dir, file_path]).startswith(user_crime_dir):
        return jsonify({'error': 'Access denied'}), 403
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    data = read_crime_data_file(file_path, chart_type)
    
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Could not read data'}), 400

@app.route('/analyze_trends/<filename>')
@login_required
def analyze_trends(filename):
    """Get crime trend analysis"""
    user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    file_path = os.path.join(user_crime_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    analysis = analyze_crime_trends(file_path)
    
    if analysis:
        return jsonify(analysis)
    else:
        return jsonify({'error': 'Could not analyze data'}), 400

@app.route('/get_dashboard_stats')
@login_required
def get_dashboard_stats():
    """Get dashboard statistics for the stats section"""
    try:
        stats = {
            'total_records': 0,
            'high_risk_areas': 0,
            'ml_models': 0,
            'avg_accuracy': 0
        }
        
        # Get total records and high risk areas from crime data files
        user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
        if os.path.exists(user_crime_dir):
            for filename in os.listdir(user_crime_dir):
                if allowed_data_file(filename):
                    file_path = os.path.join(user_crime_dir, filename)
                    try:
                        df = load_tabular_data(file_path)
                        
                        stats['total_records'] += int(len(df))
                        
                        # Count high risk columns from cleaned numeric series.
                        for col in df.columns:
                            numeric_col = clean_numeric_series(df[col])
                            if numeric_col.notna().sum() < 2:
                                continue
                            if float(numeric_col.mean()) > 50:
                                stats['high_risk_areas'] += 1
                    except:
                        pass
        
        # Get model count and average accuracy
        user_models_dir = os.path.join(MODELS_FOLDER, current_user.id)
        if os.path.exists(user_models_dir):
            model_files = [f for f in os.listdir(user_models_dir) if f.endswith('_info.json')]
            stats['ml_models'] = len(model_files)
            
            # Calculate average accuracy
            accuracy_scores = []
            for model_file in model_files:
                info_path = os.path.join(user_models_dir, model_file)
                try:
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                        accuracy_scores.append(estimate_accuracy_percent(model_info, key='test_accuracy'))
                except:
                    pass
            
            if accuracy_scores:
                stats['avg_accuracy'] = round(sum(accuracy_scores) / len(accuracy_scores), 1)
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export_predictions', methods=['POST'])
@login_required
def export_predictions():
    """Export predictions to CSV"""
    model_name = normalize_text(request.form.get('model_name'))
    features = normalize_text(request.form.get('features'))
    try:
        periods = max(1, min(120, int(request.form.get('periods', 12))))
    except (TypeError, ValueError):
        flash('Periods must be a valid number.', 'error')
        return redirect(url_for('dashboard'))
    
    if not model_name or not features:
        flash('All fields are required!', 'error')
        return redirect(url_for('dashboard'))
    
    feature_list, parse_error = parse_feature_list(features)
    if parse_error:
        flash(parse_error, 'error')
        return redirect(url_for('dashboard'))
    
    model_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        flash('Model not found!', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        info_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}_info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)

        valid_count, validation_error = validate_feature_count(feature_list, info)
        if not valid_count:
            flash(validation_error, 'error')
            return redirect(url_for('dashboard'))
        
        # Generate predictions for multiple periods
        predictions = []
        current_features = feature_list.copy()
        historical_mean, historical_std = get_risk_baseline(info)
        
        for i in range(periods):
            pred = predict_crime(model, current_features, info.get('model_type', 'linear'))
            if pred is not None:
                # Calculate risk
                risk_level, _, _ = calculate_risk_level(pred, historical_mean, historical_std)
                
                predictions.append({
                    'period': i + 1,
                    'prediction': round(pred, 2),
                    'risk_level': risk_level
                })
                current_features = current_features[1:] + [pred]
            else:
                break
        
        # Create CSV response
        csv_data = "Period,Prediction,Risk Level\n"
        for p in predictions:
            csv_data += f"{p['period']},{p['prediction']},{p['risk_level']}\n"
        
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = f'attachment; filename={model_name}_predictions.csv'
        response.headers['Content-Type'] = 'text/csv'

        if predictions:
            log_prediction_event(
                user_id=current_user.id,
                model_name=model_name,
                algorithm=info.get('model_type_name') or get_model_display_name(info.get('model_type', 'linear')),
                prediction_type='export_batch',
                prediction_value=predictions[-1]['prediction'],
                risk_level=predictions[-1]['risk_level'],
                risk_score=0,
                confidence=None,
                feature_values=feature_list,
                notes=f'Exported {len(predictions)} periods'
            )

        return response
        
    except Exception as e:
        flash(f'Error exporting predictions: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/get_model_insights/<model_name>')
@login_required
def get_model_insights(model_name):
    info_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}_info.json")
    if not os.path.exists(info_path):
        return jsonify({'error': 'Model insights not found'}), 404

    try:
        with open(info_path, 'r') as f:
            info = json.load(f)

        return jsonify({
            'model_name': model_name,
            'algorithm': info.get('model_type_name') or get_model_display_name(info.get('model_type', 'linear')),
            'target_column': info.get('target_column'),
            'training_samples': info.get('training_samples'),
            'feature_count': info.get('feature_count'),
            'feature_importance': info.get('feature_importance', []),
            'top_features': info.get('top_features', []),
            'accuracy': info.get('accuracy'),
            'test_accuracy': info.get('test_accuracy'),
            'accuracy_percent': info.get('accuracy_percent', estimate_accuracy_percent(info, key='accuracy')),
            'test_accuracy_percent': info.get('test_accuracy_percent', estimate_accuracy_percent(info, key='test_accuracy')),
            'rmse': info.get('rmse')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export_prediction_history')
@login_required
def export_prediction_history():
    history_file = get_user_history_file(current_user.id)
    if not os.path.exists(history_file):
        flash('No prediction history available yet.', 'warning')
        return redirect(url_for('dashboard'))

    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            csv_data = f.read()

        response = make_response(csv_data)
        response.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response
    except Exception as e:
        flash(f'Error exporting history: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/uploads/<user_id>/<filename>')
@login_required
def uploaded_file(user_id, filename):
    if current_user.id != user_id:
        flash('Access denied!', 'error')
        return redirect(url_for('dashboard'))
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], user_id), filename)

@app.route('/delete_crime_file/<filename>')
@login_required
def delete_crime_file(filename):
    user_crime_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id, 'crime_data')
    file_path = os.path.join(user_crime_dir, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'Crime data "{filename}" deleted successfully!', 'success')
    else:
        flash('File not found!', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/delete_model/<model_name>')
@login_required
def delete_model(model_name):
    model_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}.pkl")
    info_path = os.path.join(MODELS_FOLDER, current_user.id, f"{model_name}_info.json")
    
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(info_path):
        os.remove(info_path)
    
    flash(f'Model "{model_name}" deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_csv_files()
    app.run(debug=True)


