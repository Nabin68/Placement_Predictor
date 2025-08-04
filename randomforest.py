import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load dataset
Orginal_dataset = pd.read_csv("Placement_Data_Full_Class.csv")

# -------------------- Classification --------------------
dataset_cls = Orginal_dataset.drop(columns=["sl_no", "salary", "mba_p"])

cat_col_cls = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_p", "workex", "specialisation"]
num_col_cls = ["ssc_p", "hsc_p", "etest_p"]

# Pipelines
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, cat_col_cls),
    ("num", num_pipeline, num_col_cls)
])

# Model and Params
params = {
    "classification_model__n_estimators": [50, 100],
    "classification_model__max_samples": [0.5, 1.0],
    "classification_model__max_features": [0.5, 1.0]
}

x_cls = dataset_cls.drop(columns=["status"])
y_cls = LabelEncoder().fit_transform(dataset_cls["status"])
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(x_cls, y_cls, test_size=0.2, random_state=42)

clf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classification_model", RandomForestClassifier(random_state=42))
])

classification_model = GridSearchCV(
    clf_pipeline,
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

classification_model.fit(x_train_cls, y_train_cls)
y_pred_cls = classification_model.predict(x_test_cls)

print("\nClassification Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("Best Classifier Params:", classification_model.best_params_)

# Save classifier
joblib.dump(classification_model.best_estimator_, "classification_model.pkl")

# -------------------- Regression --------------------
dataset_reg = Orginal_dataset.drop(columns=["sl_no", "mba_p"]).copy()
dataset_reg = dataset_reg[dataset_reg["status"] == "Placed"]

cat_col_reg = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_p", "workex", "specialisation", "status"]
num_col_reg = ["ssc_p", "hsc_p", "etest_p"]

x_reg = dataset_reg.drop(columns=["salary"])
y_reg = dataset_reg["salary"]
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
reg_preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, cat_col_reg),
    ("num", num_pipeline, num_col_reg)
])

reg_pipeline = Pipeline([
    ("preprocessing", reg_preprocessor),
    ("regression_model", RandomForestRegressor(random_state=42))
])

params_reg = {
    "regression_model__n_estimators": [50, 100],
    "regression_model__max_samples": [0.5, 1.0],
    "regression_model__max_features": [0.5, 1.0]
}

regression_model = GridSearchCV(
    reg_pipeline,
    params_reg, 
    cv=5, 
    n_jobs=-1, 
    verbose=1
)

regression_model.fit(x_train_reg, y_train_reg)

y_pred_reg = regression_model.predict(x_test_reg)

print("\nRegression RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
print("Best Regressor Params:", regression_model.best_params_)

# Save regressor
joblib.dump(regression_model.best_estimator_, "regression_model.pkl")

# Classification Accuracy: 0.813953488372093
# Best Classifier Params: {'classification_model__max_features': 1.0, 'classification_model__max_samples': 1.0, 'classification_model__n_estimators': 50}
# Fitting 5 folds for each of 8 candidates, totalling 40 fits

# Regression RMSE: 90118.33494541126
# Best Regressor Params: {'regression_model__max_features': 0.5, 'regression_model__max_samples': 1.0, 'regression_model__n_estimators': 50}
