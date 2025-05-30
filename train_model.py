import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings # Added
from utils.config_loader import load_config # Changed

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Attempt to import DATA_SCHEMA or define relevant parts for feature identification
# For simplicity in this standalone script, we'll manually define feature lists.
# In a real project, this might come from a shared config or by loading fetch_mp_data.py schema.

def train_models():
    """
    Loads data, preprocesses it, and trains separate models for various material properties.
    Uses settings from config.yml.
    """
    full_config = load_config() # Use the new centralized loader
    if not full_config: # load_config returns {} on error or not found
        warnings.warn("Failed to load or parse config.yml for train_models. Using default script parameters.", UserWarning)
        train_config_params = {}
    else:
        train_config_params = full_config.get('train_model', {})

    csv_file = train_config_params.get('dataset_filename', 'Fe_materials_dataset.csv')
    default_test_size = 0.2
    default_random_state = 42
    default_n_estimators = 10 # From original logic when config was introduced

    test_size = train_config_params.get('test_size', default_test_size)
    random_state = train_config_params.get('random_state', default_random_state)
    n_estimators_config = train_config_params.get('n_estimators', default_n_estimators)


    # Model and preprocessor filenames from config
    default_model_filenames_map = { # Renamed for clarity
        "dos_at_fermi": "model_dos_at_fermi.joblib",
        "target_band_gap": "model_target_band_gap.joblib",
        "target_formation_energy": "model_target_formation_energy.joblib",
        "target_is_metal": "model_target_is_metal.joblib"
    }
    model_filenames_map = train_config_params.get('models', default_model_filenames_map)

    default_preprocessor_filenames_map = { # Renamed for clarity
        "main": "preprocessor_main.joblib",
        "dos_at_fermi": "preprocessor_dos_at_fermi.joblib"
    }
    preprocessor_filenames_map = train_config_params.get('preprocessors', default_preprocessor_filenames_map)

    # Specific filenames from loaded config or defaults
    preprocessor_main_file = preprocessor_filenames_map.get("main", "preprocessor_main.joblib")

    model_dos_file = model_filenames_map.get("dos_at_fermi", "model_dos_at_fermi.joblib")
    model_band_gap_file = model_filenames_map.get("target_band_gap", "model_target_band_gap.joblib")
    model_formation_energy_file = model_filenames_map.get("target_formation_energy", "model_target_formation_energy.joblib")
    model_is_metal_file = model_filenames_map.get("target_is_metal", "model_target_is_metal.joblib")


    if not os.path.exists(csv_file):
        print(f"Error: Dataset file '{csv_file}' not found. Please generate it first.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("Error: Dataset is empty.")
        return

    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- Define Features and Targets ---
    target_columns_reg = ['target_band_gap', 'target_formation_energy', 'target_dos_at_fermi']
    target_column_clf = 'target_is_metal'

    # Based on DATA_SCHEMA from fetch_mp_data.py (manually transcribed for this script)
    categorical_features = ['elements', 'crystal_system_pg']

    numerical_features = [
        'band_gap_mp', 'formation_energy_per_atom_mp', # These are also targets, but can be features if not the current target
        # Pymatgen derived features:
        'num_elements', 'density_pg', 'volume_pg', 'volume_per_atom_pg',
        'spacegroup_number_pg', 'lattice_a_pg', 'lattice_b_pg', 'lattice_c_pg',
        'lattice_alpha_pg', 'lattice_beta_pg', 'lattice_gamma_pg', 'num_sites_pg'
    ]
    # Exclude material_id and direct target columns from features when training for that target.
    # 'dos_at_fermi' (from _mp) is handled specially for its own model.

    # --- Preprocessing Setup ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # This preprocessor will be fitted specifically for each model or dataset subset
    # General preprocessor definition (to be cloned or refitted)
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like material_id) if X includes them, or 'drop'
    )

    # --- Train Regressor for target_dos_at_fermi (Metals Only) ---
    print("\n--- Training Model for DOS at Fermi (Metals Only) ---")
    df_dos = df[df['target_is_metal'] == True].copy()
    df_dos.dropna(subset=['target_dos_at_fermi'], inplace=True)

    if not df_dos.empty:
        # Features for DOS model (exclude target_dos_at_fermi if it's in numerical_features)
        numerical_features_dos = [f for f in numerical_features if f != 'target_dos_at_fermi' and f != 'dos_at_fermi']

        # Ensure 'is_metal' is not a feature for predicting 'dos_at_fermi' among metals
        if 'is_metal' in numerical_features_dos: numerical_features_dos.remove('is_metal')
        if 'is_metal' in categorical_features: categorical_features.remove('is_metal')


        X_dos = df_dos[numerical_features_dos + categorical_features]
        y_dos = df_dos['target_dos_at_fermi']

        if X_dos.empty or y_dos.empty:
            print("Not enough data to train DOS at Fermi model after filtering.")
        else:
            X_dos_train, X_dos_test, y_dos_train, y_dos_test = train_test_split(X_dos, y_dos, test_size=test_size, random_state=random_state)

            # Fit a new preprocessor specifically for this dataset subset
            # The config has a preprocessor_dos_at_fermi.joblib, implying the DOS preprocessor might be distinct
            # and should be saved. The original script embedded it in the pipeline.
            # For now, let's assume the pipeline handles the DOS preprocessor internally as before,
            # but if 'preprocessor_dos_at_fermi.joblib' is meant to be a standalone file for the GUI,
            # then this preprocessor_dos should be fitted and saved separately.
            # The config structure suggests the DOS preprocessor IS saved separately.
            preprocessor_dos_file = preprocessor_filenames_map.get("dos_at_fermi", "preprocessor_dos_at_fermi.joblib")

            preprocessor_dos = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features_dos),
                    ('cat', categorical_transformer, categorical_features)
                ], remainder='drop'
            )
            # Fit and save the DOS preprocessor
            print(f"Fitting DOS preprocessor for {X_dos_train.shape[0]} samples...")
            fitted_preprocessor_dos = preprocessor_dos.fit(X_dos_train)
            joblib.dump(fitted_preprocessor_dos, preprocessor_dos_file)
            print(f"Saved DOS preprocessor to {preprocessor_dos_file}")


            pipeline_dos = Pipeline(steps=[
                # The preprocessor is now fitted and saved, but pipeline still needs a preprocessor step.
                # We pass the *fitted* preprocessor instance.
                ('preprocessor', fitted_preprocessor_dos),
                ('regressor', RandomForestRegressor(random_state=random_state, n_estimators=n_estimators_config))
            ])

            print(f"Training DOS model on {X_dos_train.shape[0]} samples...")
            # Pipeline will use the already fitted preprocessor 'fitted_preprocessor_dos' for transform,
            # and then fit the regressor.
            pipeline_dos.fit(X_dos_train, y_dos_train) # Regressor is fit here. Preprocessor was already fit.

            # Evaluate
            y_pred_dos = pipeline_dos.predict(X_dos_test) # Uses transform from preprocessor, predict from regressor
            mae_dos = mean_absolute_error(y_dos_test, y_pred_dos)
            r2_dos = r2_score(y_dos_test, y_pred_dos)
            print(f"DOS Model (Metals Only) - MAE: {mae_dos:.4f}, R2: {r2_dos:.4f}")

            joblib.dump(pipeline_dos, model_dos_file) # Save the whole pipeline
            print(f"Saved DOS model pipeline to {model_dos_file}")

    else:
        print("No metallic samples with DOS data found to train the DOS model.")

    # --- Prepare Data for Main Models ---
    print("\n--- Preparing Data for Main Models (Band Gap, Formation Energy, Is Metal) ---")
    df_main = df.copy()
    # Drop rows where any of the primary targets are NaN
    main_target_cols = ['target_band_gap', 'target_formation_energy', 'target_is_metal']
    df_main.dropna(subset=main_target_cols, inplace=True)

    if df_main.empty:
        print("Error: No data available for main models after dropping NaNs in target columns.")
        return

    # Define features for main models (exclude all direct targets)
    numerical_features_main = [f for f in numerical_features if f not in target_columns_reg and f not in [target_column_clf, 'dos_at_fermi']]

    X_main = df_main[numerical_features_main + categorical_features]

    # Fit the main preprocessor
    preprocessor_main = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_main), # Use main numerical features
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop'
    )
    print(f"Fitting main preprocessor on {X_main.shape[0]} samples for main models...")
    # We need a train split of X_main to fit this preprocessor properly before using it for multiple models
    # If X_main is too small, train_test_split might behave unexpectedly or return fewer than 2 arrays.
    # We only need X_main_temp_train for fitting the preprocessor.
    if X_main.shape[0] > 1: # Ensure there's enough data to split
        split_result = train_test_split(X_main, df_main[main_target_cols[0]], test_size=test_size, random_state=random_state)
        X_main_temp_train = split_result[0]
    else: # Not enough data to split, use X_main as is for fitting preprocessor
        X_main_temp_train = X_main.copy() # Use a copy to be safe

    preprocessor_main_fitted = preprocessor_main.fit(X_main_temp_train)
    joblib.dump(preprocessor_main_fitted, preprocessor_main_file)
    print(f"Saved main preprocessor to {preprocessor_main_file}")

    # --- Train Regressors (Band Gap, Formation Energy) ---
    # These models will use the main_preprocessor_fitted and then their own regressor.
    # The saved model will be just the regressor, not the full pipeline,
    # as the GUI expects to load preprocessor and model separately for these.

    reg_targets_map = {
        "target_band_gap": model_band_gap_file,
        "target_formation_energy": model_formation_energy_file
    }

    for target_name, model_file in reg_targets_map.items():
        print(f"\n--- Training Regressor for {target_name} ---")
        y_reg = df_main[target_name]

        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_main, y_reg, test_size=test_size, random_state=random_state)

        # Transform data using the already fitted main preprocessor
        X_reg_train_processed = preprocessor_main_fitted.transform(X_reg_train)
        X_reg_test_processed = preprocessor_main_fitted.transform(X_reg_test)

        regressor_model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators_config)

        print(f"Training {target_name} model on {X_reg_train_processed.shape[0]} samples...")
        regressor_model.fit(X_reg_train_processed, y_reg_train)

        y_pred_reg = regressor_model.predict(X_reg_test_processed)
        mae_reg = mean_absolute_error(y_reg_test, y_pred_reg)
        r2_reg = r2_score(y_reg_test, y_pred_reg)
        print(f"{target_name} Model - MAE: {mae_reg:.4f}, R2: {r2_reg:.4f}")

        joblib.dump(regressor_model, model_file)
        print(f"Saved {target_name} model to {model_file}")

    # --- Train Classifier (is_metal) ---
    # This model also uses main_preprocessor_fitted and then its own classifier.
    print(f"\n--- Training Classifier for {target_column_clf} ---")
    y_clf = df_main[target_column_clf].astype(int) # Ensure boolean is int

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_main, y_clf, test_size=test_size, random_state=random_state)

    # Transform data using the already fitted main preprocessor
    X_clf_train_processed = preprocessor_main_fitted.transform(X_clf_train)
    X_clf_test_processed = preprocessor_main_fitted.transform(X_clf_test)

    classifier_model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators_config)

    print(f"Training {target_column_clf} model on {X_clf_train_processed.shape[0]} samples...")
    classifier_model.fit(X_clf_train_processed, y_clf_train)

    y_pred_clf = classifier_model.predict(X_clf_test_processed)
    accuracy_clf = accuracy_score(y_clf_test, y_pred_clf)
    f1_clf = f1_score(y_clf_test, y_pred_clf, average='weighted')
    print(f"{target_column_clf} Model - Accuracy: {accuracy_clf:.4f}, F1 Score: {f1_clf:.4f}")

    joblib.dump(classifier_model, model_is_metal_file)
    print(f"Saved {target_column_clf} model to {model_is_metal_file}")

    print("\n--- Model Training Completed ---")

if __name__ == "__main__":
    train_models()
