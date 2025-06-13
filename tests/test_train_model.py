import pytest
import pandas as pd
import numpy as np
import os
import joblib
from unittest.mock import patch, MagicMock, mock_open, call

# Module to be tested
from scripts.train_model import train_models # Assuming train_model.py is in the root
from utils.schema import DATA_SCHEMA # To help structure mock data

# --- Test Data and Fixtures ---

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame similar to Fe_materials_dataset.csv."""
    data = {
        # Pymatgen-derived features (numerical)
        'num_elements': [2, 3, 2, 1, 3, 2],
        'density_pg': [5.0, 4.5, 5.2, 7.8, 4.0, 6.0],
        'volume_pg': [20.0, 30.0, 22.0, 10.0, 35.0, 25.0],
        'volume_per_atom_pg': [10.0, 10.0, 11.0, 10.0, 11.67, 12.5],
        'spacegroup_number_pg': [225, 12, 225, 229, 166, 62],
        'lattice_a_pg': [3.0, 4.0, 3.1, 2.8, 4.2, 3.5],
        'lattice_b_pg': [3.0, 4.0, 3.1, 2.8, 4.2, 3.8],
        'lattice_c_pg': [3.0, 5.0, 3.1, 2.8, 4.5, 4.0],
        'lattice_alpha_pg': [90, 90, 90, 90, 90, 90],
        'lattice_beta_pg': [90, 90, 90, 90, 90, 90],
        'lattice_gamma_pg': [90, 120, 90, 90, 120, 90],
        'num_sites_pg': [2, 3, 2, 1, 3, 2],
        # Pymatgen-derived features (categorical)
        'elements': ['Fe,O', 'Fe,Si,O', 'Fe,Ni', 'Fe', 'Al,Fe,O', 'Fe,S'],
        'crystal_system_pg': ['cubic', 'hexagonal', 'cubic', 'cubic', 'trigonal', 'orthorhombic'],
        # MP-derived features (can also be targets)
        'band_gap_mp': [0.0, 1.5, 0.0, 0.0, 2.0, 0.2], # two metals, one insulator, one narrow gap
        'formation_energy_per_atom_mp': [-0.5, -1.0, -0.3, 0.0, -1.2, -0.1],
        # Target properties (these should align with some of the _mp versions or be derived)
        'target_band_gap': [0.0, 1.5, 0.0, 0.0, 2.0, 0.2],
        'target_formation_energy': [-0.5, -1.0, -0.3, 0.0, -1.2, -0.1],
        'target_is_metal': [True, False, True, True, False, True], # Converted to int in script
        'target_dos_at_fermi': [10.5, np.nan, 8.2, 12.0, np.nan, 5.0], # NaN for insulators
        # Material ID (not a feature, but good to have)
        'material_id': ['mp-m1', 'mp-i1', 'mp-m2', 'mp-m3', 'mp-i2', 'mp-m4']
    }
    df = pd.DataFrame(data)

    # Add some NaNs to numerical features to test imputation
    df.loc[0, 'density_pg'] = np.nan
    df.loc[1, 'volume_pg'] = np.nan
    return df

@pytest.fixture
def mock_input_csv_path(tmp_path, sample_dataframe):
    """Creates a dummy CSV file from the sample_dataframe and returns its path."""
    csv_path = tmp_path / "dummy_dataset.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def mock_config_for_training(mock_input_csv_path, tmp_path):
    """Provides a mock configuration dictionary for training."""
    # Define a temporary directory for saving models, within tmp_path for cleanup
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir()

    return {
        "train_model": {
            "dataset_filename": mock_input_csv_path,
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 5, # Keep low for fast tests
            "models": {
                "dos_at_fermi": str(model_save_dir / "test_model_dos_at_fermi.joblib"),
                "target_band_gap": str(model_save_dir / "test_model_target_band_gap.joblib"),
                "target_formation_energy": str(model_save_dir / "test_model_target_formation_energy.joblib"),
                "target_is_metal": str(model_save_dir / "test_model_target_is_metal.joblib")
            },
            "preprocessors": {
                "main": str(model_save_dir / "test_preprocessor_main.joblib"),
                "dos_at_fermi": str(model_save_dir / "test_preprocessor_dos_at_fermi.joblib")
            }
        }
    }

# --- Test Cases ---

@patch('scripts.train_model.joblib.dump')
@patch('scripts.train_model.RandomForestRegressor')
@patch('scripts.train_model.RandomForestClassifier')
def test_train_models_successful_execution(mock_classifier, mock_regressor, mock_joblib_dump,
                                           mock_config_for_training, sample_dataframe, capsys):
    """
    Test the successful execution of the train_models pipeline.
    Focuses on calls to model fitting and saving, not on model performance.
    """
    # Configure mocks for estimators
    # Each time RandomForestRegressor/Classifier is called, it returns a new mock instance
    # These instances will have their 'fit' and 'predict' methods as MagicMock objects by default.

    def mock_estimator_instance(*args, **kwargs):
        instance = MagicMock()

        # Make predict return an array of appropriate length based on input X
        def mock_predict(X_input):
            return np.zeros(X_input.shape[0])
        instance.predict.side_effect = mock_predict

        # Make predict_proba return an array of appropriate shape based on input X
        def mock_predict_proba(X_input):
            return np.zeros((X_input.shape[0], 2)) # Assuming 2 classes for simplicity
        instance.predict_proba.side_effect = mock_predict_proba

        return instance

    mock_regressor.side_effect = mock_estimator_instance
    mock_classifier.side_effect = mock_estimator_instance


    # Mock pd.read_csv to return our sample DataFrame
    with patch('scripts.train_model.pd.read_csv', return_value=sample_dataframe):
        # Mock load_config to return our controlled config
        with patch('scripts.train_model.load_config', return_value=mock_config_for_training):
            train_models()

            # --- Assertions for preprocessor saving ---
            preprocessor_paths = mock_config_for_training['train_model']['preprocessors']
            # Check main preprocessor was saved
            # Call for main preprocessor: joblib.dump(preprocessor_main_fitted, preprocessor_main_file)
            # Call for DOS preprocessor: joblib.dump(fitted_preprocessor_dos, preprocessor_dos_config_name)

            # We need to check that joblib.dump was called with these paths.
            # The first argument to dump will be the actual preprocessor object.
            # We can check the second argument (filename string).

            # Get all filenames passed to joblib.dump
            dumped_filenames = [call_args[0][1] for call_args in mock_joblib_dump.call_args_list]

            assert preprocessor_paths['main'] in dumped_filenames
            assert preprocessor_paths['dos_at_fermi'] in dumped_filenames

            # --- Assertions for model saving ---
            model_paths = mock_config_for_training['train_model']['models']
            assert model_paths['dos_at_fermi'] in dumped_filenames
            assert model_paths['target_band_gap'] in dumped_filenames
            assert model_paths['target_formation_energy'] in dumped_filenames
            assert model_paths['target_is_metal'] in dumped_filenames

            # Total dumps: 2 preprocessors + 4 models
            assert mock_joblib_dump.call_count == 6

            # --- Assertions for model fitting ---
            # RandomForestRegressor should be called 3 times (DOS, band_gap, formation_energy)
            # RandomForestClassifier should be called 1 time (is_metal)
            # The current train_model.py structure saves the fitted regressor/classifier directly, not the pipeline
            # for band_gap, formation_energy, and is_metal models.
            # The DOS model saves the entire pipeline.

            # Check fit calls on the regressor instances
            # For DOS model (pipeline), the regressor is inside the pipeline.
            # For other regressors, they are standalone.
            # mock_rf_regressor_instance.fit.call_count should reflect direct calls.
            # The pipeline_dos.fit will call fit on its internal regressor.

            # Total calls to RandomForestRegressor().fit()
            # One for pipeline_dos's regressor, one for target_band_gap, one for target_formation_energy.
            # The mock_rf_regressor_instance is reused by the mock framework if not careful.
            # Let's check the total number of fit calls made by ANY RandomForestRegressor.
            # This requires a more sophisticated mock setup if we want to distinguish.
            # --- Assertions for model fitting ---
            # RandomForestRegressor class should be instantiated 3 times.
            # RandomForestClassifier class should be instantiated 1 time.
            assert mock_regressor.call_count == 3
            assert mock_classifier.call_count == 1

            # Each of these instances should have its 'fit' method called once.
            # To check this, we need to inspect the instances created by the side_effect.
            # This is tricky because the instances are created inside train_models.
            # A simpler check is that the .fit was called on the *last* instance,
            # or sum of calls if the mock object was the same one returned.
            # Since side_effect creates new mocks, we can't directly use mock_rf_regressor_instance.fit.call_count.

            # However, we can check the number of calls to the 'fit' method of the *mocked class's return_value*
            # This is a bit indirect. A more robust way is to check calls to the mock instances.
            # For now, let's assert that `fit` was called on *some* instance that was returned by the class mock.
            # This means at least one of the .fit() methods on the mocked instances was called.
            # This is implicitly tested by the fact that the script runs to completion and tries to save models.

            # A better check for fit calls would be to get all instances from side_effect if possible,
            # or to verify that the print statements for MAE/R2/Accuracy (which happen after fit/predict) are present.
            captured = capsys.readouterr()
            assert "Training DOS model on" in captured.out
            assert "DOS Model (Metals Only) - MAE:" in captured.out # Implies fit and predict were called
            assert "Training target_band_gap model on" in captured.out
            assert "target_band_gap Model - MAE:" in captured.out
            assert "Training target_formation_energy model on" in captured.out
            assert "target_formation_energy Model - MAE:" in captured.out
            assert "Training target_is_metal model on" in captured.out
            assert "target_is_metal Model - Accuracy:" in captured.out
            assert "Model Training Completed" in captured.out


def test_train_models_missing_input_file(mock_config_for_training, capsys):
    """Test behavior when the input CSV file is missing."""
    config_val = mock_config_for_training.copy()
    config_val['train_model']['dataset_filename'] = "non_existent_dataset.csv"

    with patch('scripts.train_model.load_config', return_value=config_val):
        with patch('os.path.exists', return_value=False) as mock_os_exists:
            train_models()
            mock_os_exists.assert_called_with("non_existent_dataset.csv")
            captured = capsys.readouterr()
            assert "Error: Dataset file 'non_existent_dataset.csv' not found." in captured.out
            assert "Model Training Completed" not in captured.out


def test_train_models_empty_input_dataframe(mock_config_for_training, capsys):
    """Test behavior when the input DataFrame is empty."""
    empty_df = pd.DataFrame()
    with patch('scripts.train_model.load_config', return_value=mock_config_for_training):
        with patch('scripts.train_model.pd.read_csv', return_value=empty_df):
            with patch('os.path.exists', return_value=True): # Ensure it passes the file check
                train_models()
                captured = capsys.readouterr()
                assert "Error: Dataset is empty." in captured.out # Adjusted to match current error message in train_model.py
                assert "Model Training Completed" not in captured.out

@patch('scripts.train_model.joblib.dump')
@patch('scripts.train_model.RandomForestRegressor') # Mock the class
@patch('scripts.train_model.RandomForestClassifier') # Mock the class
def test_dos_model_training_path(mock_classifier_cls, mock_regressor_cls, mock_joblib_dump,
                                 mock_config_for_training, sample_dataframe, capsys):
    """Test the specific path for training the DOS model (metals only)."""

    # Setup mocks for estimator instances, similar to test_train_models_successful_execution
    def mock_estimator_instance_for_dos_test(*args, **kwargs):
        instance = MagicMock()
        def mock_predict(X_input): # Corrected indentation
            return np.zeros(X_input.shape[0])
        instance.predict.side_effect = mock_predict
        # DOS model is a regressor, so no predict_proba needed here for it.
        return instance

    mock_regressor_cls.side_effect = mock_estimator_instance_for_dos_test

    # We still need to mock the classifier for the other parts of train_models to run
    def mock_classifier_instance_for_dos_test(*args, **kwargs): # Corrected indentation
        instance = MagicMock()
        def mock_predict_clf(X_input): # Corrected indentation
            return np.zeros(X_input.shape[0])
        instance.predict.side_effect = mock_predict_clf
        def mock_predict_proba_clf(X_input): # Corrected indentation
            return np.zeros((X_input.shape[0], 2))
        instance.predict_proba.side_effect = mock_predict_proba_clf
        return instance
    mock_classifier_cls.side_effect = mock_classifier_instance_for_dos_test


    with patch('scripts.train_model.pd.read_csv', return_value=sample_dataframe):
        with patch('scripts.train_model.load_config', return_value=mock_config_for_training):
            # We are NOT mocking Pipeline.fit here, so it will execute
            train_models()

            # Verify that the DOS model and its preprocessor were saved
            dumped_filenames = [call_args[0][1] for call_args in mock_joblib_dump.call_args_list]
            model_paths = mock_config_for_training['train_model']['models']
            preprocessor_paths = mock_config_for_training['train_model']['preprocessors']

            assert model_paths['dos_at_fermi'] in dumped_filenames
            assert preprocessor_paths['dos_at_fermi'] in dumped_filenames

            # Check that the regressor for DOS model was fit.
            # The DOS model pipeline uses the first instance of RandomForestRegressor.
            # The mock_regressor_cls was called 3 times in total (DOS, band_gap, formation_energy).
            # Its first instance's .fit method should have been called.

            # This relies on the order of instantiation if using a single mock_reg_instance.
            # If side_effect is a list of mocks, it's easier.
            # For now, check if any of the mocked regressor instances had fit called.
            # And that the printout for DOS model training occurred.

            # Check if the fit method of the instance returned by RandomForestRegressor for the DOS model was called.
            # This is implicitly confirmed if the "DOS Model (Metals Only) - MAE:" printout exists,
            # as that requires fit and predict.
            captured = capsys.readouterr()
            assert "Training DOS model on 3 samples..." in captured.out # 4 metals, 3 after train/test split for DOS
            assert "DOS Model (Metals Only) - MAE:" in captured.out

# TODO: Add tests for feature selection/preprocessing logic if it becomes more complex or configurable.
