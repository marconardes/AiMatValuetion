import pytest
import os
import json
import csv
import joblib
import yaml
import pandas as pd # For reading CSV to verify
from unittest.mock import patch, MagicMock

# Import main functions from the scripts
from fetch_mp_data import fetch_data
from process_raw_data import process_data
from train_model import train_models

from utils.config_loader import load_config # To verify config loading if needed, though test will write its own

# --- Helper Mock Classes (similar to test_fetch_mp_data.py) ---

class MockSummaryDoc:
    def __init__(self, material_id, formula_pretty, nelements, band_gap, formation_energy_per_atom):
        self.material_id = material_id
        self.formula_pretty = formula_pretty
        self.nelements = nelements
        self.band_gap = band_gap
        self.formation_energy_per_atom = formation_energy_per_atom

class MockStructure:
    def __init__(self, material_id="mp-test"):
        self.material_id = material_id
        self.formula = "FeMock" if "1" in material_id else "NiMock" # Vary formula for different materials

        self.composition = MagicMock()
        self.composition.reduced_formula = self.formula
        self.composition.elements = [MagicMock(symbol=self.formula.replace("Mock",""))]

        self.density = 7.87 if "Fe" in self.formula else 8.90
        self.volume = 11.82 if "Fe" in self.formula else 10.94
        self.num_sites = 1

        self.lattice = MagicMock(spec=True)
        self.lattice.a = 2.866 if "Fe" in self.formula else 3.52
        self.lattice.b = self.lattice.a
        self.lattice.c = self.lattice.a
        self.lattice.alpha, self.lattice.beta, self.lattice.gamma = 90.0, 90.0, 90.0

        self.get_space_group_info = MagicMock(return_value=("Im-3m", 229) if "Fe" in self.formula else ("Fm-3m", 225))
        self.get_crystal_system = MagicMock(return_value="cubic")

    def to(self, fmt="cif", **kwargs):
        if fmt == "cif":
            symbol = self.formula.replace("Mock","")
            return f"""
data_{self.material_id}
_cell_length_a                        {self.lattice.a}
_cell_length_b                        {self.lattice.b}
_cell_length_c                        {self.lattice.c}
_cell_angle_alpha                     {self.lattice.alpha}
_cell_angle_beta                      {self.lattice.beta}
_cell_angle_gamma                     {self.lattice.gamma}
_symmetry_space_group_name_H-M        'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{symbol}1 {symbol} 0.0 0.0 0.0
"""
        raise NotImplementedError

class MockDos:
    def __init__(self, material_id="mp-test"):
        self.material_id = material_id
        self.efermi = 0.5 if "1" in material_id else 0.6
        self.energies = [-1, self.efermi, 2]
        self._dos_data = {"efermi": self.efermi, "energies": self.energies, "densities": {"1": [1,2,3]}}
        self.get_dos_at_fermi = MagicMock(return_value=2.0 if "1" in material_id else 2.5)

    def as_dict(self):
        return self._dos_data

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def integration_test_paths(tmp_path_factory):
    """Creates a temporary directory for all integration test files."""
    temp_dir = tmp_path_factory.mktemp("integration_data")
    return {
        "config_file": temp_dir / "test_integration_config.yml",
        "raw_data_json": temp_dir / "integration_raw_data.json",
        "processed_csv": temp_dir / "integration_processed_dataset.csv",
        "model_dir": temp_dir / "models",
    }

@pytest.fixture(scope="module")
def integration_test_config(integration_test_paths):
    """Generates the configuration dictionary for the integration test."""
    model_dir = integration_test_paths["model_dir"]
    model_dir.mkdir(exist_ok=True)

    config_dict = {
        "mp_api_key": "MOCK_INTEGRATION_API_KEY",
        "fetch_data": {
            "max_total_materials": 2, # Fetch 2 materials for better train/test split
            "output_filename": str(integration_test_paths["raw_data_json"]),
            "criteria_sets": [
                {"target_n_elements": 1, "limit_per_set": 2, "description": "integration test mono"},
            ]
        },
        "process_data": {
            "raw_data_filename": str(integration_test_paths["raw_data_json"]),
            "output_filename": str(integration_test_paths["processed_csv"])
        },
        "train_model": {
            "dataset_filename": str(integration_test_paths["processed_csv"]),
            "test_size": 0.5, # Ensure at least 1 sample in train and test for 2 total samples
            "random_state": 42,
            "n_estimators": 1,
            "models": {
                "dos_at_fermi": str(model_dir / "int_model_dos.joblib"),
                "target_band_gap": str(model_dir / "int_model_band_gap.joblib"),
                "target_formation_energy": str(model_dir / "int_model_formation_energy.joblib"),
                "target_is_metal": str(model_dir / "int_model_is_metal.joblib")
            },
            "preprocessors": {
                "main": str(model_dir / "int_preprocessor_main.joblib"),
                "dos_at_fermi": str(model_dir / "int_preprocessor_dos.joblib")
            }
        },
        "gui": { # Add a gui section in case any part of the tested code tries to access it
            "title": "Integration Test GUI",
            "geometry": "100x100",
            "models_to_load": {}, # Keep empty as GUI part not tested
            "manual_entry_csv_filename": str(integration_test_paths["processed_csv"]) # Or a different dummy
        }
    }
    # Write this config to the designated config file path
    with open(integration_test_paths["config_file"], 'w') as f:
        yaml.dump(config_dict, f)
    return config_dict


@pytest.fixture
def mock_mp_rester_for_integration(monkeypatch):
    """Mocks MPRester for the integration test."""
    mock_mpr_instance = MagicMock(name="MockMPResterInstance")

    # Mock summary search to return two Fe-like elemental materials
    mock_summary_docs_list = [
        MockSummaryDoc("mp-int-1", "Fe", 1, 0.0, -0.1),
        MockSummaryDoc("mp-int-2", "Ni", 1, 0.0, -0.2) # Another elemental metal
    ]
    mock_mpr_instance.materials.summary.search.return_value = mock_summary_docs_list

    # Mock detail calls to handle both material_ids
    def get_structure_side_effect(material_id):
        if material_id == "mp-int-1": return MockStructure("mp-int-1")
        if material_id == "mp-int-2": return MockStructure("mp-int-2")
        return None
    mock_mpr_instance.get_structure_by_material_id.side_effect = get_structure_side_effect

    def get_dos_side_effect(material_id):
        if material_id == "mp-int-1": return MockDos("mp-int-1")
        if material_id == "mp-int-2": return MockDos("mp-int-2")
        return None
    mock_mpr_instance.get_dos_by_material_id.side_effect = get_dos_side_effect

    # Patch MPRester class
    # When MPRester is called, it returns a mock that, when used as a context manager,
    # returns our mock_mpr_instance.
    mock_MPRester_class = MagicMock(name="MockMPResterClass")
    mock_MPRester_class.return_value.__enter__.return_value = mock_mpr_instance

    monkeypatch.setattr("fetch_mp_data.MPRester", mock_MPRester_class)
    return mock_MPRester_class, mock_mpr_instance


# --- Integration Test ---

def test_data_pipeline_integration(integration_test_config, integration_test_paths, mock_mp_rester_for_integration, capsys):
    """
    Tests the data pipeline from fetching raw data, processing it, to training models.
    It uses the actual config file written by the integration_test_config fixture.
    """
    config_file_path = str(integration_test_paths["config_file"])

    # --- 1. Execute fetch_data ---
    # fetch_data internally calls load_config(config_path='config.yml') by default if utils.config_loader is used.
    # We need to make sure it loads OUR test config.
    # The load_config in utils/config_loader.py defaults to 'config.yml'.
    # So, we patch it to use our specific integration config file.
    with patch('utils.config_loader.DEFAULT_CONFIG_PATH', config_file_path):
        fetch_data() # Should use the config specified by DEFAULT_CONFIG_PATH patch

    # Assertions for fetch_data
    assert os.path.exists(integration_test_paths["raw_data_json"]), "Raw data JSON file was not created."
    with open(integration_test_paths["raw_data_json"], 'r') as f:
        raw_data = json.load(f)
        assert len(raw_data) == 2 # Should fetch 2 materials now
        assert raw_data[0]['material_id'] == "mp-int-1"
        assert raw_data[1]['material_id'] == "mp-int-2"

    # --- 2. Execute process_data ---
    def side_effect_structure_from_str(cif_string, fmt="cif"):
        if "mp-int-1" in cif_string:
            return MockStructure("mp-int-1")
        elif "mp-int-2" in cif_string:
            return MockStructure("mp-int-2")
        return MockStructure("mp-generic-error") # Should not happen in this test

    def side_effect_dos_from_dict(dos_dict_arg):
        # Based on the efermi value defined in MockDos for mp-int-1 and mp-int-2
        if dos_dict_arg and dos_dict_arg.get('efermi') == 0.5: # mp-int-1's DOS
            return MockDos("mp-int-1")
        elif dos_dict_arg and dos_dict_arg.get('efermi') == 0.6: # mp-int-2's DOS
            return MockDos("mp-int-2")
        return MockDos("mp-generic-error-dos")


    with patch('utils.config_loader.DEFAULT_CONFIG_PATH', config_file_path):
        with patch('process_raw_data.Structure.from_str', side_effect=side_effect_structure_from_str) as mock_struct_from_str_proc:
            with patch('process_raw_data.Dos.from_dict', side_effect=side_effect_dos_from_dict) as mock_dos_from_dict_proc:
                process_data()

    # Assertions for process_data
    assert os.path.exists(integration_test_paths["processed_csv"]), "Processed CSV file was not created."
    processed_df = pd.read_csv(integration_test_paths["processed_csv"])
    assert len(processed_df) == 2 # Expect 2 rows
    assert processed_df.iloc[0]['material_id'] == "mp-int-1"
    assert processed_df.iloc[1]['material_id'] == "mp-int-2"
    assert 'formula_pretty' in processed_df.columns
    assert processed_df.iloc[0]['formula_pretty'] == "FeMock"
    assert processed_df.iloc[1]['formula_pretty'] == "NiMock"


    # --- 3. Execute train_models ---
    with patch('utils.config_loader.DEFAULT_CONFIG_PATH', config_file_path):
        train_models()

    # Assertions for train_models
    model_config = integration_test_config['train_model']
    for model_file in model_config['models'].values():
        assert os.path.exists(model_file), f"Model file {model_file} was not created."
    for preprocessor_file in model_config['preprocessors'].values():
        assert os.path.exists(preprocessor_file), f"Preprocessor file {preprocessor_file} was not created."

    # Optionally, load a model to verify it's valid (simple check)
    try:
        loaded_model = joblib.load(model_config['models']['target_band_gap'])
        assert loaded_model is not None
    except Exception as e:
        pytest.fail(f"Failed to load a saved model: {e}")

    # Check for no obvious errors in stdout for the last step (train_models)
    captured = capsys.readouterr()
    assert "Error" not in captured.err # A very basic check on stderr
    assert "Traceback" not in captured.err
    assert "Model Training Completed" in captured.out

    print("Integration test completed. Files created in temporary directory.")
    print(f"Config file: {integration_test_paths['config_file']}")
    print(f"Raw JSON: {integration_test_paths['raw_data_json']}")
    print(f"Processed CSV: {integration_test_paths['processed_csv']}")
    print(f"Model dir: {integration_test_paths['model_dir']}")
