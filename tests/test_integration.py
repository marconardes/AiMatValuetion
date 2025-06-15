import pytest
import os
import json
import csv
import joblib
import yaml
import pandas as pd # For reading CSV to verify
from unittest.mock import patch, MagicMock

# Import main functions from the scripts
from scripts.fetch_mp_data import fetch_data
from scripts.process_raw_data import process_data
from scripts.train_model import train_models

from utils.config_loader import load_config # To verify config loading if needed, though test will write its own

# --- Helper Mock Classes (defined locally for this test's clarity) ---
class MinimalMockSummaryDoc:
    def __init__(self, material_id, formula_pretty, nelements, band_gap_mp, formation_energy_per_atom_mp, cif_string_mp, **kwargs): # Changed keys
        self.material_id = material_id
        self.formula_pretty = formula_pretty
        self.nelements = nelements # Used by script
        self.band_gap_mp = band_gap_mp # Changed key
        self.formation_energy_per_atom_mp = formation_energy_per_atom_mp # Changed key
        self.cif_string_mp = cif_string_mp # Used by script
        self.theoretical = True
        self.energy_above_hull = 0.0
        self.deprecated = False
        # Fields required by DATA_SCHEMA for process_raw_data, ensure they exist even if with None/default
        self.composition_reduced = formula_pretty # Simplified
        self.chemsys = "-".join(sorted(list(set(c for c in formula_pretty if c.isupper()))))
        self.volume = kwargs.get('volume', 10.0)
        self.density = kwargs.get('density', 5.0)
        self.symmetry = kwargs.get('symmetry', {'crystal_system': 'cubic', 'symbol': 'Im-3m', 'number': 229, 'point_group': 'm-3m'})
        # Add any other field that select_best_mp_entry or later processing might expect from a SummaryDoc pydantic model
        for k,v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self): # process_raw_data uses this
        return self.__dict__

class MinimalMockStructure:
    def __init__(self, material_id="mp-test-int"):
        self.material_id = material_id
        if material_id == "mp-int-1":
            self.formula = "Fe"
            self.density = 7.87
            self.volume = 11.82
            self.lattice_param = 2.8
            self.sg_info = ("Im-3m", 229)
            mock_el = MagicMock(); mock_el.symbol = "Fe"
            self.elements = [mock_el]
        elif material_id == "mp-int-2":
            self.formula = "SiC"
            self.density = 3.21
            self.volume = 20.8
            self.lattice_param = 4.35 # Example for SiC
            self.sg_info = ("F-43m", 216)
            mock_el_si = MagicMock(); mock_el_si.symbol = "Si"
            mock_el_c = MagicMock(); mock_el_c.symbol = "C"
            self.elements = [mock_el_si, mock_el_c]
        else: # Default fallback
            self.formula = "Unknown"
            self.density = 1.0
            self.volume = 1.0
            self.lattice_param = 1.0
            self.sg_info = ("P1", 1)
            mock_el_x = MagicMock(); mock_el_x.symbol = "X"
            self.elements = [mock_el_x]

        self.composition = MagicMock()
        self.composition.reduced_formula = self.formula
        self.composition.elements = self.elements
        self.num_sites = len(self.elements) # Simplified

        self.lattice = MagicMock()
        self.lattice.a = self.lattice_param
        self.lattice.b = self.lattice_param
        self.lattice.c = self.lattice_param
        self.lattice.alpha, self.lattice.beta, self.lattice.gamma = 90, 90, 90
        self.get_space_group_info = MagicMock(return_value=self.sg_info)
        self.get_crystal_system = MagicMock(return_value="cubic") # Keep as cubic for simplicity or make dynamic

    def to(self, fmt="cif", **kwargs):
        if fmt == "cif":
            # Generate a very simple CIF string based on the formula
            atom_site_lines = "\n".join([f"{el.symbol}  {el.symbol}  0.0  0.0  0.0" for el in self.elements])
            return f"""
data_{self.material_id}
_cell_length_a                        {self.lattice.a}
_cell_length_b                        {self.lattice.b}
_cell_length_c                        {self.lattice.c}
_cell_angle_alpha                     {self.lattice.alpha}
_cell_angle_beta                      {self.lattice.beta}
_cell_angle_gamma                     {self.lattice.gamma}
_symmetry_space_group_name_H-M        '{self.sg_info[0]}'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{atom_site_lines}
"""
        raise NotImplementedError

class MinimalMockDOS:
    def __init__(self, material_id="mp-test-int"):
        self.material_id = material_id
        self.efermi = 0.5
        self.energies = [-1, 0.5, 2]
        # Ensure 'densities' is a dict with at least one key like 'Spin.up' or '1' if script sums over spins
        self._dos_data = {"efermi": self.efermi, "energies": self.energies, "densities": {"1": [1.0, 2.0, 3.0]}}
        # Ensure get_dos_at_fermi is present if called by process_raw_data
        self.get_dos_at_fermi = MagicMock(return_value=2.0)


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
    """Mocks MPRester for the integration test, configured for fetch_data."""
    mock_mpr_instance = MagicMock(name="MockMPResterInstance")

    # Define a single mock document that will be returned
    mock_doc_fe = MinimalMockSummaryDoc(
        material_id='mp-int-1', formula_pretty='Fe', nelements=1, band_gap_mp=0.0,
        formation_energy_per_atom_mp=-0.1, cif_string_mp='CIF_Fe_integration_test'
    )
    mock_doc_sic = MinimalMockSummaryDoc( # Changed SiC to be metallic for DOS model training
        material_id='mp-int-2', formula_pretty='SiC', nelements=2, band_gap_mp=0.0,
        formation_energy_per_atom_mp=-0.5, cif_string_mp='CIF_SiC_integration_test'
    )

    # Mock for mpr.materials.search (chemsys-based)
    def materials_search_side_effect(chemsys, fields):
        if chemsys == "Fe": return [mock_doc_fe]
        if chemsys == "C-Si": return [mock_doc_sic] # For SiC
        return []
    mock_mpr_instance.materials.search.side_effect = materials_search_side_effect

    # Mock for mpr.materials.summary.search (material_id-based)
    def summary_search_side_effect(material_ids, fields):
        results = []
        if "mp-int-1" in material_ids: results.append(mock_doc_fe)
        if "mp-int-2" in material_ids: results.append(mock_doc_sic)
        return results
    mock_mpr_instance.materials.summary.search.side_effect = summary_search_side_effect

    # Mock detail calls
    mock_mpr_instance.get_structure_by_material_id.side_effect = \
        lambda mid: MinimalMockStructure(mid) if mid in ["mp-int-1", "mp-int-2"] else None
    mock_mpr_instance.get_dos_by_material_id.side_effect = \
        lambda mid: MinimalMockDOS(mid) if mid in ["mp-int-1", "mp-int-2"] else None

    mock_MPRester_class = MagicMock(name="MockMPResterClass")
    mock_MPRester_class.return_value.__enter__.return_value = mock_mpr_instance
    monkeypatch.setattr("scripts.fetch_mp_data.MPRester", mock_MPRester_class)
    return mock_MPRester_class, mock_mpr_instance


# --- Integration Test ---
@patch("scripts.fetch_mp_data.get_supercon_compositions")
def test_data_pipeline_integration(mock_get_supercon_compositions, integration_test_config, integration_test_paths, mock_mp_rester_for_integration, capsys):
    # Docstring removed to fix syntax error from nested triple quotes
    config_file_path = str(integration_test_paths["config_file"])

    # The following block was mistakenly wrapped in triple quotes. Removing them.
    # config_file_path = str(integration_test_paths["config_file"]) # This line is a duplicate from above, removing.

    # Setup mock for get_supercon_compositions
    # Max_total_materials in config is 2. We provide 2 compositions.
    mock_get_supercon_compositions.return_value = {"Fe": 10.0, "SiC": 1.5}


    # --- 1. Execute fetch_data ---
    with patch('utils.config_loader.DEFAULT_CONFIG_PATH', config_file_path):
        fetch_data()

    # Assertions for fetch_data
    assert os.path.exists(integration_test_paths["raw_data_json"]), "Raw data JSON file was not created."
    with open(integration_test_paths["raw_data_json"], 'r') as f:
        raw_data_output = json.load(f) # This is now a dictionary
        # The number of keys in raw_data_output should be 2 (Fe and SiC)
        assert len(raw_data_output) == 2
        assert "Fe" in raw_data_output
        assert "SiC" in raw_data_output
        assert raw_data_output["Fe"]['material_id'] == "mp-int-1"
        assert raw_data_output["SiC"]['material_id'] == "mp-int-2"
        # Assert against the CIF content generated by the mock structure's .to() method
        assert raw_data_output["Fe"]['cif_string_mp'] == MinimalMockStructure("mp-int-1").to(fmt="cif")
        assert raw_data_output["SiC"]['cif_string_mp'] == MinimalMockStructure("mp-int-2").to(fmt="cif")


    # --- 2. Execute process_data ---
    # The mocks for Structure.from_str and Dos.from_dict need to align with what fetch_data produced.
    # fetch_data writes cif_string_mp and dos_object_mp (which is already a dict)

    # No need to mock Structure.from_str or Dos.from_dict if MinimalMockStructure/DOS are used by MPRester mock
    # and their .to() and .as_dict() methods produce compatible data for pymatgen's from_str/from_dict.
    # However, process_raw_data calls Structure.from_str(cif_string, fmt="cif") and Dos.from_dict(dos_dict)
    # So, the MinimalMockStructure and MinimalMockDOS must be compatible with these.
    # The .to() method of MinimalMockStructure should return a valid CIF string.
    # The .as_dict() method of MinimalMockDOS should return a dict that Dos.from_dict can parse.

    # For simplicity, we can directly patch Structure.from_str and Dos.from_dict as before,
    # but ensure they use the material_ids from the mocked fetch_data output.
    def side_effect_structure_from_str_integration(cif_string, fmt="cif"):
            # Check based on material_id which should be in the 'data_...' line
            if "data_mp-int-1" in cif_string:
                return MinimalMockStructure("mp-int-1") # Indented
            elif "data_mp-int-2" in cif_string:
                return MinimalMockStructure("mp-int-2") # Indented
            else: # Added else to make the raise part of the conditional block
                raise ValueError(f"Unexpected CIF string in integration test: {cif_string}")

    def side_effect_dos_from_dict_integration(dos_dict_arg):
        # Example: map based on a unique value in the dos_dict if possible, or material_id if it were passed
        # For now, assume it's called sequentially or we can inspect dos_dict_arg
        if dos_dict_arg.get('efermi') == MinimalMockDOS("mp-int-1").efermi : # Crude check
             return MinimalMockDOS("mp-int-1")
        elif dos_dict_arg.get('efermi') == MinimalMockDOS("mp-int-2").efermi:
             return MinimalMockDOS("mp-int-2")
        raise ValueError("Unexpected DOS dict in integration test")

    with patch('utils.config_loader.DEFAULT_CONFIG_PATH', config_file_path):
        with patch('scripts.process_raw_data.Structure.from_str', side_effect=side_effect_structure_from_str_integration):
            with patch('scripts.process_raw_data.Dos.from_dict', side_effect=side_effect_dos_from_dict_integration):
                with patch('scripts.process_raw_data.structure_to_graph') as mock_s2g_in_integration:
                    mock_s2g_in_integration.return_value = {
                        "nodes": [{"atomic_number": 26, "electronegativity": 1.83, "original_site_index": 0}],
                        "edges": [], "num_nodes": 1, "num_edges": 0
                    }
                    process_data()

    # Assertions for process_data
    assert os.path.exists(integration_test_paths["processed_csv"]), "Processed CSV file was not created."
    processed_df = pd.read_csv(integration_test_paths["processed_csv"])
    assert len(processed_df) == 2
    assert processed_df.iloc[0]['material_id'] == "mp-int-1" # Based on Fe
    assert processed_df.iloc[1]['material_id'] == "mp-int-2" # Based on SiC
    assert processed_df.iloc[0]['formula_pretty'] == "Fe" # From MinimalMockStructure
    assert processed_df.iloc[1]['formula_pretty'] == "SiC" # Updated based on dynamic MinimalMockStructure


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
