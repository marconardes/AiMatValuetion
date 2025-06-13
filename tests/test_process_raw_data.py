import pytest
import json
import csv
import os
import warnings
from unittest.mock import patch, mock_open, MagicMock, call

# Modules to be tested
from scripts.process_raw_data import process_data
from utils.schema import DATA_SCHEMA

# Define sample raw data for testing
SAMPLE_RAW_MATERIAL_FULL = {
    "material_id": "mp-123",
    "cif_string": """
data_Si
_symmetry_space_group_name_H-M   'F d -3 m'
_cell_length_a   5.4300
_cell_length_b   5.4300
_cell_length_c   5.4300
_cell_angle_alpha   90
_cell_angle_beta   90
_cell_angle_gamma   90
loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Si  0.00000  0.00000  0.00000
""",
    "band_gap_mp": 0.5, # Insulator
    "formation_energy_per_atom_mp": -1.0,
    "dos_object_mp": { # Simplified mock DOS dict structure
        "efermi": 0.0,
        "energies": [-1, 0, 1],
        "densities": {"1": [0.1, 0.2, 0.3]} # Spin up channel
    }
}

SAMPLE_RAW_MATERIAL_METAL = {
    "material_id": "mp-456",
    "cif_string": SAMPLE_RAW_MATERIAL_FULL["cif_string"], # Reuse valid CIF
    "band_gap_mp": 0.0, # Metal
    "formation_energy_per_atom_mp": -0.5,
    "dos_object_mp": {
        "efermi": 1.0,
        "energies": [0, 1, 2],
        "densities": {"1": [0.5, 1.5, 0.5]}
    }
}

SAMPLE_RAW_MATERIAL_MISSING_CIF = {
    "material_id": "mp-789",
    "cif_string": None,
    "band_gap_mp": 1.0,
    "formation_energy_per_atom_mp": -0.8,
    "dos_object_mp": None
}

SAMPLE_RAW_MATERIAL_MISSING_DOS = {
    "material_id": "mp-101",
    "cif_string": SAMPLE_RAW_MATERIAL_FULL["cif_string"], # Reuse valid CIF
    "band_gap_mp": 0.0, # Metal
    "formation_energy_per_atom_mp": -0.2,
    "dos_object_mp": None
}

SAMPLE_RAW_MATERIAL_DOS_EFERMI_OUTSIDE_RANGE = {
    "material_id": "mp-112",
    "cif_string": SAMPLE_RAW_MATERIAL_FULL["cif_string"],
    "band_gap_mp": 0.0,
    "formation_energy_per_atom_mp": -0.3,
    "dos_object_mp": {"efermi": 10.0, "energies": [-1, 0, 1], "densities": {"1": [0.1,0.2,0.3]}}
}

SAMPLE_RAW_MATERIAL_DOS_NO_EFERMI = {
    "material_id": "mp-113",
    "cif_string": SAMPLE_RAW_MATERIAL_FULL["cif_string"],
    "band_gap_mp": 0.0,
    "formation_energy_per_atom_mp": -0.4,
    "dos_object_mp": {"energies": [-1, 0, 1], "densities": {"1": [0.1,0.2,0.3]}} # efermi key missing
}


@pytest.fixture
def mock_raw_data_file(tmp_path):
    def _create_json_file(data_list, filename="mp_raw_data.json"):
        file_path = tmp_path / filename
        with open(file_path, 'w') as f:
            json.dump(data_list, f)
        return str(file_path)
    return _create_json_file

@pytest.fixture
def mock_config_for_processing(tmp_path):
    # This fixture will mock the load_config call within process_raw_data
    # to control input and output filenames.
    def _get_config(raw_filename="mp_raw_data.json", output_filename="output.csv"):
        return {
            "process_data": {
                "raw_data_filename": str(tmp_path / raw_filename),
                "output_filename": str(tmp_path / output_filename)
            }
        }
    return _get_config

# Expected CSV columns based on DATA_SCHEMA (excluding non-CSV fields)
EXPECTED_CSV_HEADERS_FROM_SCHEMA = [key for key in DATA_SCHEMA.keys() if key not in ['cif_string', 'dos_object_mp']]

@patch('scripts.process_raw_data.Structure.from_str')
@patch('scripts.process_raw_data.Dos.from_dict')
def test_successful_processing_with_mocks(mock_dos_from_dict, mock_structure_from_str, tmp_path, mock_raw_data_file, mock_config_for_processing, capsys):
    """Test basic successful processing with detailed mocks for pymatgen objects."""

    # --- Setup mock Structure object ---
    mock_structure = MagicMock(spec=True) # spec=True helps catch incorrect attribute access

    # Mock attributes for composition
    mock_composition = MagicMock()
    mock_composition.reduced_formula = "SiMock"
    mock_composition.elements = [MagicMock(symbol='Si')] # Simulate element objects
    mock_structure.composition = mock_composition

    mock_structure.density = 1.23
    mock_structure.volume = 100.0
    mock_structure.num_sites = 2
    # Ensure methods are explicitly mocked if spec=True is used
    mock_structure.get_space_group_info = MagicMock(return_value=("Fd-3m", 227))
    mock_structure.get_crystal_system = MagicMock(return_value="cubic") # Must be lowercase as per code

    mock_lattice = MagicMock(spec=True)
    mock_lattice.a, mock_lattice.b, mock_lattice.c = 5.0, 5.0, 5.0
    mock_lattice.alpha, mock_lattice.beta, mock_lattice.gamma = 90, 90, 90
    mock_structure.lattice = mock_lattice

    mock_structure_from_str.return_value = mock_structure

    # --- Setup mock Dos object behavior per material_id ---
    # This mapping will help the side_effect choose the right mock
    dos_mock_mapping = {}

    mock_dos_full = MagicMock(spec=True, name="DosForFull_mp-123")
    mock_dos_full.efermi = 0.0
    mock_dos_full.energies = [-1, 0, 1]
    mock_dos_full.get_dos_at_fermi = MagicMock(return_value=0.1)
    dos_mock_mapping['mp-123'] = mock_dos_full

    mock_dos_metal = MagicMock(spec=True, name="DosForMetal_mp-456")
    mock_dos_metal.efermi = 1.0
    mock_dos_metal.energies = [0,1,2]
    mock_dos_metal.get_dos_at_fermi = MagicMock(return_value=0.5)
    dos_mock_mapping['mp-456'] = mock_dos_metal

    # Side effect function for Dos.from_dict
    # It will use the material_id from the raw_material_doc being processed
    # This requires a bit of finesse as from_dict itself doesn't know the material_id
    # We'll rely on the order of processing or pass material_id if we can intercept higher up.
    # For this test structure, we'll have to make it simpler: assume it's called with the dos_object_mp
    def side_effect_dos_from_dict(dos_dict_arg):
        if dos_dict_arg is None: return None # Should not happen if dos_object_mp is None
        # Crude way to map based on efermi, assuming efermi is unique in test data DOS objects
        if dos_dict_arg.get('efermi') == 0.0: return dos_mock_mapping['mp-123']
        if dos_dict_arg.get('efermi') == 1.0: return dos_mock_mapping['mp-456']
        return MagicMock(spec=True, name="DefaultDosMock") # Default if no specific match
    mock_dos_from_dict.side_effect = side_effect_dos_from_dict

    sample_data = [SAMPLE_RAW_MATERIAL_FULL, SAMPLE_RAW_MATERIAL_METAL, SAMPLE_RAW_MATERIAL_MISSING_CIF]
    raw_json_path = mock_raw_data_file(sample_data)
    output_csv_name = "processed_test_data.csv"

    config_val = mock_config_for_processing(raw_filename=os.path.basename(raw_json_path), output_filename=output_csv_name)

    with patch('scripts.process_raw_data.load_config', return_value=config_val):
        mock_csv_writer = MagicMock()
        with patch('csv.DictWriter', return_value=mock_csv_writer):
            process_data()

            mock_csv_writer.writeheader.assert_called_once()
            assert mock_csv_writer.writerow.call_count == len(sample_data)

            # Check mp-123 (SAMPLE_RAW_MATERIAL_FULL)
            args_full, _ = mock_csv_writer.writerow.call_args_list[0]
            row_full = args_full[0]
            assert row_full['material_id'] == 'mp-123'
            assert row_full['formula_pretty'] == 'SiMock'
            assert row_full['density_pg'] == 1.23
            assert row_full['crystal_system_pg'] == 'cubic'
            assert row_full['is_metal'] is False
            assert row_full['dos_at_fermi'] == 0.1

            # Check mp-456 (SAMPLE_RAW_MATERIAL_METAL)
            args_metal, _ = mock_csv_writer.writerow.call_args_list[1]
            row_metal = args_metal[0]
            assert row_metal['material_id'] == 'mp-456'
            assert row_metal['is_metal'] is True
            assert row_metal['dos_at_fermi'] == 0.5

            # Check mp-789 (SAMPLE_RAW_MATERIAL_MISSING_CIF)
            args_missing_cif, _ = mock_csv_writer.writerow.call_args_list[2]
            row_missing_cif = args_missing_cif[0]
            assert row_missing_cif['material_id'] == 'mp-789'
            assert row_missing_cif['formula_pretty'] is None
            assert row_missing_cif['density_pg'] is None
            assert row_missing_cif['dos_at_fermi'] is None

    captured = capsys.readouterr()
    assert f"Successfully processed and saved data for {len(sample_data)} materials" in captured.out


def test_missing_input_file(mock_config_for_processing, capsys):
    """Test behavior when the raw data JSON file is missing."""
    # Configure to use a non-existent raw file
    config_val = mock_config_for_processing(raw_filename="non_existent_raw.json")

    with patch('scripts.process_raw_data.load_config', return_value=config_val):
        # os.path.exists will correctly return False for a file in tmp_path that doesn't exist
        process_data()

        captured = capsys.readouterr()
        expected_raw_path = config_val['process_data']['raw_data_filename']
        assert f"Error: Raw data file '{expected_raw_path}' not found." in captured.out
        assert "Successfully processed and saved data" not in captured.out


@patch('scripts.process_raw_data.Dos.from_dict') # Keep this for DOS part
@patch('scripts.process_raw_data.Structure.from_str') # Mock this for Structure part
def test_pymatgen_cif_parsing_error(mock_structure_from_str, mock_dos_from_dict, tmp_path, mock_raw_data_file, mock_config_for_processing, capsys):
    """Test handling of errors during Structure.from_str (CIF parsing)."""
    raw_data_with_bad_cif = [{
        "material_id": "mp-bad-cif", "cif_string": "this is not a valid cif",
        "band_gap_mp": 0.1, "formation_energy_per_atom_mp": -0.1, "dos_object_mp": None
    }]
    raw_json_path = mock_raw_data_file(raw_data_with_bad_cif)

    mock_structure_from_str.side_effect = Exception("Mocked CIF parsing error")

    mock_csv_writer = MagicMock()
    config_val = mock_config_for_processing(raw_filename=os.path.basename(raw_json_path))

    with patch('scripts.process_raw_data.load_config', return_value=config_val):
        with patch('csv.DictWriter', return_value=mock_csv_writer):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                process_data()

                assert any("Pymatgen parsing/feature extraction failed for mp-bad-cif" in str(warn.message) for warn in w)

            mock_csv_writer.writerow.assert_called_once()
            call_args = mock_csv_writer.writerow.call_args[0][0]
            assert call_args['material_id'] == "mp-bad-cif"
            assert call_args['formula_pretty'] is None
            assert call_args['density_pg'] is None


@patch('scripts.process_raw_data.Structure.from_str') # Mock Structure for all DOS tests too
@patch('scripts.process_raw_data.Dos.from_dict')
def test_dos_processing_logic(mock_dos_from_dict, mock_structure_from_str, tmp_path, mock_raw_data_file, mock_config_for_processing):
    """Test various scenarios for DOS processing."""

    # Default mock for structure, can be overridden per test case if needed by re-patching or more complex side_effect
    mock_default_structure = MagicMock(spec=True)
    mock_default_composition = MagicMock()
    mock_default_composition.reduced_formula = "MockFormula"
    mock_default_composition.elements = [MagicMock(symbol='X')]
    mock_default_structure.composition = mock_default_composition
    mock_default_structure.density = 1.0
    mock_default_structure.volume = 1.0
    mock_default_structure.num_sites = 1
    mock_default_structure.get_space_group_info = MagicMock(return_value=("P1", 1))
    mock_default_structure.get_crystal_system = MagicMock(return_value="triclinic")
    mock_default_lattice = MagicMock(spec=True)
    mock_default_lattice.a, mock_default_lattice.b, mock_default_lattice.c = 1,1,1
    mock_default_lattice.alpha, mock_default_lattice.beta, mock_default_lattice.gamma = 90,90,90
    mock_default_structure.lattice = mock_default_lattice
    mock_structure_from_str.return_value = mock_default_structure

    # Scenario 1: Valid DOS object, efermi in range
    mock_dos_s1 = MagicMock(spec=True, name="DosCase1_mp-123")
    mock_dos_s1.efermi = 0.5
    mock_dos_s1.energies = [0, 0.5, 1.0]
    mock_dos_s1.get_dos_at_fermi = MagicMock(return_value=1.23)

    # Scenario 2: efermi outside DOS energy range
    mock_dos_s2 = MagicMock(spec=True, name="DosCase2_mp-112")
    mock_dos_s2.efermi = 5.0
    mock_dos_s2.energies = [0, 1, 2]
    mock_dos_s2.get_dos_at_fermi = MagicMock(return_value=0.0) # Define it even if logic might not call

    # Scenario 3: efermi is None
    mock_dos_s3 = MagicMock(spec=True, name="DosCase3_mp-113")
    mock_dos_s3.efermi = None
    mock_dos_s3.get_dos_at_fermi = MagicMock(return_value=None) # Define it

    test_cases_dos = [
        (SAMPLE_RAW_MATERIAL_FULL, mock_dos_s1, 1.23), # mp-123 uses mock_dos_s1
        (SAMPLE_RAW_MATERIAL_DOS_EFERMI_OUTSIDE_RANGE, mock_dos_s2, 0.0),
        (SAMPLE_RAW_MATERIAL_DOS_NO_EFERMI, mock_dos_s3, None),
        (SAMPLE_RAW_MATERIAL_MISSING_DOS, None, None)
    ]

    # Special side effect for mock_dos_from_dict for this specific test
    # to ensure the correct mock instance is returned for the material being processed.
    def dos_side_effect_for_test(dos_dict_arg):
        if dos_dict_arg == SAMPLE_RAW_MATERIAL_FULL['dos_object_mp']:
            return mock_dos_s1
        elif dos_dict_arg == SAMPLE_RAW_MATERIAL_DOS_EFERMI_OUTSIDE_RANGE['dos_object_mp']:
            return mock_dos_s2
        elif dos_dict_arg == SAMPLE_RAW_MATERIAL_DOS_NO_EFERMI['dos_object_mp']:
            return mock_dos_s3
        return MagicMock(spec=True, name="DefaultDosInTest") # Default if no specific match
    mock_dos_from_dict.side_effect = dos_side_effect_for_test


    for i, (raw_material, _, expected_dos_val) in enumerate(test_cases_dos): # mock_dos_instance no longer directly used here
        # The side_effect for mock_dos_from_dict will pick the correct mock
        raw_json_path = mock_raw_data_file([raw_material], filename=f"raw_dos_test_{i}.json")

        mock_csv_writer = MagicMock()
        config_val = mock_config_for_processing(raw_filename=os.path.basename(raw_json_path), output_filename=f"out_dos_{i}.csv")

        with patch('scripts.process_raw_data.load_config', return_value=config_val):
            with patch('csv.DictWriter', return_value=mock_csv_writer):
                with warnings.catch_warnings(record=True) as w_dos:
                    warnings.simplefilter("always")
                    process_data()

                mock_csv_writer.writerow.assert_called_once()
                processed_output = mock_csv_writer.writerow.call_args[0][0]

                assert processed_output['material_id'] == raw_material['material_id']
                assert processed_output['dos_at_fermi'] == expected_dos_val
                assert processed_output['target_dos_at_fermi'] == expected_dos_val

                if raw_material['material_id'] == "mp-112":
                    assert any("is outside DOS energy range" in str(warn.message) for warn in w_dos)
                if raw_material['material_id'] == "mp-113":
                    assert any("Fermi level not found" in str(warn.message) for warn in w_dos)


# Basic test to check if the script runs without real files if config is empty
def test_process_data_no_config_runs(capsys):
    with patch('scripts.process_raw_data.load_config', return_value={}):
        with patch('os.path.exists', return_value=False) as mock_exists:
            process_data()
            mock_exists.assert_called_once_with("data/mp_raw_data.json")
            captured = capsys.readouterr()
            assert "Error: Raw data file 'data/mp_raw_data.json' not found" in captured.out

# Test with a completely empty raw data file
def test_process_empty_raw_data_file(tmp_path, mock_raw_data_file, mock_config_for_processing, capsys):
    raw_json_path = mock_raw_data_file([])

    config_val = mock_config_for_processing(raw_filename=os.path.basename(raw_json_path))
    with patch('scripts.process_raw_data.load_config', return_value=config_val):
        mock_csv_writer = MagicMock()
        with patch('csv.DictWriter', return_value=mock_csv_writer):
            process_data()
            # If there's no data, the header should ideally still be written to indicate an empty dataset
            # that conforms to the schema. This matches current code behavior.
            mock_csv_writer.writeheader.assert_called_once()
            mock_csv_writer.writerow.assert_not_called()
            captured = capsys.readouterr()
            assert "Starting processing for 0 raw material entries..." in captured.out
            assert "Successfully processed and saved data for 0 materials" in captured.out
