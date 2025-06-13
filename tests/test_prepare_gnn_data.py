import unittest
import os
import json
import torch
import tempfile
import shutil
import yaml
from torch_geometric.data import Data

# Add project root to sys.path to allow direct import of prepare_gnn_data
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import prepare_gnn_data
# from utils.config_loader import load_config # Not strictly needed for test if prepare_gnn_data handles path

# CIF string for Silicon - must be properly escaped for JSON
SI_CIF_STRING = """data_Si
_symmetry_space_group_name_H-M   'F d -3 m'
_cell_length_a   5.43000
_cell_length_b   5.43000
_cell_length_c   5.43000
_cell_angle_alpha   90.00000
_cell_angle_beta   90.00000
_cell_angle_gamma   90.00000
_symmetry_Int_Tables_number   227
_chemical_formula_structural   Si
_chemical_formula_sum   Si8
_cell_volume   160.165
_cell_formula_units_Z   8
loop_
  _symmetry_equiv_pos_site_id
  _symmetry_equiv_pos_as_xyz
   1  'x,y,z'
loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
   Si  Si1       8  0.00000  0.00000  0.00000  1.00
"""

class TestPrepareGnnData(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create dummy raw data
        self.dummy_raw_data = []
        for i in range(5): # Create 5 entries
            self.dummy_raw_data.append({
                "material_id": f"mp-test-{i+1}",
                "cif": SI_CIF_STRING, # Using Si CIF for all
                "band_gap_mp": 2.5 + i*0.1,
                "formation_energy_per_atom_mp": -0.5 - i*0.05
            })

        self.dummy_raw_data_path = os.path.join(self.test_dir, "dummy_raw_data.json")
        with open(self.dummy_raw_data_path, 'w') as f:
            json.dump(self.dummy_raw_data, f)

        # Create dummy config
        self.processed_graphs_name = "processed_graphs.pt"
        self.train_graphs_name = "train_graphs.pt"
        self.val_graphs_name = "val_graphs.pt"
        self.test_graphs_name = "test_graphs.pt"

        self.dummy_config_data = {
            'prepare_gnn_data': {
                'raw_data_filename': self.dummy_raw_data_path,
                'processed_graphs_filename': os.path.join(self.test_dir, self.processed_graphs_name),
                'train_graphs_filename': os.path.join(self.test_dir, self.train_graphs_name),
                'val_graphs_filename': os.path.join(self.test_dir, self.val_graphs_name),
                'test_graphs_filename': os.path.join(self.test_dir, self.test_graphs_name),
                'random_seed': 42,
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1
            }
        }
        self.dummy_config_path = os.path.join(self.test_dir, "dummy_config.yml")
        with open(self.dummy_config_path, 'w') as f:
            yaml.dump(self.dummy_config_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_script_runs_and_creates_outputs(self):
        # Call the main function of prepare_gnn_data.py
        prepare_gnn_data.create_graph_dataset(config_path=self.dummy_config_path)

        # Assert output files exist
        processed_path = os.path.join(self.test_dir, self.processed_graphs_name)
        train_path = os.path.join(self.test_dir, self.train_graphs_name)
        val_path = os.path.join(self.test_dir, self.val_graphs_name)
        test_path = os.path.join(self.test_dir, self.test_graphs_name)

        self.assertTrue(os.path.exists(processed_path), "Processed graphs file not found.")
        self.assertTrue(os.path.exists(train_path), "Train graphs file not found.")
        self.assertTrue(os.path.exists(val_path), "Validation graphs file not found.")
        self.assertTrue(os.path.exists(test_path), "Test graphs file not found.")

        # Load data and check contents
        all_graphs = torch.load(processed_path, weights_only=False)
        train_graphs = torch.load(train_path, weights_only=False)
        val_graphs = torch.load(val_path, weights_only=False)
        test_graphs = torch.load(test_path, weights_only=False)

        self.assertIsInstance(all_graphs, list)
        self.assertTrue(all(isinstance(g, Data) for g in all_graphs))
        total_graphs = len(self.dummy_raw_data) # Expect all 5 to be processed
        self.assertEqual(len(all_graphs), total_graphs)

        self.assertIsInstance(train_graphs, list)
        self.assertTrue(all(isinstance(g, Data) for g in train_graphs))
        self.assertIsInstance(val_graphs, list)
        self.assertTrue(all(isinstance(g, Data) for g in val_graphs))
        self.assertIsInstance(test_graphs, list)
        self.assertTrue(all(isinstance(g, Data) for g in test_graphs))

        self.assertEqual(len(train_graphs) + len(val_graphs) + len(test_graphs), total_graphs)

        # Check split numbers. With 5 samples and 70/20/10 split & fixed seed:
        # Train: 5 * 0.7 = 3.5. floor/ceil based on sklearn split logic.
        #        (0.2+0.1)*5 = 1.5 (for temp). So train is 3 or 4.
        #        If train = 3, temp = 2. Then 2 * (0.1 / (0.2+0.1)) = 2 * (1/3) = 0.66 for test. test=1, val=1. (3,1,1)
        #        If train = 4, temp = 1. Then 1 * (0.1 / (0.2+0.1)) = 1 * (1/3) = 0.33 for test. test=0, val=1. (4,1,0)
        # The exact numbers depend on sklearn's internal logic for train_test_split with small floats.
        # We'll check approximate ratios.
        # With seed 42, 5 samples, 0.7/0.2/0.1 split:
        # 1st split (test_size=0.3): train=3, temp=2
        # 2nd split (test_size=0.1/0.3 = 1/3) on temp=2: val=1, test=1.  (Expected: train=3, val=1, test=1)

        self.assertEqual(len(train_graphs), 3, f"Expected 3 train samples, got {len(train_graphs)}")
        self.assertEqual(len(val_graphs), 1, f"Expected 1 val sample, got {len(val_graphs)}")
        self.assertEqual(len(test_graphs), 1, f"Expected 1 test sample, got {len(test_graphs)}")

        # Check if material_id is preserved
        self.assertIsNotNone(all_graphs[0].material_id)
        self.assertTrue(all_graphs[0].material_id.startswith("mp-test-"))


if __name__ == "__main__":
    unittest.main()
