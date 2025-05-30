import json
import csv
import warnings
import os # For checking file existence
from utils.config_loader import load_config
from utils.schema import DATA_SCHEMA # Added

from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.dos import Dos # To reconstruct DOS objects

# DATA_SCHEMA is now imported from utils.schema

def process_data():
    """
    Loads raw data (filename from config), processes it,
    and saves it to a CSV file (filename from config).
    """
    full_config = load_config() # Use the new centralized loader
    if not full_config: # load_config returns {} on error or not found
        warnings.warn("Failed to load or parse config.yml for process_data. Using default script parameters.", UserWarning)
        process_config_params = {}
    else:
        process_config_params = full_config.get('process_data', {})

    raw_data_filename = process_config_params.get('raw_data_filename', "mp_raw_data.json")
    csv_filename_out = process_config_params.get('output_filename', "Fe_materials_dataset.csv")

    if not os.path.exists(raw_data_filename):
        print(f"Error: Raw data file '{raw_data_filename}' not found. Please run fetch_mp_data.py first.")
        return

    with open(raw_data_filename, 'r') as f:
        raw_materials_data = json.load(f)

    processed_materials_data = []
    print(f"Starting processing for {len(raw_materials_data)} raw material entries...")

    for i, raw_material_doc in enumerate(raw_materials_data):
        material_id = raw_material_doc.get('material_id', f"unknown_id_{i}")
        print(f"Processing material: {material_id} ({i+1}/{len(raw_materials_data)})")

        processed_doc = {'material_id': material_id}

        # --- Pymatgen Feature Extraction ---
        cif_string = raw_material_doc.get('cif_string')
        if cif_string:
            try:
                struct = Structure.from_str(cif_string, fmt="cif")

                processed_doc['formula_pretty'] = struct.composition.reduced_formula
                processed_doc['num_elements'] = len(struct.composition.elements)
                processed_doc['elements'] = ','.join(sorted([el.symbol for el in struct.composition.elements]))
                processed_doc['density_pg'] = struct.density
                processed_doc['volume_pg'] = struct.volume
                processed_doc['volume_per_atom_pg'] = struct.volume / struct.num_sites
                processed_doc['spacegroup_number_pg'] = struct.get_space_group_info()[1] # (symbol, number)
                processed_doc['crystal_system_pg'] = struct.get_crystal_system()

                lat = struct.lattice
                processed_doc['lattice_a_pg'] = lat.a
                processed_doc['lattice_b_pg'] = lat.b
                processed_doc['lattice_c_pg'] = lat.c
                processed_doc['lattice_alpha_pg'] = lat.alpha
                processed_doc['lattice_beta_pg'] = lat.beta
                processed_doc['lattice_gamma_pg'] = lat.gamma
                processed_doc['num_sites_pg'] = struct.num_sites

            except Exception as e:
                warnings.warn(f"Pymatgen parsing/feature extraction failed for {material_id}: {e}")
                # Fill relevant pymatgen features with None or N/A if parsing fails
                pg_features = [k for k in DATA_SCHEMA if k.endswith('_pg') or k in ["formula_pretty", "num_elements", "elements"]]
                for feat in pg_features:
                    processed_doc[feat] = None
        else:
            warnings.warn(f"CIF string missing for {material_id}.")
            pg_features = [k for k in DATA_SCHEMA if k.endswith('_pg') or k in ["formula_pretty", "num_elements", "elements"]]
            for feat in pg_features:
                processed_doc[feat] = None

        # --- Process MP Features and DOS ---
        band_gap_mp = raw_material_doc.get('band_gap_mp')
        processed_doc['band_gap_mp'] = band_gap_mp
        processed_doc['formation_energy_per_atom_mp'] = raw_material_doc.get('formation_energy_per_atom_mp')

        if band_gap_mp is not None:
            processed_doc['is_metal'] = (band_gap_mp == 0.0)
        else:
            processed_doc['is_metal'] = None

        dos_dict = raw_material_doc.get('dos_object_mp')
        processed_doc['dos_at_fermi'] = None # Default to None
        if dos_dict:
            try:
                pymatgen_dos = Dos.from_dict(dos_dict)
                # Check if efermi is present and valid before calculating DOS at Fermi
                if pymatgen_dos.efermi is not None:
                    # Ensure Fermi level is within the energy range of the DOS plot
                    min_energy = min(pymatgen_dos.energies)
                    max_energy = max(pymatgen_dos.energies)
                    if min_energy <= pymatgen_dos.efermi <= max_energy:
                        dos_val = pymatgen_dos.get_dos_at_fermi()
                        # get_dos_at_fermi() might return a small list if multiple spin channels, sum them for total DOS
                        processed_doc['dos_at_fermi'] = sum(dos_val) if isinstance(dos_val, list) else dos_val
                    else:
                        warnings.warn(f"Fermi level {pymatgen_dos.efermi} for {material_id} is outside DOS energy range [{min_energy}, {max_energy}]. Setting dos_at_fermi to 0.")
                        processed_doc['dos_at_fermi'] = 0.0 # Or None, depending on desired handling
                else:
                     warnings.warn(f"Fermi level not found in DOS object for {material_id}. Cannot calculate dos_at_fermi.")
            except Exception as e:
                warnings.warn(f"DOS processing failed for {material_id}: {e}")

        # --- Set Target Columns ---
        processed_doc['target_band_gap'] = processed_doc.get('band_gap_mp')
        processed_doc['target_formation_energy'] = processed_doc.get('formation_energy_per_atom_mp')
        processed_doc['target_is_metal'] = processed_doc.get('is_metal')
        processed_doc['target_dos_at_fermi'] = processed_doc.get('dos_at_fermi')

        processed_materials_data.append(processed_doc)

    # --- Write to CSV ---
    # Define CSV fieldnames - ensure order and exclude non-CSV fields
    csv_fieldnames = [key for key in DATA_SCHEMA.keys() if key not in ['cif_string', 'dos_object_mp']]

    # Ensure all keys in processed_doc are in csv_fieldnames, add if any are missing (e.g. due to dynamic keys)
    # However, it's better to strictly adhere to DATA_SCHEMA for CSV columns.
    # For this implementation, we assume processed_doc keys will be a subset of DATA_SCHEMA keys intended for CSV.

    # csv_filename_out is sourced from config or default at the start of the function
    print(f"\nWriting processed data to {csv_filename_out}...")
    try:
        with open(csv_filename_out, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames, extrasaction='ignore') # extrasaction='ignore' is safer
            writer.writeheader()
            for row_data in processed_materials_data:
                writer.writerow(row_data)
        print(f"Successfully processed and saved data for {len(processed_materials_data)} materials to {csv_filename_out}.")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    process_data()
