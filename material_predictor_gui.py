import tkinter as tk
from tkinter import ttk # Import ttk
from tkinter import filedialog
from tkinter import messagebox
from pymatgen.core import Structure
import csv
import os
import joblib
import pandas as pd
import numpy as np

# DATA_SCHEMA (consistent with Fe_materials_dataset.csv structure)
# This helps in defining fields and saving data.
CSV_HEADERS = [
    "material_id", "band_gap_mp", "formation_energy_per_atom_mp", "is_metal", "dos_at_fermi",
    "formula_pretty", "num_elements", "elements", "density_pg", "volume_pg", "volume_per_atom_pg",
    "spacegroup_number_pg", "crystal_system_pg", "lattice_a_pg", "lattice_b_pg", "lattice_c_pg",
    "lattice_alpha_pg", "lattice_beta_pg", "lattice_gamma_pg", "num_sites_pg",
    "target_band_gap", "target_formation_energy", "target_is_metal", "target_dos_at_fermi"
]

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Material Property Predictor Prototype")
        self.geometry("900x700") # Adjusted size for more content

        # Tabbed interface
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # --- Prediction Tab (existing functionality) ---
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text='Predict from CIF')
        self.setup_prediction_tab()

        # --- Manual Data Entry Tab ---
        self.manual_entry_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.manual_entry_tab, text='Manual Data Entry')
        self.manual_entry_fields = {} # To store Entry/Combobox widgets
        self.setup_manual_entry_tab()

        self.selected_manual_cif_path = tk.StringVar()

        # Load models and preprocessors
        self.load_models()


    def load_models(self):
        print("Loading models and preprocessors...")
        models_to_load = {
            "preprocessor_main": "preprocessor_main.joblib",
            "preprocessor_dos": "preprocessor_dos_at_fermi.joblib", # Corrected key for consistency
            "model_band_gap": "model_target_band_gap.joblib",
            "model_formation_energy": "model_target_formation_energy.joblib",
            "model_is_metal": "model_target_is_metal.joblib",
            "model_dos_at_fermi": "model_dos_at_fermi.joblib"
        }
        for attr_name, filename in models_to_load.items():
            try:
                setattr(self, attr_name, joblib.load(filename))
                print(f"Successfully loaded {filename} as self.{attr_name}")
            except FileNotFoundError:
                warnings.warn(f"Warning: {filename} not found. Predictions using this will be unavailable.")
                setattr(self, attr_name, None)
            except Exception as e:
                warnings.warn(f"Warning: Error loading {filename}: {e}. Predictions using this will be unavailable.")
                setattr(self, attr_name, None)


    def setup_prediction_tab(self):
        # Existing prediction functionality widgets moved here
        self.selected_file_path = ""
        self.current_structure = None
        # These _extracted_ variables are for pymatgen direct values, not model predictions
        # self.extracted_formula = "N/A"
        # self.extracted_density = "N/A"
        # self.extracted_volume = "N/A"

        select_button = ttk.Button(self.prediction_tab, text="Select CIF File", command=self.select_file_for_prediction)
        select_button.pack(pady=10)

        self.file_label = ttk.Label(self.prediction_tab, text="No file selected")
        self.file_label.pack(pady=5)

        predict_button = ttk.Button(self.prediction_tab, text="Predict Properties", command=self.perform_prediction)
        predict_button.pack(pady=10)

        results_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Results")
        results_frame.pack(padx=10, pady=10, fill="x", expand=True)

        # Pymatgen Extracted Features Display
        pg_features_frame = ttk.LabelFrame(results_frame, text="Pymatgen-Derived Features")
        pg_features_frame.pack(padx=5, pady=5, fill="x", expand=True)
        self.formula_label_value = self._create_display_label(pg_features_frame, "Chemical Formula:")
        self.density_label_value = self._create_display_label(pg_features_frame, "Density:")
        self.volume_label_value = self._create_display_label(pg_features_frame, "Cell Volume:")

        # Model Predictions Display
        model_preds_frame = ttk.LabelFrame(results_frame, text="Model Predictions")
        model_preds_frame.pack(padx=5, pady=5, fill="x", expand=True)
        self.model_is_metal_label = self._create_display_label(model_preds_frame, "Is Metal:")
        self.model_band_gap_label = self._create_display_label(model_preds_frame, "Band Gap (Model):")
        self.model_formation_energy_label = self._create_display_label(model_preds_frame, "Formation Energy (Model):")
        self.model_dos_at_fermi_label = self._create_display_label(model_preds_frame, "DOS at Fermi (Model):")


    def _create_display_label(self, parent, text):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2) # Added padx
        ttk.Label(frame, text=text, width=25, anchor='w').pack(side=tk.LEFT) # Fixed width for title
        value_label = ttk.Label(frame, text="N/A", anchor='w') # Dynamic width for value
        value_label.pack(side=tk.LEFT, fill='x', expand=True)
        return value_label

    def setup_manual_entry_tab(self):
        # Scrollable Frame Setup
        canvas = tk.Canvas(self.manual_entry_tab)
        scrollbar = ttk.Scrollbar(self.manual_entry_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Labelframes for sections ---
        # Identifier Section
        id_frame = ttk.LabelFrame(scrollable_frame, text="Identifier")
        id_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(id_frame, "material_id", "Material ID*:", CSV_HEADERS[0])

        # Load CIF Section
        load_cif_frame = ttk.LabelFrame(scrollable_frame, text="Load CIF for Feature Extraction")
        load_cif_frame.pack(padx=10, pady=5, fill="x")
        ttk.Button(load_cif_frame, text="Load CIF File", command=self.load_cif_for_manual_entry).pack(side=tk.LEFT, padx=5, pady=5)
        self.manual_cif_path_label = ttk.Label(load_cif_frame, text="No CIF file loaded.")
        self.manual_cif_path_label.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)


        # Pymatgen-Derived Features
        pg_frame = ttk.LabelFrame(scrollable_frame, text="Pymatgen-Derived Features (auto-filled from CIF or manual)")
        pg_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(pg_frame, "formula_pretty", "Formula (Pretty):", CSV_HEADERS[5])
        self._add_manual_entry_field(pg_frame, "num_elements", "Number of Elements:", CSV_HEADERS[6], entry_type='number')
        self._add_manual_entry_field(pg_frame, "elements", "Elements (comma-sep, sorted):", CSV_HEADERS[7])
        self._add_manual_entry_field(pg_frame, "density_pg", "Density (g/cm³):", CSV_HEADERS[8], entry_type='number')
        self._add_manual_entry_field(pg_frame, "volume_pg", "Cell Volume (Å³):", CSV_HEADERS[9], entry_type='number')
        self._add_manual_entry_field(pg_frame, "volume_per_atom_pg", "Volume per Atom (Å³/atom):", CSV_HEADERS[10], entry_type='number')
        self._add_manual_entry_field(pg_frame, "spacegroup_number_pg", "Space Group Number:", CSV_HEADERS[11], entry_type='number')
        crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic', 'N/A']
        self._add_manual_entry_field(pg_frame, "crystal_system_pg", "Crystal System:", CSV_HEADERS[12], options=crystal_systems)
        self._add_manual_entry_field(pg_frame, "lattice_a_pg", "Lattice a (Å):", CSV_HEADERS[13], entry_type='number')
        self._add_manual_entry_field(pg_frame, "lattice_b_pg", "Lattice b (Å):", CSV_HEADERS[14], entry_type='number')
        self._add_manual_entry_field(pg_frame, "lattice_c_pg", "Lattice c (Å):", CSV_HEADERS[15], entry_type='number')
        self._add_manual_entry_field(pg_frame, "lattice_alpha_pg", "Lattice α (°):", CSV_HEADERS[16], entry_type='number')
        self._add_manual_entry_field(pg_frame, "lattice_beta_pg", "Lattice β (°):", CSV_HEADERS[17], entry_type='number')
        self._add_manual_entry_field(pg_frame, "lattice_gamma_pg", "Lattice γ (°):", CSV_HEADERS[18], entry_type='number')
        self._add_manual_entry_field(pg_frame, "num_sites_pg", "Number of Sites:", CSV_HEADERS[19], entry_type='number')

        # Externally Sourced / MP Features
        mp_frame = ttk.LabelFrame(scrollable_frame, text="Externally Sourced Properties (e.g., from Materials Project)")
        mp_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(mp_frame, "band_gap_mp", "Band Gap (MP, eV):", CSV_HEADERS[1], entry_type='number')
        self._add_manual_entry_field(mp_frame, "formation_energy_per_atom_mp", "Formation Energy (MP, eV/atom):", CSV_HEADERS[2], entry_type='number')
        self._add_manual_entry_field(mp_frame, "is_metal", "Is Metal (MP):", CSV_HEADERS[3], options=["True", "False", "N/A"])
        self._add_manual_entry_field(mp_frame, "dos_at_fermi", "DOS at Fermi (MP, eV⁻¹):", CSV_HEADERS[4], entry_type='number')

        # Target Properties
        target_frame = ttk.LabelFrame(scrollable_frame, text="Target Properties (for ML model training)")
        target_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(target_frame, "target_band_gap", "Target Band Gap (eV)*:", CSV_HEADERS[20], entry_type='number')
        self._add_manual_entry_field(target_frame, "target_formation_energy", "Target Formation Energy (eV/atom)*:", CSV_HEADERS[21], entry_type='number')
        self._add_manual_entry_field(target_frame, "target_is_metal", "Target Is Metal*:", CSV_HEADERS[22], options=["True", "False", "N/A"])
        self._add_manual_entry_field(target_frame, "target_dos_at_fermi", "Target DOS at Fermi (eV⁻¹):", CSV_HEADERS[23], entry_type='number')

        # Action Buttons
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(padx=10, pady=10, fill="x")
        ttk.Button(action_frame, text="Save to Dataset", command=self.save_manual_entry).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear Fields", command=self.clear_manual_entry_fields).pack(side=tk.LEFT, padx=5)

    def _add_manual_entry_field(self, parent, key_name, label_text, schema_key_ref, options=None, entry_type='text'):
        # schema_key_ref is mostly for reference here, actual key is key_name
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(frame, text=label_text, width=30).pack(side=tk.LEFT)

        if options:
            widget = ttk.Combobox(frame, values=options, width=37)
            if "N/A" in options: widget.set("N/A")
            elif options: widget.set(options[0])
        else:
            widget = ttk.Entry(frame, width=40)

        widget.pack(side=tk.LEFT, fill="x", expand=True)
        self.manual_entry_fields[key_name] = widget

    def load_cif_for_manual_entry(self):
        filepath = filedialog.askopenfilename(
            title="Select CIF File",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if not filepath:
            return

        self.selected_manual_cif_path.set(filepath)
        self.manual_cif_path_label.config(text=os.path.basename(filepath))

        try:
            struct = Structure.from_file(filepath)

            # Auto-fill pymatgen fields
            self.manual_entry_fields['formula_pretty'].delete(0, tk.END)
            self.manual_entry_fields['formula_pretty'].insert(0, struct.composition.reduced_formula)

            self.manual_entry_fields['num_elements'].delete(0, tk.END)
            self.manual_entry_fields['num_elements'].insert(0, str(len(struct.composition.elements)))

            self.manual_entry_fields['elements'].delete(0, tk.END)
            self.manual_entry_fields['elements'].insert(0, ','.join(sorted([el.symbol for el in struct.composition.elements])))

            self.manual_entry_fields['density_pg'].delete(0, tk.END)
            self.manual_entry_fields['density_pg'].insert(0, f"{struct.density:.4f}")

            self.manual_entry_fields['volume_pg'].delete(0, tk.END)
            self.manual_entry_fields['volume_pg'].insert(0, f"{struct.volume:.4f}")

            self.manual_entry_fields['volume_per_atom_pg'].delete(0, tk.END)
            self.manual_entry_fields['volume_per_atom_pg'].insert(0, f"{struct.volume / struct.num_sites:.4f}")

            self.manual_entry_fields['spacegroup_number_pg'].delete(0, tk.END)
            self.manual_entry_fields['spacegroup_number_pg'].insert(0, str(struct.get_space_group_info()[1]))

            self.manual_entry_fields['crystal_system_pg'].set(struct.get_crystal_system())

            lat = struct.lattice
            for param_key, param_val in zip(
                ['lattice_a_pg', 'lattice_b_pg', 'lattice_c_pg', 'lattice_alpha_pg', 'lattice_beta_pg', 'lattice_gamma_pg'],
                [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma]
            ):
                self.manual_entry_fields[param_key].delete(0, tk.END)
                self.manual_entry_fields[param_key].insert(0, f"{param_val:.4f}")

            self.manual_entry_fields['num_sites_pg'].delete(0, tk.END)
            self.manual_entry_fields['num_sites_pg'].insert(0, str(struct.num_sites))

            messagebox.showinfo("CIF Loaded", "Pymatgen-derived features have been auto-filled.")

        except Exception as e:
            messagebox.showerror("CIF Parsing Error", f"Could not read or parse CIF file: {e}")
            self.manual_cif_path_label.config(text="CIF loading failed.")


    def save_manual_entry(self):
        data_row = {}
        try:
            # Validate and collect data
            # Identifier (required)
            material_id = self.manual_entry_fields['material_id'].get()
            if not material_id.strip():
                messagebox.showerror("Validation Error", "Material ID is required.")
                return
            data_row['material_id'] = material_id.strip()

            # Pymatgen features
            for key in ["formula_pretty", "elements", "crystal_system_pg"]: # String/Combobox
                 data_row[key] = self.manual_entry_fields[key].get()

            pg_numeric_keys = [
                "num_elements", "density_pg", "volume_pg", "volume_per_atom_pg", "spacegroup_number_pg",
                "lattice_a_pg", "lattice_b_pg", "lattice_c_pg", "lattice_alpha_pg", "lattice_beta_pg",
                "lattice_gamma_pg", "num_sites_pg"
            ]
            for key in pg_numeric_keys:
                val_str = self.manual_entry_fields[key].get()
                data_row[key] = float(val_str) if val_str else '' # Allow empty for non-required

            # Externally Sourced / MP Features
            mp_numeric_keys = ["band_gap_mp", "formation_energy_per_atom_mp", "dos_at_fermi"]
            for key in mp_numeric_keys:
                val_str = self.manual_entry_fields[key].get()
                data_row[key] = float(val_str) if val_str else ''

            is_metal_val = self.manual_entry_fields['is_metal'].get()
            data_row['is_metal'] = is_metal_val if is_metal_val != "N/A" else ''


            # Target Properties (validate required ones)
            target_band_gap_str = self.manual_entry_fields['target_band_gap'].get()
            if not target_band_gap_str: messagebox.showerror("Validation Error", "Target Band Gap is required."); return
            data_row['target_band_gap'] = float(target_band_gap_str)

            target_formation_energy_str = self.manual_entry_fields['target_formation_energy'].get()
            if not target_formation_energy_str: messagebox.showerror("Validation Error", "Target Formation Energy is required."); return
            data_row['target_formation_energy'] = float(target_formation_energy_str)

            target_is_metal_val = self.manual_entry_fields['target_is_metal'].get()
            if not target_is_metal_val or target_is_metal_val == "N/A": messagebox.showerror("Validation Error", "Target Is Metal is required."); return
            data_row['target_is_metal'] = target_is_metal_val

            target_dos_str = self.manual_entry_fields['target_dos_at_fermi'].get()
            data_row['target_dos_at_fermi'] = float(target_dos_str) if target_dos_str else ''


        except ValueError as ve:
            messagebox.showerror("Validation Error", f"Invalid number format for one of the fields: {ve}")
            return
        except Exception as ex:
            messagebox.showerror("Error", f"An unexpected error occurred: {ex}")
            return

        # Save to CSV
        csv_filename = "Fe_materials_dataset.csv"
        file_exists = os.path.isfile(csv_filename)

        try:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
                if not file_exists or os.path.getsize(csv_filename) == 0:
                    writer.writeheader() # Write header if file is new or empty

                # Ensure all CSV_HEADERS are present in data_row, fill with empty string if not
                for header_key in CSV_HEADERS:
                    if header_key not in data_row:
                        data_row[header_key] = ''
                writer.writerow(data_row)
            messagebox.showinfo("Success", f"Data for {material_id} saved to {csv_filename}")
            self.clear_manual_entry_fields() # Optionally clear after successful save
        except Exception as e:
            messagebox.showerror("CSV Write Error", f"Could not save data to CSV: {e}")


    def clear_manual_entry_fields(self):
        for key, widget in self.manual_entry_fields.items():
            if isinstance(widget, ttk.Combobox):
                if "N/A" in widget['values']: widget.set("N/A") # Corrected widgetc to widget
                else: widget.set('') # Or set to first option if that's preferred
            else: # Entry widgets
                widget.delete(0, tk.END)
        self.manual_cif_path_label.config(text="No CIF file loaded.")
        self.selected_manual_cif_path.set("")


    # --- Methods for Prediction Tab (slight refactor for clarity) ---
    def select_file_for_prediction(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".cif",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.clear_prediction_labels() # Clear old predictions
        else:
            self.selected_file_path = ""
            self.file_label.config(text="No file selected")
            self.clear_prediction_labels()

    def clear_prediction_labels(self):
        # Pymatgen derived
        self.formula_label_value.config(text="N/A")
        self.density_label_value.config(text="N/A")
        self.volume_label_value.config(text="N/A")
        # Model predictions
        self.model_is_metal_label.config(text="N/A")
        self.model_band_gap_label.config(text="N/A")
        self.model_formation_energy_label.config(text="N/A")
        self.model_dos_at_fermi_label.config(text="N/A")


    def perform_prediction(self):
        if not self.selected_file_path or not hasattr(self, 'preprocessor_main'): # Check if preprocessor loaded
            messagebox.showinfo("Setup Incomplete", "Please select a CIF file and ensure models are loaded.")
            self.clear_prediction_labels()
            return

        try:
            s = Structure.from_file(self.selected_file_path)
            self.current_structure = s # Store for potential later use, though direct features are used below

            # Update Pymatgen-derived feature labels
            self.formula_label_value.config(text=s.composition.reduced_formula)
            self.density_label_value.config(text=f"{s.density:.4f} g/cm³")
            self.volume_label_value.config(text=f"{s.volume:.4f} Å³")

            # Prepare data for model prediction (single row DataFrame)
            # Must match features used in train_model.py for preprocessor_main
            feature_data = {
                'num_elements': len(s.composition.elements),
                'density_pg': s.density,
                'volume_pg': s.volume,
                'volume_per_atom_pg': s.volume / s.num_sites,
                'spacegroup_number_pg': s.get_space_group_info()[1],
                'lattice_a_pg': s.lattice.a,
                'lattice_b_pg': s.lattice.b,
                'lattice_c_pg': s.lattice.c,
                'lattice_alpha_pg': s.lattice.alpha,
                'lattice_beta_pg': s.lattice.beta,
                'lattice_gamma_pg': s.lattice.gamma,
                'num_sites_pg': s.num_sites,
                'elements': ','.join(sorted([el.symbol for el in s.composition.elements])),
                'crystal_system_pg': s.get_crystal_system().lower(),
                # The 'band_gap_mp' and 'formation_energy_per_atom_mp' were features in training
                # but they are what we predict or similar to. For a pure CIF prediction,
                # these would not be available as inputs.
                # We should use only pymatgen derivable features for prediction from CIF.
                # The training script's numerical_features_main needs to be consistent.
                # Assuming for now these _mp features are NOT used by preprocessor_main for CIF-only prediction.
            }

            # Define numerical and categorical features exactly as in train_model.py for main models
            # These are the features the preprocessor_main was trained on.
            numerical_features_for_main_pred = [
                'num_elements', 'density_pg', 'volume_pg', 'volume_per_atom_pg',
                'spacegroup_number_pg', 'lattice_a_pg', 'lattice_b_pg', 'lattice_c_pg',
                'lattice_alpha_pg', 'lattice_beta_pg', 'lattice_gamma_pg', 'num_sites_pg'
            ] # These should NOT include band_gap_mp or formation_energy_per_atom_mp if predicting from CIF alone
            categorical_features_for_main_pred = ['elements', 'crystal_system_pg']

            df_predict = pd.DataFrame([feature_data], columns=numerical_features_for_main_pred + categorical_features_for_main_pred)

            predicted_is_metal = None # To guide DOS prediction

            # Predict is_metal
            if self.preprocessor_main and self.model_is_metal:
                try:
                    processed_data_main = self.preprocessor_main.transform(df_predict)
                    pred_is_metal_val = self.model_is_metal.predict(processed_data_main)[0]
                    pred_is_metal_proba = self.model_is_metal.predict_proba(processed_data_main)[0]
                    predicted_is_metal = bool(pred_is_metal_val)
                    self.model_is_metal_label.config(text=f"{predicted_is_metal} (Prob: {pred_is_metal_proba[pred_is_metal_val]:.2f})")
                except Exception as e:
                    self.model_is_metal_label.config(text="Error")
                    warnings.warn(f"Error predicting is_metal: {e}")
            else:
                self.model_is_metal_label.config(text="N/A (model not loaded)")

            # Predict band_gap
            if self.preprocessor_main and self.model_band_gap:
                try:
                    # processed_data_main would have been computed during 'is_metal' prediction if models loaded.
                    # Recompute or ensure it's available if 'is_metal' part was skipped.
                    if not 'processed_data_main' in locals() and self.preprocessor_main: # Check if it exists in local scope
                         processed_data_main = self.preprocessor_main.transform(df_predict)

                    pred_band_gap = self.model_band_gap.predict(processed_data_main)[0]
                    self.model_band_gap_label.config(text=f"{pred_band_gap:.3f} eV")
                except Exception as e:
                    self.model_band_gap_label.config(text="Error")
                    warnings.warn(f"Error predicting band_gap: {e}")
            else:
                self.model_band_gap_label.config(text="N/A (model not loaded)")

            # Predict formation_energy
            if self.preprocessor_main and self.model_formation_energy:
                try:
                    # Ensure processed_data_main is available
                    if not 'processed_data_main' in locals() and self.preprocessor_main:
                         processed_data_main = self.preprocessor_main.transform(df_predict)

                    pred_form_energy = self.model_formation_energy.predict(processed_data_main)[0]
                    self.model_formation_energy_label.config(text=f"{pred_form_energy:.3f} eV/atom")
                except Exception as e:
                    self.model_formation_energy_label.config(text="Error")
                    warnings.warn(f"Error predicting formation_energy: {e}")
            else:
                self.model_formation_energy_label.config(text="N/A (model not loaded)")

            # Predict dos_at_fermi (only if predicted as metal and models are available)
            if predicted_is_metal is True and self.preprocessor_dos and self.model_dos_at_fermi:
                 # Features for DOS model (must match training)
                numerical_features_for_dos_pred = [
                    'num_elements', 'density_pg', 'volume_pg', 'volume_per_atom_pg',
                    'spacegroup_number_pg', 'lattice_a_pg', 'lattice_b_pg', 'lattice_c_pg',
                    'lattice_alpha_pg', 'lattice_beta_pg', 'lattice_gamma_pg', 'num_sites_pg'
                    # Crucially, these must be the same features preprocessor_dos was trained on.
                    # If band_gap_mp or formation_energy_mp were part of its training features,
                    # they are not available here. This implies preprocessor_dos and model_dos_at_fermi
                    # should have been trained *without* them if predicting from pure CIF.
                    # For now, assume they are consistent with features_for_main_pred.
                ]
                df_predict_dos = pd.DataFrame([feature_data], columns=numerical_features_for_dos_pred + categorical_features_for_main_pred)
                try:
                    processed_data_dos = self.preprocessor_dos.transform(df_predict_dos)
                    pred_dos_val = self.model_dos_at_fermi.predict(processed_data_dos)[0]
                    self.model_dos_at_fermi_label.config(text=f"{pred_dos_val:.3f} eV⁻¹")
                except Exception as e:
                    self.model_dos_at_fermi_label.config(text="Error")
                    warnings.warn(f"Error predicting dos_at_fermi: {e}")
            elif predicted_is_metal is False:
                 self.model_dos_at_fermi_label.config(text="N/A (predicted non-metal)")
            else:
                self.model_dos_at_fermi_label.config(text="N/A (model/preprocessor error or not metal)")

        except FileNotFoundError:
            messagebox.showerror("CIF Error", "CIF file not found at the selected path.")
            self.clear_prediction_labels()
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not perform prediction: {e}")
            self.clear_prediction_labels()
            # Also clear pymatgen derived features if structure parsing itself failed
            self.formula_label_value.config(text="Error")
            self.density_label_value.config(text="Error")
            self.volume_label_value.config(text="Error")


    # Placeholder get_placeholder_predictions removed as it's no longer used by perform_prediction
    # def get_placeholder_predictions(self, structure): ...


if __name__ == '__main__':
    app = Application()
    app.mainloop()
