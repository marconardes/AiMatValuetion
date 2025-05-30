import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from pymatgen.core import Structure
import csv
import os
import joblib
import pandas as pd
import numpy as np
import warnings
# import yaml # No longer needed here, will use centralized loader
from utils.config_loader import load_config
from utils.schema import MANUAL_ENTRY_CSV_HEADERS # Added

# MANUAL_ENTRY_CSV_HEADERS is now imported from utils.schema

class PredictionTab(ttk.Frame):
    def __init__(self, parent, loaded_models_dict, **kwargs):
        super().__init__(parent, **kwargs)

        self.preprocessor_main = loaded_models_dict.get('preprocessor_main')
        self.preprocessor_dos = loaded_models_dict.get('preprocessor_dos')
        self.model_is_metal = loaded_models_dict.get('model_is_metal')
        self.model_band_gap = loaded_models_dict.get('model_band_gap')
        self.model_formation_energy = loaded_models_dict.get('model_formation_energy')
        self.model_dos_at_fermi = loaded_models_dict.get('model_dos_at_fermi')

        self.current_structure = None
        self.selected_file_path = None

        self._setup_widgets()

    def _setup_widgets(self):
        controls_frame = ttk.Frame(self)
        controls_frame.pack(pady=10, padx=10, fill='x')

        self.btn_select_file = ttk.Button(controls_frame, text="Select CIF File", command=self.select_file_for_prediction)
        self.btn_select_file.pack(side=tk.LEFT, padx=(0,10))

        self.file_path_label = ttk.Label(controls_frame, text="No file selected")
        self.file_path_label.pack(side=tk.LEFT, fill='x', expand=True)

        self.btn_predict = ttk.Button(self, text="Predict Properties", command=self.perform_prediction)
        self.btn_predict.pack(pady=10, padx=10)

        results_display_frame = ttk.LabelFrame(self, text="Results")
        results_display_frame.pack(padx=10, pady=10, fill="both", expand=True)

        pg_features_frame = ttk.LabelFrame(results_display_frame, text="Pymatgen-Derived Features")
        pg_features_frame.pack(padx=10, pady=5, fill="x", expand=True)
        self.formula_label = self._create_display_widget_pair(pg_features_frame, "Chemical Formula:")
        self.density_label = self._create_display_widget_pair(pg_features_frame, "Density:")
        self.volume_label = self._create_display_widget_pair(pg_features_frame, "Cell Volume:")

        model_preds_frame = ttk.LabelFrame(results_display_frame, text="Model Predictions")
        model_preds_frame.pack(padx=10, pady=5, fill="x", expand=True)
        self.model_is_metal_label = self._create_display_widget_pair(model_preds_frame, "Is Metal:")
        self.model_band_gap_label = self._create_display_widget_pair(model_preds_frame, "Band Gap (Model):")
        self.model_formation_energy_label = self._create_display_widget_pair(model_preds_frame, "Formation Energy (Model):")
        self.model_dos_at_fermi_label = self._create_display_widget_pair(model_preds_frame, "DOS at Fermi (Model):")

    def _create_display_widget_pair(self, parent, text_label):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(frame, text=text_label, width=25, anchor='w').pack(side=tk.LEFT)
        value_label = ttk.Label(frame, text="N/A", anchor='w')
        value_label.pack(side=tk.LEFT, fill='x', expand=True)
        return value_label

    def select_file_for_prediction(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".cif",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_path_label.config(text=os.path.basename(file_path))
            self.clear_prediction_labels()
            try:
                self.current_structure = Structure.from_file(self.selected_file_path)
                self.formula_label.config(text=self.current_structure.composition.reduced_formula)
                self.density_label.config(text=f"{self.current_structure.density:.4f} g/cm³")
                self.volume_label.config(text=f"{self.current_structure.volume:.4f} Å³")
            except Exception as e:
                messagebox.showerror("CIF Parsing Error", f"Could not read/parse CIF: {e}")
                self.selected_file_path = None
                self.current_structure = None
                self.file_path_label.config(text="Invalid CIF selected")
                self.clear_prediction_labels()
        else:
            self.selected_file_path = None
            self.current_structure = None
            self.file_path_label.config(text="No file selected")
            self.clear_prediction_labels()

    def perform_prediction(self):
        if not self.selected_file_path or not self.current_structure:
            messagebox.showinfo("Setup Incomplete", "Please select a valid CIF file first.")
            return
        if not self.preprocessor_main:
             messagebox.showwarning("Model Error", "Main preprocessor not loaded. Cannot perform predictions.")
             return

        s = self.current_structure

        try:
            feature_data = {
                'num_elements': len(s.composition.elements), 'density_pg': s.density,
                'volume_pg': s.volume, 'volume_per_atom_pg': s.volume / s.num_sites,
                'spacegroup_number_pg': s.get_space_group_info()[1],
                'lattice_a_pg': s.lattice.a, 'lattice_b_pg': s.lattice.b, 'lattice_c_pg': s.lattice.c,
                'lattice_alpha_pg': s.lattice.alpha, 'lattice_beta_pg': s.lattice.beta, 'lattice_gamma_pg': s.lattice.gamma,
                'num_sites_pg': s.num_sites,
                'elements': ','.join(sorted([el.symbol for el in s.composition.elements])),
                'crystal_system_pg': s.get_crystal_system().lower(),
            }
            numerical_features_for_main_pred = [
                'num_elements', 'density_pg', 'volume_pg', 'volume_per_atom_pg',
                'spacegroup_number_pg', 'lattice_a_pg', 'lattice_b_pg', 'lattice_c_pg',
                'lattice_alpha_pg', 'lattice_beta_pg', 'lattice_gamma_pg', 'num_sites_pg'
            ]
            categorical_features_for_main_pred = ['elements', 'crystal_system_pg']
            df_predict = pd.DataFrame([feature_data], columns=numerical_features_for_main_pred + categorical_features_for_main_pred)

            predicted_is_metal = None
            processed_data_main = None

            if self.preprocessor_main:
                 processed_data_main = self.preprocessor_main.transform(df_predict)

            if self.model_is_metal and processed_data_main is not None:
                try:
                    pred_is_metal_val = self.model_is_metal.predict(processed_data_main)[0]
                    pred_is_metal_proba = self.model_is_metal.predict_proba(processed_data_main)[0]
                    predicted_is_metal = bool(pred_is_metal_val)
                    self.model_is_metal_label.config(text=f"{predicted_is_metal} (Prob: {pred_is_metal_proba[pred_is_metal_val]:.2f})")
                except Exception as e:
                    self.model_is_metal_label.config(text="Error")
                    warnings.warn(f"Error predicting is_metal: {e}")
            else:
                self.model_is_metal_label.config(text="N/A (model/preprocessor error)")

            if self.model_band_gap and processed_data_main is not None:
                try:
                    pred_band_gap = self.model_band_gap.predict(processed_data_main)[0]
                    self.model_band_gap_label.config(text=f"{pred_band_gap:.3f} eV")
                except Exception as e:
                    self.model_band_gap_label.config(text="Error")
                    warnings.warn(f"Error predicting band_gap: {e}")
            else:
                self.model_band_gap_label.config(text="N/A (model/preprocessor error)")

            if self.model_formation_energy and processed_data_main is not None:
                try:
                    pred_form_energy = self.model_formation_energy.predict(processed_data_main)[0]
                    self.model_formation_energy_label.config(text=f"{pred_form_energy:.3f} eV/atom")
                except Exception as e:
                    self.model_formation_energy_label.config(text="Error")
                    warnings.warn(f"Error predicting formation_energy: {e}")
            else:
                self.model_formation_energy_label.config(text="N/A (model/preprocessor error)")

            if predicted_is_metal is True and self.preprocessor_dos and self.model_dos_at_fermi:
                numerical_features_for_dos_pred = numerical_features_for_main_pred
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
                self.model_dos_at_fermi_label.config(text="N/A")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred: {e}")
            self.clear_prediction_labels()

    def clear_prediction_labels(self):
        self.formula_label.config(text="N/A")
        self.density_label.config(text="N/A")
        self.volume_label.config(text="N/A")
        self.model_is_metal_label.config(text="N/A")
        self.model_band_gap_label.config(text="N/A")
        self.model_formation_energy_label.config(text="N/A")
        self.model_dos_at_fermi_label.config(text="N/A")


class ManualEntryTab(ttk.Frame):
    def __init__(self, parent, **kwargs):
        # Extract app_config from kwargs. If it's not there, default to an empty dict.
        self.app_config = kwargs.pop('app_config', {})
        super().__init__(parent, **kwargs) # Call super with cleaned kwargs
        # self.parent_notebook = parent # parent is already passed to super, ttk.Frame handles it.

        self.CSV_HEADERS = MANUAL_ENTRY_CSV_HEADERS # Use the global one for consistency

        self.manual_entry_fields = {}
        self.loaded_cif_path_manual_var = tk.StringVar(value="No CIF loaded.") # Used by a label
        # self.app_config is now set from kwargs

        self._setup_widgets()

    def _setup_widgets(self):
        # Scrollable Frame Setup
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Labelframes
        id_frame = ttk.LabelFrame(scrollable_frame, text="Identifier")
        id_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(id_frame, "material_id", "Material ID*:", self.CSV_HEADERS[0])

        load_cif_frame = ttk.LabelFrame(scrollable_frame, text="Load CIF for Feature Extraction")
        load_cif_frame.pack(padx=10, pady=5, fill="x")
        ttk.Button(load_cif_frame, text="Load CIF File", command=self.load_cif_for_manual_entry).pack(side=tk.LEFT, padx=5, pady=5)
        self.cif_path_display_label = ttk.Label(load_cif_frame, textvariable=self.loaded_cif_path_manual_var)
        self.cif_path_display_label.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)

        pg_frame = ttk.LabelFrame(scrollable_frame, text="Pymatgen-Derived Features")
        pg_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(pg_frame, "formula_pretty", "Formula (Pretty):", self.CSV_HEADERS[5])
        self._add_manual_entry_field(pg_frame, "num_elements", "Num Elements:", self.CSV_HEADERS[6], entry_type='number')
        self._add_manual_entry_field(pg_frame, "elements", "Elements (comma-sep):", self.CSV_HEADERS[7])
        self._add_manual_entry_field(pg_frame, "density_pg", "Density (g/cm³):", self.CSV_HEADERS[8], entry_type='number')
        self._add_manual_entry_field(pg_frame, "volume_pg", "Cell Volume (Å³):", self.CSV_HEADERS[9], entry_type='number')
        self._add_manual_entry_field(pg_frame, "volume_per_atom_pg", "Vol/Atom (Å³/atom):", self.CSV_HEADERS[10], entry_type='number')
        self._add_manual_entry_field(pg_frame, "spacegroup_number_pg", "Space Group Num:", self.CSV_HEADERS[11], entry_type='number')
        crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic', 'N/A']
        self._add_manual_entry_field(pg_frame, "crystal_system_pg", "Crystal System:", self.CSV_HEADERS[12], options=crystal_systems)
        # Lattice params
        lp_frame = ttk.Frame(pg_frame) # Sub-frame for better layout of lattice params
        lp_frame.pack(fill='x')
        self._add_manual_entry_field(lp_frame, "lattice_a_pg", "Lattice a (Å):", self.CSV_HEADERS[13], entry_type='number', width=10)
        self._add_manual_entry_field(lp_frame, "lattice_b_pg", "Lattice b (Å):", self.CSV_HEADERS[14], entry_type='number', width=10)
        self._add_manual_entry_field(lp_frame, "lattice_c_pg", "Lattice c (Å):", self.CSV_HEADERS[15], entry_type='number', width=10)
        self._add_manual_entry_field(lp_frame, "lattice_alpha_pg", "Lattice α (°):", self.CSV_HEADERS[16], entry_type='number', width=10)
        self._add_manual_entry_field(lp_frame, "lattice_beta_pg", "Lattice β (°):", self.CSV_HEADERS[17], entry_type='number', width=10)
        self._add_manual_entry_field(lp_frame, "lattice_gamma_pg", "Lattice γ (°):", self.CSV_HEADERS[18], entry_type='number', width=10)
        self._add_manual_entry_field(pg_frame, "num_sites_pg", "Num Sites:", self.CSV_HEADERS[19], entry_type='number')


        mp_frame = ttk.LabelFrame(scrollable_frame, text="Externally Sourced Properties")
        mp_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(mp_frame, "band_gap_mp", "Band Gap (MP, eV):", self.CSV_HEADERS[1], entry_type='number')
        self._add_manual_entry_field(mp_frame, "formation_energy_per_atom_mp", "Form. Energy (MP, eV/atom):", self.CSV_HEADERS[2], entry_type='number')
        self._add_manual_entry_field(mp_frame, "is_metal", "Is Metal (MP):", self.CSV_HEADERS[3], options=["True", "False", "N/A"])
        self._add_manual_entry_field(mp_frame, "dos_at_fermi", "DOS at Fermi (MP, eV⁻¹):", self.CSV_HEADERS[4], entry_type='number')

        target_frame = ttk.LabelFrame(scrollable_frame, text="Target Properties (for ML)")
        target_frame.pack(padx=10, pady=5, fill="x")
        self._add_manual_entry_field(target_frame, "target_band_gap", "Target Band Gap (eV)*:", self.CSV_HEADERS[20], entry_type='number')
        self._add_manual_entry_field(target_frame, "target_formation_energy", "Target Form. Energy (eV/atom)*:", self.CSV_HEADERS[21], entry_type='number')
        self._add_manual_entry_field(target_frame, "target_is_metal", "Target Is Metal*:", self.CSV_HEADERS[22], options=["True", "False", "N/A"])
        self._add_manual_entry_field(target_frame, "target_dos_at_fermi", "Target DOS at Fermi (eV⁻¹):", self.CSV_HEADERS[23], entry_type='number')

        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(padx=10, pady=10, fill="x", side=tk.BOTTOM) # Ensure it's at the bottom
        ttk.Button(action_frame, text="Save to Dataset", command=self.save_manual_entry).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear Fields", command=self.clear_manual_entry_fields).pack(side=tk.LEFT, padx=5)

    def _add_manual_entry_field(self, parent, key_name, label_text, schema_key_ref, options=None, entry_type='text', width=37):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(frame, text=label_text, width=30).pack(side=tk.LEFT)

        widget_width = width if entry_type == 'text' or options else 15 # Smaller width for numbers if not specified
        if options:
            widget = ttk.Combobox(frame, values=options, width=widget_width, state="readonly")
            if "N/A" in options: widget.set("N/A")
            elif options: widget.set(options[0])
        else:
            widget = ttk.Entry(frame, width=widget_width + 3) # Entry width seems to behave differently

        widget.pack(side=tk.LEFT, padx=(0,5)) # Add some padding
        self.manual_entry_fields[key_name] = widget

    def load_cif_for_manual_entry(self):
        filepath = filedialog.askopenfilename(title="Select CIF File", filetypes=[("CIF files", "*.cif"), ("All files", "*.*")])
        if not filepath: return

        self.loaded_cif_path_manual_var.set(os.path.basename(filepath))
        try:
            struct = Structure.from_file(filepath)
            self.manual_entry_fields['formula_pretty'].delete(0, tk.END); self.manual_entry_fields['formula_pretty'].insert(0, struct.composition.reduced_formula)
            self.manual_entry_fields['num_elements'].delete(0, tk.END); self.manual_entry_fields['num_elements'].insert(0, str(len(struct.composition.elements)))
            self.manual_entry_fields['elements'].delete(0, tk.END); self.manual_entry_fields['elements'].insert(0, ','.join(sorted([el.symbol for el in struct.composition.elements])))
            self.manual_entry_fields['density_pg'].delete(0, tk.END); self.manual_entry_fields['density_pg'].insert(0, f"{struct.density:.4f}")
            self.manual_entry_fields['volume_pg'].delete(0, tk.END); self.manual_entry_fields['volume_pg'].insert(0, f"{struct.volume:.4f}")
            self.manual_entry_fields['volume_per_atom_pg'].delete(0, tk.END); self.manual_entry_fields['volume_per_atom_pg'].insert(0, f"{struct.volume / struct.num_sites:.4f}")
            self.manual_entry_fields['spacegroup_number_pg'].delete(0, tk.END); self.manual_entry_fields['spacegroup_number_pg'].insert(0, str(struct.get_space_group_info()[1]))
            self.manual_entry_fields['crystal_system_pg'].set(struct.get_crystal_system().lower())
            lat = struct.lattice
            self.manual_entry_fields['lattice_a_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_a_pg'].insert(0, f"{lat.a:.4f}")
            self.manual_entry_fields['lattice_b_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_b_pg'].insert(0, f"{lat.b:.4f}")
            self.manual_entry_fields['lattice_c_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_c_pg'].insert(0, f"{lat.c:.4f}")
            self.manual_entry_fields['lattice_alpha_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_alpha_pg'].insert(0, f"{lat.alpha:.4f}")
            self.manual_entry_fields['lattice_beta_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_beta_pg'].insert(0, f"{lat.beta:.4f}")
            self.manual_entry_fields['lattice_gamma_pg'].delete(0, tk.END); self.manual_entry_fields['lattice_gamma_pg'].insert(0, f"{lat.gamma:.4f}")
            self.manual_entry_fields['num_sites_pg'].delete(0, tk.END); self.manual_entry_fields['num_sites_pg'].insert(0, str(struct.num_sites))
            messagebox.showinfo("CIF Loaded", "Pymatgen-derived features auto-filled.")
        except Exception as e:
            messagebox.showerror("CIF Parsing Error", f"Could not parse CIF: {e}")
            self.loaded_cif_path_manual_var.set("CIF loading failed.")

    def save_manual_entry(self):
        data_row = {}
        try:
            material_id = self.manual_entry_fields['material_id'].get()
            if not material_id.strip(): messagebox.showerror("Validation Error", "Material ID is required."); return
            data_row['material_id'] = material_id.strip()

            for key in ["formula_pretty", "elements", "crystal_system_pg"]:
                 val = self.manual_entry_fields[key].get(); data_row[key] = val if val != "N/A" else ''

            pg_numeric_keys = ["num_elements", "density_pg", "volume_pg", "volume_per_atom_pg", "spacegroup_number_pg", "lattice_a_pg", "lattice_b_pg", "lattice_c_pg", "lattice_alpha_pg", "lattice_beta_pg", "lattice_gamma_pg", "num_sites_pg"]
            for key in pg_numeric_keys:
                val_str = self.manual_entry_fields[key].get(); data_row[key] = float(val_str) if val_str.strip() else ''

            mp_numeric_keys = ["band_gap_mp", "formation_energy_per_atom_mp", "dos_at_fermi"]
            for key in mp_numeric_keys:
                val_str = self.manual_entry_fields[key].get(); data_row[key] = float(val_str) if val_str.strip() else ''

            is_metal_val = self.manual_entry_fields['is_metal'].get(); data_row['is_metal'] = is_metal_val if is_metal_val != "N/A" else ''

            target_band_gap_str = self.manual_entry_fields['target_band_gap'].get()
            if not target_band_gap_str.strip(): messagebox.showerror("Validation Error", "Target Band Gap is required."); return
            data_row['target_band_gap'] = float(target_band_gap_str)

            target_formation_energy_str = self.manual_entry_fields['target_formation_energy'].get()
            if not target_formation_energy_str.strip(): messagebox.showerror("Validation Error", "Target Formation Energy is required."); return
            data_row['target_formation_energy'] = float(target_formation_energy_str)

            target_is_metal_val = self.manual_entry_fields['target_is_metal'].get()
            if not target_is_metal_val or target_is_metal_val == "N/A": messagebox.showerror("Validation Error", "Target Is Metal is required."); return
            data_row['target_is_metal'] = target_is_metal_val

            target_dos_str = self.manual_entry_fields['target_dos_at_fermi'].get()
            data_row['target_dos_at_fermi'] = float(target_dos_str) if target_dos_str.strip() else ''
        except ValueError as ve: messagebox.showerror("Validation Error", f"Invalid number: {ve}"); return
        except Exception as ex: messagebox.showerror("Error", f"Unexpected error: {ex}"); return

        # Get filename from app's config (passed to ManualEntryTab or reloaded)
        # Assuming self.app_config holds the 'gui' part of the main config
        csv_filename = self.app_config.get('manual_entry_csv_filename', "Fe_materials_dataset.csv")

        file_exists = os.path.isfile(csv_filename)
        try:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.CSV_HEADERS)
                if not file_exists or os.path.getsize(csv_filename) == 0: writer.writeheader()
                final_row = {k: data_row.get(k, '') for k in self.CSV_HEADERS} # Ensure all headers present
                writer.writerow(final_row)
            messagebox.showinfo("Success", f"Data for {material_id} saved to {csv_filename}")
            self.clear_manual_entry_fields()
        except Exception as e: messagebox.showerror("CSV Write Error", f"Could not save to CSV: {e}")

    def clear_manual_entry_fields(self):
        for key, widget in self.manual_entry_fields.items():
            if isinstance(widget, ttk.Combobox):
                if "N/A" in widget['values']: widget.set("N/A")
                elif widget['values']: widget.set(widget['values'][0])
                else: widget.set('')
            else: widget.delete(0, tk.END)
        self.loaded_cif_path_manual_var.set("No CIF loaded.")


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        full_config = load_config() # Use the new centralized loader
        self.gui_config = full_config.get('gui', {}) if full_config else {}

        app_title = self.gui_config.get('title', "Material Property Predictor Prototype") # Default if gui_config is empty or key missing
        app_geometry = self.gui_config.get('geometry', "900x700") # Default if gui_config is empty or key missing

        self.title(app_title)
        self.geometry(app_geometry)

        self.loaded_models_dict = self.load_all_models_from_config() # Use specific config for models

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.prediction_tab_instance = PredictionTab(self.notebook, self.loaded_models_dict)
        self.notebook.add(self.prediction_tab_instance, text='Predict from CIF')

        # Instantiate the new ManualEntryTab, pass the gui_config to it
        self.manual_entry_tab_instance = ManualEntryTab(self.notebook, app_config=self.gui_config)
        self.notebook.add(self.manual_entry_tab_instance, text='Manual Data Entry')


    def load_all_models_from_config(self):
        """Loads all models and preprocessors using filenames from the 'gui' section of config."""
        print("Loading models and preprocessors for Application...")
        loaded_models = {}

        # Default model filenames if not found in config
        default_models_to_load_paths = {
            "preprocessor_main": "preprocessor_main.joblib",
            "preprocessor_dos": "preprocessor_dos_at_fermi.joblib",
            "model_is_metal": "model_target_is_metal.joblib",
            "model_band_gap": "model_target_band_gap.joblib",
            "model_formation_energy": "model_target_formation_energy.joblib",
            "model_dos_at_fermi": "model_dos_at_fermi.joblib"
        }
        # Get the dictionary mapping attribute name (e.g., "preprocessor_main") to filename
        models_config_paths = self.gui_config.get('models_to_load', default_models_to_load_paths)

        for attr_name, filename in models_config_paths.items():
            try:
                loaded_models[attr_name] = joblib.load(filename)
                print(f"Successfully loaded {filename} for Application as {attr_name}")
            except FileNotFoundError:
                warnings.warn(f"App Warning: {filename} (for {attr_name}) not found. Predictions relying on it will be unavailable.")
                loaded_models[attr_name] = None
            except Exception as e:
                warnings.warn(f"App Warning: Error loading {filename} (for {attr_name}): {e}.")
                loaded_models[attr_name] = None
        return loaded_models

    # Commented out original manual entry methods from Application class
    # def setup_manual_entry_tab(self): ...
    # def _add_manual_entry_field(self, parent, key_name, label_text, schema_key_ref, options=None, entry_type='text', width=37): ...
    # def load_cif_for_manual_entry(self): ...
    # def save_manual_entry(self): ...
    # def clear_manual_entry_fields(self): ...

if __name__ == '__main__':
    # Ensure that when ManualEntryTab is instantiated, it receives the app_config
    # Application class now handles passing its self.gui_config to ManualEntryTab.
    app = Application()
    app.mainloop()
