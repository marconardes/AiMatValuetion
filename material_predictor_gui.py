import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox # Ensure messagebox is imported
from pymatgen.core import Structure # Import Structure

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Material Property Predictor Prototype")
        self.selected_file_path = ""
        self.current_structure = None # To store the parsed pymatgen structure
        self.extracted_formula = "N/A"
        self.extracted_density = "N/A"
        self.extracted_volume = "N/A"

        # Optionally set window size
        # self.geometry("800x600")

        self.select_button = tk.Button(self, text="Select CIF File", command=self.select_file)
        self.select_button.pack(pady=10)

        self.file_label = tk.Label(self, text="No file selected")
        self.file_label.pack(pady=5)

        self.predict_button = tk.Button(self, text="Predict", command=self.perform_prediction)
        self.predict_button.pack(pady=10)

        # Labels for displaying predictions
        self.band_gap_label_title = tk.Label(self, text="Band Gap:")
        self.band_gap_label_title.pack()
        self.band_gap_label_value = tk.Label(self, text="N/A")
        self.band_gap_label_value.pack()

        self.dos_label_title = tk.Label(self, text="DOS:")
        self.dos_label_title.pack()
        self.dos_label_value = tk.Label(self, text="N/A")
        self.dos_label_value.pack()

        self.formation_energy_label_title = tk.Label(self, text="Formation Energy:")
        self.formation_energy_label_title.pack()
        self.formation_energy_label_value = tk.Label(self, text="N/A")
        self.formation_energy_label_value.pack(pady=(0,5))

        # Labels for pymatgen-derived properties
        self.formula_label_title = tk.Label(self, text="Chemical Formula:")
        self.formula_label_title.pack()
        self.formula_label_value = tk.Label(self, text="N/A")
        self.formula_label_value.pack()

        self.density_label_title = tk.Label(self, text="Density:")
        self.density_label_title.pack()
        self.density_label_value = tk.Label(self, text="N/A")
        self.density_label_value.pack()

        self.volume_label_title = tk.Label(self, text="Cell Volume:")
        self.volume_label_title.pack()
        self.volume_label_value = tk.Label(self, text="N/A")
        self.volume_label_value.pack()


    def select_file(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".cif",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.config(text=self.selected_file_path)
        else:
            self.selected_file_path = ""
            self.file_label.config(text="No file selected")
            self.clear_predictions()

    def clear_predictions(self):
        self.band_gap_label_value.config(text="N/A")
        self.dos_label_value.config(text="N/A")
        self.formation_energy_label_value.config(text="N/A")
        self.formula_label_value.config(text="N/A")
        self.density_label_value.config(text="N/A")
        self.volume_label_value.config(text="N/A")

    def perform_prediction(self):
        # messagebox is imported at the top of the file.
        if not self.selected_file_path:
            messagebox.showinfo("No File Selected", "Please select a CIF file first.")
            self.clear_predictions()
            return

        # CIF Parsing Block
        try:
            self.current_structure = Structure.from_file(self.selected_file_path)
            self.extracted_formula = self.current_structure.formula
            # Format density and volume to a few decimal places
            self.extracted_density = f"{self.current_structure.density:.4f} g/cm³"
            self.extracted_volume = f"{self.current_structure.volume:.4f} Å³"

            # Update GUI labels for CIF properties immediately after successful parsing
            self.formula_label_value.config(text=self.extracted_formula)
            self.density_label_value.config(text=self.extracted_density)
            self.volume_label_value.config(text=self.extracted_volume)

        except Exception as e:
            messagebox.showerror("CIF Parsing Error", f"Could not read CIF file: {e}")
            self.current_structure = None
            self.extracted_formula = "Error"
            self.extracted_density = "Error"
            self.extracted_volume = "Error"
            # Update GUI labels to show error state
            self.formula_label_value.config(text=self.extracted_formula)
            self.density_label_value.config(text=self.extracted_density)
            self.volume_label_value.config(text=self.extracted_volume)
            # Clear other prediction labels as well
            self.band_gap_label_value.config(text="N/A")
            self.dos_label_value.config(text="N/A")
            self.formation_energy_label_value.config(text="N/A")
            return # Stop further processing if CIF parsing fails

        # Proceed with placeholder predictions ONLY if CIF parsing was successful
        predictions = self.get_placeholder_predictions(self.current_structure) # Pass the structure object

        self.band_gap_label_value.config(text=predictions.get("band_gap", "N/A"))
        self.dos_label_value.config(text=predictions.get("dos", "N/A"))
        self.formation_energy_label_value.config(text=predictions.get("formation_energy", "N/A"))


    def get_placeholder_predictions(self, structure): # Argument changed to structure
        if structure is None:
            return {"band_gap": "N/A", "dos": "N/A", "formation_energy": "N/A"}

        # Use reduced formula for lookup
        formula = structure.composition.reduced_formula

        placeholder_data = {
            "Si": {"band_gap": "1.1 eV", "dos": "Si_dos.png", "formation_energy": "0 eV/atom"},
            "GaAs": {"band_gap": "1.4 eV", "dos": "GaAs_dos.png", "formation_energy": "-0.7 eV/atom"},
            "NaCl": {"band_gap": "8.5 eV", "dos": "NaCl_dos.png", "formation_energy": "-3.0 eV/atom"},
            "CsPbI3": {"band_gap": "1.9 eV", "dos": "CsPbI3_dos.png", "formation_energy": "-0.9 eV/atom"}
            # Example: CsPbI3 formation energy updated slightly
        }

        if formula in placeholder_data:
            return placeholder_data[formula]
        else:
            return {"band_gap": "N/A", "dos": "N/A", "formation_energy": "N/A"}

if __name__ == '__main__':
    app = Application()
    app.mainloop()
