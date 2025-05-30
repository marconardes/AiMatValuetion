import tkinter as tk
from tkinter import filedialog

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Material Property Predictor Prototype")
        self.selected_file_path = ""

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
        self.formation_energy_label_value.pack()


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

    def perform_prediction(self):
        from tkinter import messagebox
        if not self.selected_file_path:
            messagebox.showinfo("No File Selected", "Please select a CIF file first.")
            self.clear_predictions()
            return

        predictions = self.get_placeholder_predictions(self.selected_file_path)

        self.band_gap_label_value.config(text=predictions.get("band_gap", "N/A"))
        self.dos_label_value.config(text=predictions.get("dos", "N/A"))
        self.formation_energy_label_value.config(text=predictions.get("formation_energy", "N/A"))


    def get_placeholder_predictions(self, file_path):
        import os
        base_filename = os.path.basename(file_path)

        placeholder_data = {
            "Si.cif": {"band_gap": "1.1 eV", "dos": "Si_dos.png", "formation_energy": "0 eV/atom"},
            "GaAs.cif": {"band_gap": "1.4 eV", "dos": "GaAs_dos.png", "formation_energy": "-0.7 eV/atom"},
            "NaCl.cif": {"band_gap": "8.5 eV", "dos": "NaCl_dos.png", "formation_energy": "-3.0 eV/atom"},
            "CsPbI3.cif": {"band_gap": "1.7 eV", "dos": "CsPbI3_dos.png", "formation_energy": "-0.8 eV/atom"}
        }

        if base_filename in placeholder_data:
            return placeholder_data[base_filename]
        else:
            return {"band_gap": "N/A", "dos": "N/A", "formation_energy": "N/A"}

if __name__ == '__main__':
    app = Application()
    app.mainloop()
