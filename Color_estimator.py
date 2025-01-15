import sys
import subprocess
import importlib

def install_and_import(package):

    try:
        return importlib.import_module(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Package '{package}' installed successfully. Importing...")
        return importlib.import_module(package)

required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "PyQt5": "PyQt5"
}

for package, import_name in required_packages.items():
    globals()[import_name] = install_and_import(package)

print("All dependencies are installed and imported successfully.")

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QCheckBox, QHBoxLayout, QMessageBox,
    QLineEdit, QSlider, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from scipy.interpolate import interp1d
from PyQt5.QtCore import Qt

class ColorUtils:
    def wavelength_to_rgb(wavelength):
        if 380 <= wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0
            b = 1
        elif 440 <= wavelength < 490:
            r = 0
            g = (wavelength - 440) / (490 - 440)
            b = 1
        elif 490 <= wavelength < 510:
            r = 0
            g = 1
            b = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1
            b = 0
        elif 580 <= wavelength < 645:
            r = 1
            g = -(wavelength - 645) / (645 - 580)
            b = 0
        elif 645 <= wavelength <= 750:
            r = 1
            g = 0
            b = 0
        else:
            r = g = b = 0

        if 380 <= wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 420 <= wavelength < 645:
            factor = 1.0
        elif 645 <= wavelength <= 750:
            factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        else:
            factor = 0

        r = round(r * factor, 3)
        g = round(g * factor, 3)
        b = round(b * factor, 3)
        return r, g, b

class DaltonismSimulator:
    REFERENCE_COLORS = {
        "red": np.array([255, 0, 0]),
        "green": np.array([0, 255, 0]),
        "blue": np.array([0, 0, 255]),
        "yellow": np.array([255, 255, 0]),
        "cyan": np.array([0, 255, 255]),
        "violet": np.array([174, 138, 255]),
        "orange": np.array([255, 128, 0]),
        "pink": np.array([255, 102, 204]),
        "light_blue": np.array([135, 206, 250]),
        "light_green": np.array([144, 238, 144]),
        "brown": np.array([139, 69, 19]),
        "gray": np.array([128, 128, 128]),
    }

    TRANSFORMATIONS = {
        "protanopia": {
            "red": np.array([134, 122, 57]),
            "green": np.array([246, 218, 0]),
            "blue": np.array([0, 74, 156]),
            "yellow": np.array([255, 226, 90]),
            "cyan": np.array([123, 226, 203]),
            "violet": np.array([120, 156, 254]),
            "orange": np.array([184, 164, 22]),
            "pink": np.array([129, 157, 246]),
            "light_blue": np.array([208, 206, 217]),
            "light_green": np.array([235, 215, 135]),
            "brown": np.array([100, 89, 24]),
            "gray": np.array([130, 127, 128]),
        },
        "deuteranopia": {
            "red": np.array([153, 115, 0]),
            "green": np.array([255, 211, 143]),
            "blue": np.array([0, 80, 132]),
            "yellow": np.array([255, 217, 112]),
            "cyan": np.array([102, 217, 195]),
            "violet": np.array([109, 159, 250]),
            "orange": np.array([207, 155, 0]),
            "pink": np.array([158, 157, 195]),
            "light_blue": np.array([216, 201, 230]),
            "light_green": np.array([255, 207, 155]),
            "brown": np.array([112, 84, 8]),
            "gray": np.array([140, 124, 129]),
        },
        "tritanopia": {
            "red": np.array([237, 30, 25]),
            "green": np.array([113, 236, 255]),
            "blue": np.array([0, 86, 89]),
            "yellow": np.array([255, 236, 143]),
            "cyan": np.array([102, 226, 255]),
            "violet": np.array([157, 158, 171]),
            "orange": np.array([255, 122, 129]),
            "pink": np.array([248, 120, 129]),
            "light_blue": np.array([146, 219, 237]),
            "light_green": np.array([164, 226, 244]),
            "brown": np.array([141, 64, 69]),
            "gray": np.array([129, 127, 137]),
        }
    }

    def interpolate_color(color, daltonism_type):
        distances = {ref: np.linalg.norm(color - val) for ref, val in DaltonismSimulator.REFERENCE_COLORS.items()}
        closest_colors = sorted(distances, key=distances.get)[:2]
        c1, c2 = closest_colors
        d1, d2 = distances[c1], distances[c2]
        t1, t2 = DaltonismSimulator.TRANSFORMATIONS[daltonism_type][c1], \
        DaltonismSimulator.TRANSFORMATIONS[daltonism_type][c2]
        weight1, weight2 = 1 - (d1 / (d1 + d2)), 1 - (d2 / (d1 + d2))
        return np.clip(t1 * weight1 + t2 * weight2, 0, 255).astype(int)

    def simulate_color(rgb, daltonism_type):
        return DaltonismSimulator.interpolate_color(np.array(rgb), daltonism_type)

class SpectraAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.absorbance_brightness = 1.0
        self.fluorescence_brightness = 2.5
        self.resulting_color = None
        self.image_copied = None
        self.current_spectrum_type = None
        self.current_wavelengths = None
        self.current_absorbance = None
        self.current_fluorescence = None
        self.available_emission_columns = []
        self.selected_emission_column = None

    def initUI(self):
        self.setWindowTitle('Absorbance and fluorescence color estimator')
        self.setGeometry(100, 100, 800, 500)
        self.setFixedSize(800, 500)

        main_layout = QVBoxLayout()

        self.label = QLabel("Load an absorbance or fluorescence spectrum")
        main_layout.addWidget(self.label)

        self.canvas = FigureCanvas(plt.figure(figsize=(8, 3)))
        main_layout.addWidget(self.canvas)

        combined_layout = QHBoxLayout()

        self.protanopia_checkbox = QCheckBox("Protanopia")
        self.deuteranopia_checkbox = QCheckBox("Deuteranopia")
        self.tritanopia_checkbox = QCheckBox("Tritanopia")

        self.protanopia_checkbox.setVisible(False)
        self.deuteranopia_checkbox.setVisible(False)
        self.tritanopia_checkbox.setVisible(False)

        self.protanopia_checkbox.stateChanged.connect(lambda: self.toggle_daltonism_mode(self.protanopia_checkbox))
        self.deuteranopia_checkbox.stateChanged.connect(lambda: self.toggle_daltonism_mode(self.deuteranopia_checkbox))
        self.tritanopia_checkbox.stateChanged.connect(lambda: self.toggle_daltonism_mode(self.tritanopia_checkbox))

        combined_layout.addWidget(self.protanopia_checkbox)
        combined_layout.addWidget(self.deuteranopia_checkbox)
        combined_layout.addWidget(self.tritanopia_checkbox)

        self.start_wavelength_label = QLabel("Start wavelength:")
        self.start_wavelength_label.setVisible(False)
        self.start_wavelength_box = QLineEdit()
        self.start_wavelength_box.setVisible(False)
        self.start_wavelength_box.editingFinished.connect(self.update_integration_range)
        self.start_wavelength_box.setFixedWidth(50)

        self.end_wavelength_label = QLabel("nm/End wavelength:")
        self.end_wavelength_label.setVisible(False)
        self.end_wavelength_box = QLineEdit()
        self.end_wavelength_box.setVisible(False)
        self.end_wavelength_box.editingFinished.connect(self.update_integration_range)
        self.end_wavelength_box.setFixedWidth(50)

        self.end_wavelength_suffix = QLabel("nm")
        self.end_wavelength_suffix.setVisible(False)

        self.export_button = QPushButton("Export spectrum")
        self.export_button.setVisible(False)
        self.export_button.clicked.connect(self.export_spectrum)

        combined_layout.addWidget(self.start_wavelength_label)
        combined_layout.addWidget(self.start_wavelength_box)
        combined_layout.addWidget(self.end_wavelength_label)
        combined_layout.addWidget(self.end_wavelength_box)
        combined_layout.addWidget(self.end_wavelength_suffix)
        combined_layout.addWidget(self.export_button)

        main_layout.addLayout(combined_layout)

        button_layout = QHBoxLayout()

        load_absorbance_button = QPushButton("Load absorbance")
        load_absorbance_button.clicked.connect(self.load_absorbance_file)
        button_layout.addWidget(load_absorbance_button)

        load_fluorescence_button = QPushButton("Load fluorescence")
        load_fluorescence_button.clicked.connect(self.load_fluorescence_file)
        button_layout.addWidget(load_fluorescence_button)

        self.copy_image_button = QPushButton("Copy color image")
        self.copy_image_button.clicked.connect(self.copy_image)
        self.copy_image_button.setVisible(False)
        button_layout.addWidget(self.copy_image_button)

        self.copy_rgb_button = QPushButton("Copy RGB code")
        self.copy_rgb_button.clicked.connect(self.copy_rgb_code)
        self.copy_rgb_button.setVisible(False)
        button_layout.addWidget(self.copy_rgb_button)

        self.copy_hex_button = QPushButton("Copy Hex code")
        self.copy_hex_button.clicked.connect(self.copy_hex_code)
        self.copy_hex_button.setVisible(False)
        button_layout.addWidget(self.copy_hex_button)

        main_layout.addLayout(button_layout)

        self.emission_selector = QComboBox()
        self.emission_selector.currentIndexChanged.connect(self.change_emission_spectrum)
        self.emission_selector.setVisible(False)
        main_layout.addWidget(self.emission_selector)

        self.setLayout(main_layout)

    def toggle_daltonism_mode(self, checkbox):
        if checkbox.isChecked():
            if checkbox == self.protanopia_checkbox:
                self.deuteranopia_checkbox.setChecked(False)
                self.tritanopia_checkbox.setChecked(False)
            elif checkbox == self.deuteranopia_checkbox:
                self.protanopia_checkbox.setChecked(False)
                self.tritanopia_checkbox.setChecked(False)
            elif checkbox == self.tritanopia_checkbox:
                self.protanopia_checkbox.setChecked(False)
                self.deuteranopia_checkbox.setChecked(False)

        self.update_plot()

    def read_jasco_spectrum(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            if ";" in lines[8]:
                data_lines = lines[8:]
                wavelengths = []
                values = []
                for line in data_lines:
                    try:
                        columns = line.split(";")
                        wavelength = float(columns[0].replace(',', '.'))
                        value = float(columns[4].replace(',', '.'))
                        wavelengths.append(wavelength)
                        values.append(value)
                    except (ValueError, IndexError):
                        continue
                return np.array(wavelengths), np.array(values)

            else:
                data_lines = lines[19:-43]
                wavelengths = []
                values = []
                for line in data_lines:
                    try:
                        wavelength, value = map(lambda x: float(x.replace(',', '.')), line.split())
                        wavelengths.append(wavelength)
                        values.append(value)
                    except ValueError:
                        continue
                return np.array(wavelengths), np.array(values)

        except Exception as e:
            QMessageBox.critical(self, "File Read Error", f"Error reading JASCO spectrum file: {e}")
            return np.array([]), np.array([])

    def extract_protein_name(self, file_path, spectrum_type="absorbance"):
        try:
            data = pd.read_csv(file_path)
            if "wavelength" not in data.columns:
                return "Unknown title/sample"

            value_column = [
                col for col in data.columns
                if (col.endswith("ab") and spectrum_type == "absorbance") or
                   (col.endswith("em") and spectrum_type == "fluorescence")
            ]

            if not value_column:
                return "Unknown title/sample"

            title = value_column[0].replace("ab", "").replace("em", "").strip()
            return title if title else "Unknown title/sample"
        except Exception as e:
            print(f"Error extracting protein name from FPbase file: {e}")
            return "Unknown title/sample"

    def update_protein_name(self, file_path, file_format, protein_name, warning_message=None):
        base_label = f"Loaded file : {file_path}\nSpectrum format : {file_format} ({protein_name})"
        self.label.setText(base_label)

        if not hasattr(self, "warning_label"):
            self.warning_label = QLabel()
            self.warning_label.setStyleSheet("color: red;")
            self.layout().insertWidget(1, self.warning_label)

        if warning_message:
            self.warning_label.setText(warning_message)
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

    def extract_avantes_title(self, file_path):
        try:
            with open(file_path, 'r') as file:
                first_line = file.readline().strip()
                return first_line if first_line else "Unknown title/sample"
        except Exception as e:
            print(f"Error extracting Avantes title: {e}")
            return "Unknown title/sample"

    def extract_jasco_title(self, file_path, file_type):
        try:
            with open(file_path, 'r') as file:
                first_line = file.readline().strip()
                if file_type == "JASCO ASCII" and "TITLE" in first_line:
                    title = first_line.split("TITLE", 1)[1].strip()
                    return title if title else "Unknown title/sample"
                elif file_type == "JASCO CSV" and "TITLE;" in first_line:
                    title = first_line.split("TITLE;", 1)[1].strip()
                    return title if title else "Unknown title/sample"
            return "Unknown title/sample"
        except Exception as e:
            print(f"Error extracting JASCO title: {e}")
            return "Unknown title/sample"

    def load_absorbance_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "",
                                                   "TXT or CSV files (*.txt *.csv)", options=options)
        if file_path:
            self.current_fluorescence = None
            self.available_emission_columns = []
            self.selected_emission_column = None
            self.emission_selector.setVisible(False)
            self.emission_selector.clear()

            file_type = self.detect_file_type(file_path)
            title = "Unknown title/sample"
            if file_type == "FPbase":
                title = self.extract_protein_name(file_path, spectrum_type="absorbance")
            elif file_type == "Avantes":
                title = self.extract_avantes_title(file_path)
            elif file_type in ["JASCO ASCII", "JASCO CSV"]:
                title = self.extract_jasco_title(file_path, file_type)

            self.label.setText(f"Loaded file : {file_path}\nSpectrum format : {file_type} ({title})")

            wavelengths, absorbance = self.read_spectrum(file_path, spectrum_type="absorbance")
            if wavelengths.size > 0 and absorbance.size > 0:
                wavelengths, absorbance = self.process_spectrum(wavelengths, absorbance)
                self.current_wavelengths, self.current_absorbance = wavelengths, absorbance
                self.current_spectrum_type = "absorbance"

                self.plot_spectrum(wavelengths, absorbance, spectrum_type="absorbance")

                self.copy_image_button.setVisible(True)
                self.copy_rgb_button.setVisible(True)
                self.copy_hex_button.setVisible(True)

                self.protanopia_checkbox.setVisible(True)
                self.deuteranopia_checkbox.setVisible(True)
                self.tritanopia_checkbox.setVisible(True)

                self.start_wavelength_label.setVisible(True)
                self.start_wavelength_box.setText(str(wavelengths[0]))
                self.start_wavelength_box.setVisible(True)
                self.end_wavelength_label.setVisible(True)
                self.end_wavelength_box.setText(str(wavelengths[-1]))
                self.end_wavelength_box.setVisible(True)
                self.end_wavelength_suffix.setVisible(True)
                self.export_button.setVisible(True)
            else:
                QMessageBox.critical(self, "Error", "Invalid absorbance spectrum data.")

    def load_fluorescence_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a fluorescence spectrum file", "",
                                                   "TXT or CSV files (*.txt *.csv)", options=options)
        if file_path:

            self.emission_selector.clear()
            self.emission_selector.setVisible(False)
            self.available_emission_columns = []
            self.selected_emission_column = None

            self.current_absorbance = None

            file_type = self.detect_file_type(file_path)
            title = "Unknown title/sample"
            if file_type == "FPbase":
                title = self.extract_protein_name(file_path, spectrum_type="fluorescence")
            elif file_type == "Avantes":
                title = self.extract_avantes_title(file_path)
            elif file_type in ["JASCO ASCII", "JASCO CSV"]:
                title = self.extract_jasco_title(file_path, file_type)

            self.label.setText(f"Loaded fluorescence file : {file_path}\nSpectrum format : {file_type} ({title})")

            wavelengths, fluorescence = self.read_spectrum(file_path, spectrum_type="fluorescence")
            if wavelengths.size > 0 and fluorescence.size > 0:
                wavelengths, fluorescence = self.process_spectrum(wavelengths, fluorescence)
                self.current_wavelengths, self.current_fluorescence = wavelengths, fluorescence
                self.current_spectrum_type = "fluorescence"

                if file_type == "FPbase":
                    value_columns = [
                        col for col in pd.read_csv(file_path).columns if col.endswith("em")
                    ]
                    self.emission_selector.clear()

                    if len(value_columns) >= 2:
                        self.available_emission_columns = value_columns
                        self.emission_selector.addItems(value_columns)
                        self.emission_selector.setVisible(True)
                    else:
                        self.available_emission_columns = []
                        self.selected_emission_column = None
                        self.emission_selector.setVisible(False)

                self.plot_spectrum(wavelengths, fluorescence, spectrum_type="fluorescence")

                if len(self.available_emission_columns) <= 1:
                    self.emission_selector.clear()
                    self.emission_selector.setVisible(False)

                self.copy_image_button.setVisible(True)
                self.copy_rgb_button.setVisible(True)
                self.copy_hex_button.setVisible(True)

                self.protanopia_checkbox.setVisible(True)
                self.deuteranopia_checkbox.setVisible(True)
                self.tritanopia_checkbox.setVisible(True)

                self.start_wavelength_label.setVisible(True)
                self.start_wavelength_box.setText(str(wavelengths[0]))
                self.start_wavelength_box.setVisible(True)
                self.end_wavelength_label.setVisible(True)
                self.end_wavelength_box.setText(str(wavelengths[-1]))
                self.end_wavelength_box.setVisible(True)
                self.end_wavelength_suffix.setVisible(True)
                self.export_button.setVisible(True)
            else:
                QMessageBox.critical(self, "Error", "Invalid fluorescence spectrum data.")

    def update_integration_range(self):
        try:
            start_wavelength = float(self.start_wavelength_box.text())
            end_wavelength = float(self.end_wavelength_box.text())

            mask = (self.current_wavelengths >= start_wavelength) & (self.current_wavelengths <= end_wavelength)
            if self.current_spectrum_type == "absorbance":
                filtered_data = np.zeros_like(self.current_absorbance)
                filtered_data[mask] = self.current_absorbance[mask]
            elif self.current_spectrum_type == "fluorescence":
                filtered_data = np.zeros_like(self.current_fluorescence)
                filtered_data[mask] = self.current_fluorescence[mask]
            else:
                return

            self.plot_spectrum(self.current_wavelengths, filtered_data, spectrum_type=self.current_spectrum_type)

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid wavelength range. Please enter numeric values.")

    def export_spectrum(self):
        try:
            start_wavelength = float(self.start_wavelength_box.text())
            end_wavelength = float(self.end_wavelength_box.text())
            mask = (self.current_wavelengths >= start_wavelength) & (self.current_wavelengths <= end_wavelength)

            filtered_wavelengths = self.current_wavelengths[mask]
            if self.current_spectrum_type == "absorbance":
                filtered_values = self.current_absorbance[mask]
            elif self.current_spectrum_type == "fluorescence":
                filtered_values = self.current_fluorescence[mask]
            else:
                QMessageBox.critical(self, "Error", "No valid spectrum loaded to export.")
                return

            if filtered_wavelengths.size == 0:
                QMessageBox.critical(self, "Error", "No data in the selected wavelength range.")
                return

            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum", "",
                                                       "ASCII Files (*.txt);;All Files (*)", options=options)

            if not file_path:
                return

            with open(file_path, 'w') as f:
                f.write("# Wavelength (nm)\tValue\n")
                for wl, val in zip(filtered_wavelengths, filtered_values):
                    f.write(f"{wl:.2f}\t{val:.6f}\n")

            QMessageBox.information(self, "Success", f"Spectrum exported successfully to:\n{file_path}")

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid wavelength range or spectrum data.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while exporting the spectrum:\n{e}")

    def detect_file_type(self, file_path):
        try:
            with open(file_path, 'r') as file:
                header_lines = [file.readline().strip() for _ in range(10)]

            if any("wavelength" in line.lower() for line in header_lines) and any(
                    "ab" in col or "em" in col for col in header_lines[0].split(",")
            ):
                return "FPbase"

            if any("JASCO" in line for line in header_lines):
                if file_path.lower().endswith('.txt'):
                    return "JASCO ASCII"
                elif file_path.lower().endswith('.csv'):
                    return "JASCO CSV"

            if any("SpectraSuite" in line for line in header_lines):
                return "SpectraSuite"

            if any("Data measured with spectrometer" in line for line in header_lines):
                return "Avantes"

            if any(any(separator in line for separator in [";", "\t", " "]) for line in header_lines):
                return "Generic"

            msg_box = QMessageBox()
            msg_box.setWindowTitle("File Type Selection")
            msg_box.setText("Unknown file format. Please select the file type:")
            button_fpbase = msg_box.addButton("FPbase", QMessageBox.AcceptRole)
            button_jasco = msg_box.addButton("JASCO", QMessageBox.AcceptRole)
            button_generic = msg_box.addButton("Generic", QMessageBox.AcceptRole)
            msg_box.exec_()

            if msg_box.clickedButton() == button_fpbase:
                return "FPbase"
            elif msg_box.clickedButton() == button_jasco:
                return "JASCO"
            else:
                return "Generic"

        except Exception as e:
            print(f"Error detecting file type: {e}")
            return "Unknown"

    def read_spectrum(self, file_path, spectrum_type="absorbance"):
        try:
            file_type = self.detect_file_type(file_path)
            print(f"Detected file type: {file_type}")

            warning_message = None
            protein_name = "Unknown title/sample"

            if file_type == "FPbase":
                data = pd.read_csv(file_path)
                if "wavelength" not in data.columns:
                    raise ValueError("Column 'wavelength' not found in FPbase file.")

                if spectrum_type == "absorbance":

                    value_columns = [col for col in data.columns if col.endswith("ab")]
                    if not value_columns:
                        value_columns = [col for col in data.columns if col.endswith("ex")]
                        warning_message = (
                            "Using excitation spectrum instead of absorbance. "
                            "This may significantly affect the resulting color that should be interpreted cautiously."
                        )
                elif spectrum_type == "fluorescence":
                    value_columns = [col for col in data.columns if col.endswith("em")]
                else:
                    raise ValueError(f"Invalid spectrum type: {spectrum_type}")

                if not value_columns:
                    raise ValueError(
                        f"There is no {'absorbance' if spectrum_type == 'absorbance' else 'emission'} data in this FPbase file."
                    )

                if len(value_columns) > 1:
                    self.available_emission_columns = value_columns
                    self.emission_selector.clear()
                    self.emission_selector.addItems(value_columns)
                    self.emission_selector.setVisible(True)
                else:
                    self.available_emission_columns = []
                    self.selected_emission_column = None
                    self.emission_selector.clear()
                    self.emission_selector.setVisible(False)

                selected_column = (
                    value_columns[0] if not self.selected_emission_column else self.selected_emission_column
                )

                wavelength = data["wavelength"]
                values = data[selected_column].fillna(0)

                protein_name = selected_column.rsplit(" ", 1)[0]
                self.update_protein_name(file_path, file_type, protein_name, warning_message)

                return wavelength.values, values.values

            elif file_type == "Avantes":
                title = self.extract_avantes_title(file_path)
                protein_name = title
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                data_lines = lines[8:]
                wavelengths, values = [], []
                for line in data_lines:
                    try:
                        columns = line.split(";")
                        wavelength = float(columns[0].replace(',', '.'))
                        value = float(columns[-1].replace(',', '.'))
                        wavelengths.append(wavelength)
                        values.append(value)
                    except (ValueError, IndexError):
                        continue

                self.update_protein_name(file_path, file_type, protein_name)
                return np.array(wavelengths), np.array(values)

            elif file_type == "JASCO ASCII":
                title = self.extract_jasco_title(file_path, file_type)
                protein_name = title
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                data_lines = lines[19:-43]
                wavelengths, values = [], []
                for line in data_lines:
                    try:
                        wavelength, value = map(lambda x: float(x.replace(',', '.')), line.split())
                        wavelengths.append(wavelength)
                        values.append(value)
                    except ValueError:
                        continue

                self.update_protein_name(file_path, file_type, protein_name)
                return np.array(wavelengths), np.array(values)

            elif file_type == "JASCO CSV":
                title = self.extract_jasco_title(file_path, file_type)
                protein_name = title
                df = pd.read_csv(file_path, delimiter=';', skiprows=19, skipfooter=45, engine='python')
                if df.shape[1] < 2:
                    raise ValueError("JASCO CSV file does not have the expected structure.")
                df.columns = ['Wavelength (nm)', 'Value']
                df['Wavelength (nm)'] = df['Wavelength (nm)'].str.replace(',', '.').astype(float)
                df['Value'] = df['Value'].str.replace(',', '.').astype(float)

                self.update_protein_name(file_path, file_type, protein_name)
                return df['Wavelength (nm)'].values, df['Value'].values

            elif file_type == "SpectraSuite":

                protein_name = "Unknown title/sample"
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith("User:"):
                            protein_name = f"User: {line.split('User:')[1].strip()}"
                            break

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                header_lines = 14
                wavelengths, values = [], []
                for line in lines[header_lines:]:
                    try:
                        columns = line.split()
                        wavelength = float(columns[0])
                        value = float(columns[1])
                        wavelengths.append(wavelength)
                        values.append(value)
                    except (ValueError, IndexError):
                        continue

                self.update_protein_name(file_path, file_type, protein_name)
                return np.array(wavelengths), np.array(values)

            elif file_type == "Generic":
                title = "Generic"
                protein_name = title
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                wavelengths, values = [], []
                for line in lines:
                    try:
                        if ";" in line:
                            wavelength, value = map(lambda x: float(x.replace(',', '.')), line.split(";"))
                        elif "\t" in line:
                            wavelength, value = map(lambda x: float(x.replace(',', '.')), line.split("\t"))
                        else:
                            wavelength, value = map(lambda x: float(x.replace(',', '.')), line.split())
                        wavelengths.append(wavelength)
                        values.append(value)
                    except ValueError:
                        continue

                self.update_protein_name(file_path, file_type, protein_name)
                return np.array(wavelengths), np.array(values)

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            QMessageBox.critical(self, "Error reading spectrum file", f"Error: {e}")
            return np.array([]), np.array([])

    def change_emission_spectrum(self, index):
        try:
            if not self.available_emission_columns or index < 0 or index >= len(self.available_emission_columns):
                return

            self.selected_emission_column = self.available_emission_columns[index]

            file_path = self.label.text().split(": ")[1].split("\n")[0]
            data = pd.read_csv(file_path)

            if self.selected_emission_column not in data.columns:
                QMessageBox.critical(self, "Error", f"Column '{self.selected_emission_column}' not found in file.")
                return

            wavelengths = data["wavelength"].values
            values = data[self.selected_emission_column].fillna(0).values

            if self.selected_emission_column.endswith("ex") or self.selected_emission_column.endswith("ab"):
                self.current_spectrum_type = "absorbance"
            elif self.selected_emission_column.endswith("em"):
                self.current_spectrum_type = "fluorescence"
            else:
                raise ValueError("Unsupported spectrum type. Use 'absorbance' or 'fluorescence'.")

            protein_name = self.selected_emission_column.rsplit(" ", 1)[0]

            warning_message = ""
            if self.selected_emission_column.endswith("ex"):
                warning_message = (
                    "No absorbance data found, using excitation instead. "
                    "This may significantly affect the resulting color that should be considered cautiously."
                )

            self.update_protein_name(file_path, "FPbase", protein_name, warning_message)

            wavelengths, values = self.process_spectrum(wavelengths, values)

            if self.current_spectrum_type == "absorbance":
                self.current_wavelengths = wavelengths
                self.current_absorbance = values
            elif self.current_spectrum_type == "fluorescence":
                self.current_wavelengths = wavelengths
                self.current_fluorescence = values

            self.plot_spectrum(wavelengths, values, spectrum_type=self.current_spectrum_type)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while changing the emission spectrum:\n{e}")

    def plot_fpbase_spectrum(self, wavelengths, absorbance):
        self.canvas.figure.clf()

        ax = self.canvas.figure.add_subplot(111)
        ax.plot(wavelengths, absorbance, label="Absorbance", color='blue')
        ax.set_title("Absorbance Spectrum (FPbase)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()

    def process_spectrum(self, wavelengths, absorbance):

        current_step = np.diff(wavelengths).mean()
        target_step = 5

        if current_step > target_step:

            interpolation_function = interp1d(wavelengths, absorbance, kind='linear', fill_value="extrapolate")
            new_wavelengths = np.arange(wavelengths.min(), wavelengths.max() + target_step, target_step)
            absorbance = interpolation_function(new_wavelengths)
            wavelengths = new_wavelengths

        sorted_indices = np.argsort(wavelengths)[::-1]
        wavelengths = wavelengths[sorted_indices]
        absorbance = absorbance[sorted_indices]

        mask_380_750 = (wavelengths >= 400) & (wavelengths <= 750)
        absorbance_380_750 = absorbance[mask_380_750]

        if absorbance_380_750.size > 0:
            absorbance_380_750 -= np.min(absorbance_380_750)
            max_absorbance = np.max(absorbance_380_750)
            if max_absorbance > 0:
                absorbance_380_750 /= max_absorbance

        normalized_absorbance = absorbance.copy()
        normalized_absorbance[mask_380_750] = absorbance_380_750

        extended_wavelengths = np.arange(250, 751, target_step)
        extended_absorbance = np.zeros_like(extended_wavelengths, dtype=float)
        for wl, ab in zip(wavelengths, normalized_absorbance):
            if 250 <= wl <= 750:
                extended_absorbance[int((wl - 250) / target_step)] = ab

        bin_edges = np.arange(380, 751, target_step)
        binned_absorbance = []
        for i in range(len(bin_edges) - 1):
            start, end = bin_edges[i], bin_edges[i + 1]
            mask = (extended_wavelengths >= start) & (extended_wavelengths < end)
            binned_absorbance.append(np.mean(extended_absorbance[mask]))

        return bin_edges[:-1], np.array(binned_absorbance)

    def load_cone_sensitivity(self, file_path, wavelengths):
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                print(f"Error: The file {file_path} is empty.")
                return np.array([])

            original_wavelengths = data.iloc[:, 0].values
            sensitivity = data.iloc[:, 1].values

            if original_wavelengths.size == 0 or sensitivity.size == 0:
                print(f"Error: The file {file_path} does not contain valid data.")
                return np.array([])

            interpolated_sensitivity = np.interp(wavelengths, original_wavelengths, sensitivity)

            return interpolated_sensitivity

        except Exception as e:
            print(f"Error loading cone sensitivity files: {e}")
            return np.array([])

    def plot_spectrum(self, wavelengths, data, spectrum_type="absorbance"):
        self.canvas.figure.clf()
        gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[1, 1])

        filter_mask = (wavelengths >= 380) & (wavelengths <= 750)
        filtered_wavelengths = wavelengths[filter_mask]
        filtered_data = data[filter_mask]

        if len(filtered_wavelengths) == 0 or len(filtered_data) == 0:
            QMessageBox.critical(self, "Error", "No valid data in the visible range (380-750 nm).")
            return

        normalized_data = (filtered_data - np.min(filtered_data)) / (np.max(filtered_data) - np.min(filtered_data))

        if spectrum_type == "absorbance":
            title_left = "Absorption and Transmission"
            bar_data = 1 - filtered_data
            bar_data[bar_data < 0] = 0
        elif spectrum_type == "fluorescence":
            title_left = "Emission"
            bar_data = filtered_data
        else:
            raise ValueError("Unsupported spectrum type. Use 'absorbance' or 'fluorescence'.")

        colors = [ColorUtils.wavelength_to_rgb(wl + 2.5) for wl in filtered_wavelengths]

        bin_edges = np.arange(380, 751, 5)
        binned_bar_data = []
        for i in range(len(bin_edges) - 1):
            start, end = bin_edges[i], bin_edges[i + 1]
            mask = (filtered_wavelengths >= start) & (filtered_wavelengths < end)
            mean_value = np.mean(bar_data[mask]) if mask.any() else 0
            binned_bar_data.append(mean_value)

        max_binned_value = np.max(binned_bar_data)
        if max_binned_value > 0:
            binned_bar_data = [value / max_binned_value for value in binned_bar_data]

        ax_main = self.canvas.figure.add_subplot(gs[0, 0])
        ax_main.bar(bin_edges[:-1], binned_bar_data, width=5, align="center", color=colors, edgecolor="black",
                    alpha=0.6)
        ax_main.plot(filtered_wavelengths, normalized_data, color="blue", linestyle="-", linewidth=1.5)

        ax_main.set_xlim(filtered_wavelengths.min(), filtered_wavelengths.max())
        ax_main.set_ylim(0, 1)

        ax_main.set_title(title_left)
        ax_main.set_xlabel("Wavelength (nm)")
        ax_main.set_ylabel("Normalized values")

        red_sensitivity = self.load_cone_sensitivity("Red.csv", filtered_wavelengths)
        green_sensitivity = self.load_cone_sensitivity("Green.csv", filtered_wavelengths)
        blue_sensitivity = self.load_cone_sensitivity("Blue.csv", filtered_wavelengths)

        if red_sensitivity.size == 0 or green_sensitivity.size == 0 or blue_sensitivity.size == 0:
            QMessageBox.critical(self, "Error", "Cone sensitivity data could not be loaded correctly.")
            return

        red_contribution = np.array(binned_bar_data) * red_sensitivity
        green_contribution = np.array(binned_bar_data) * green_sensitivity
        blue_contribution = np.array(binned_bar_data) * blue_sensitivity

        if spectrum_type == "absorbance":
            red_weight, green_weight, blue_weight = 1.0, 1.0, 1.4
            resulting_color = (
                np.clip(np.sum(red_contribution) * red_weight / np.sum(red_sensitivity), 0, 1),
                np.clip(np.sum(green_contribution) * green_weight / np.sum(green_sensitivity), 0, 1),
                np.clip(np.sum(blue_contribution) * blue_weight / np.sum(blue_sensitivity), 0, 1),
            )
            resulting_color = tuple(min(c * self.absorbance_brightness, 1.0) for c in resulting_color)

        elif spectrum_type == "fluorescence":
            red_weight, green_weight, blue_weight = 1.3, 0.9, 1.0
            resulting_color = (
                np.clip(np.sum(red_contribution) * red_weight / np.sum(red_sensitivity), 0, 1),
                np.clip(np.sum(green_contribution) * green_weight / np.sum(green_sensitivity), 0, 1),
                np.clip(np.sum(blue_contribution) * blue_weight / np.sum(blue_sensitivity), 0, 1),
            )
            resulting_color = tuple(min(c * self.fluorescence_brightness, 1.0) for c in resulting_color)

            import colorsys
            r, g, b = resulting_color
            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            saturation_boost = 1.5
            luminance_boost = 2.0

            s = np.clip(s * saturation_boost, 0, 1)
            v = np.clip(v * luminance_boost, 0, 1)

            resulting_color = colorsys.hsv_to_rgb(h, s, v)

        self.resulting_color = resulting_color

        original_color = [c * 255 for c in resulting_color]

        ax_cones = self.canvas.figure.add_subplot(gs[0, 1])

        transformed_color = original_color
        if self.protanopia_checkbox.isChecked():
            transformed_color = DaltonismSimulator.simulate_color(original_color, "protanopia")
        elif self.deuteranopia_checkbox.isChecked():
            transformed_color = DaltonismSimulator.simulate_color(original_color, "deuteranopia")
        elif self.tritanopia_checkbox.isChecked():
            transformed_color = DaltonismSimulator.simulate_color(original_color, "tritanopia")
        transformed_color = [c / 255 for c in transformed_color]

        ax_cones.set_facecolor(transformed_color)

        ax_cones.bar(filtered_wavelengths, red_contribution, width=5, color="red", edgecolor="black", alpha=0.5)
        ax_cones.bar(filtered_wavelengths, green_contribution, width=5, color="green", edgecolor="black", alpha=0.5)
        ax_cones.bar(filtered_wavelengths, blue_contribution, width=5, color="blue", edgecolor="black", alpha=0.5)

        ax_cones.plot(filtered_wavelengths, red_sensitivity / max(red_sensitivity), color="darkred", linestyle="--")
        ax_cones.plot(filtered_wavelengths, green_sensitivity / max(green_sensitivity), color="darkgreen",
                      linestyle="--")
        ax_cones.plot(filtered_wavelengths, blue_sensitivity / max(blue_sensitivity), color="darkblue", linestyle="--")

        ax_cones.set_title("Color perception")
        ax_cones.set_xlim(filtered_wavelengths.min(), filtered_wavelengths.max())
        ax_cones.set_ylim(0, 1)
        ax_cones.set_xlabel("Wavelength (nm)")
        ax_cones.set_ylabel("Normalized contribution")

        rgb_255 = tuple(int(c * 255) for c in transformed_color)
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_255)
        ax_cones.text(
            0.9, 0.6, f"RGB: {rgb_255}\nHex: {hex_color}",
            ha="center", va="center", rotation=90,
            transform=ax_cones.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
        )

        plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15, hspace=0.5, wspace=0.4)

        self.canvas.draw()

    def copy_image(self):
        if self.resulting_color:
            width, height = 100, 100
            image = QImage(width, height, QImage.Format_RGB32)
            painter = QPainter(image)
            color = QColor(int(self.resulting_color[0] * 255), int(self.resulting_color[1] * 255),
                           int(self.resulting_color[2] * 255))
            painter.fillRect(0, 0, width, height, color)
            painter.end()

            clipboard = QApplication.clipboard()
            clipboard.setImage(image)
            QMessageBox.information(self, "Success", "Color image copied to clipboard.")
        else:
            QMessageBox.warning(self, "Error", "No color to copy. Please load a spectrum first.")

    def copy_rgb_code(self):
        if self.resulting_color:
            rgb_255 = tuple(int(c * 255) for c in self.resulting_color)
            rgb_text = f"{rgb_255[0]},{rgb_255[1]},{rgb_255[2]}"
            QApplication.clipboard().setText(rgb_text)
            QMessageBox.information(self, "Success", f"RGB code {rgb_text} copied to clipboard.")
        else:
            QMessageBox.warning(self, "Error", "No color to copy. Please load a spectrum first.")

    def copy_hex_code(self):
        if self.resulting_color:
            rgb_255 = tuple(int(c * 255) for c in self.resulting_color)
            hex_code = "#{:02x}{:02x}{:02x}".format(*rgb_255)
            QApplication.clipboard().setText(hex_code)
            QMessageBox.information(self, "Success", f"Hex code {hex_code} copied to clipboard.")
        else:
            QMessageBox.warning(self, "Error", "No color to copy. Please load a spectrum first.")

    def update_plot(self):
        try:
            start_wavelength = float(self.start_wavelength_box.text())
            end_wavelength = float(self.end_wavelength_box.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid wavelength range. Please enter numeric values.")
            return

        mask = (self.current_wavelengths >= start_wavelength) & (self.current_wavelengths <= end_wavelength)

        if self.current_spectrum_type == "absorbance" and self.current_absorbance is not None:
            filtered_data = np.zeros_like(self.current_absorbance)
            filtered_data[mask] = self.current_absorbance[mask]
            self.plot_spectrum(self.current_wavelengths, filtered_data, spectrum_type="absorbance")
        elif self.current_spectrum_type == "fluorescence" and self.current_fluorescence is not None:
            filtered_data = np.zeros_like(self.current_fluorescence)
            filtered_data[mask] = self.current_fluorescence[mask]
            self.plot_spectrum(self.current_wavelengths, filtered_data, spectrum_type="fluorescence")

    def update_cone_checkboxes_visibility(self):
        visible = self.current_spectrum_type in ["absorbance", "fluorescence"]
        self.blue_contribution_checkbox.setVisible(visible)
        self.green_contribution_checkbox.setVisible(visible)
        self.red_contribution_checkbox.setVisible(visible)

    def set_absorbance_brightness(self, brightness):
        self.absorbance_brightness = brightness
        if self.current_spectrum_type == "absorbance":
            self.plot_spectrum(self.current_wavelengths, self.current_absorbance, spectrum_type="absorbance")

    def set_fluorescence_brightness(self, brightness):
        self.fluorescence_brightness = brightness
        if self.current_spectrum_type == "fluorescence":
            self.plot_spectrum(self.current_wavelengths, self.current_fluorescence, spectrum_type="fluorescence")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectraAnalyzer()
    window.show()
    sys.exit(app.exec_())