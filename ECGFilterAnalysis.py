import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, firwin, filtfilt, iirnotch
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
import os

# Use default font to avoid font issues and set a consistent style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid') # Optional: A nice plotting style

class ECGFilterAnalysis:
    def __init__(self, fs=360):
        self.data = None
        self.fs = fs
        self.time = None
        self.max_time = 0
        self.current_lead = None # Store the currently selected lead

    def load_data(self, csv_path):
        """Loads data from a CSV file and identifies the ECG column."""
        try:
            self.data = pd.read_csv(csv_path)
            # Try to strip quotes from column names, common with some CSV exports
            self.data.columns = self.data.columns.str.strip("'\" ").str.replace('\ufeff', '')

            # Attempt to find a suitable numeric column for ECG
            numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in the CSV file.")

            # Prioritize 'ECG', 'MLII', or the first numeric column
            if 'ECG' in numeric_cols:
                self.current_lead = 'ECG'
            elif 'MLII' in numeric_cols:
                self.current_lead = 'MLII'
            else:
                self.current_lead = numeric_cols[0] # Default to the first numeric column

            self.time = np.arange(len(self.data[self.current_lead])) / self.fs
            self.max_time = self.time[-1] if len(self.time) > 0 else 0
            print(f"Loaded {len(self.data)} samples. Identified lead: {self.current_lead}")
            return True
        except Exception as e:
            print(f"Error loading file or identifying lead: {e}")
            self.data = None
            self.time = None
            self.max_time = 0
            self.current_lead = None
            return False

    def get_numeric_columns(self):
        """Returns a list of numeric column names from the loaded data."""
        if self.data is not None:
            return self.data.select_dtypes(include=np.number).columns.tolist()
        return []

    def design_fir_filter(self, cutoff, numtaps, window='hamming', pass_zero=False):
        """Designs a FIR filter and returns its coefficients."""
        nyquist = self.fs / 2
        cutoff_norm = cutoff / nyquist
        if numtaps % 2 == 0: # Ensure odd taps for linear phase
            numtaps += 1
        return firwin(numtaps, cutoff_norm, window=window, pass_zero=pass_zero)

    def design_iir_filter(self, cutoff_low, cutoff_high, order, ftype='butter', btype='bandpass', rp=1, rs=20):
        """Designs an IIR filter (Butterworth, Chebyshev Type I/II, Elliptic) and returns coefficients (b, a)."""
        nyquist = self.fs / 2
        Wn = [cutoff_low / nyquist, cutoff_high / nyquist] if btype in ['bandpass', 'bandstop'] else cutoff_low / nyquist

        if ftype == 'butter':
            b, a = butter(order, Wn, btype=btype, analog=False)
        elif ftype == 'cheby1':
            b, a = signal.cheby1(order, rp, Wn, btype=btype, analog=False)
        elif ftype == 'cheby2':
            b, a = signal.cheby2(order, rs, Wn, btype=btype, analog=False)
        elif ftype == 'ellip':
            b, a = signal.ellip(order, rp, rs, Wn, btype=btype, analog=False)
        else:
            raise ValueError(f"Unsupported IIR filter type: {ftype}")
        return b, a

    def design_notch_filter(self, freq, Q):
        """Designs a notch filter for a specific frequency and Q factor."""
        nyquist = self.fs / 2
        w0 = freq / nyquist
        b, a = iirnotch(w0, Q)
        return b, a

    def apply_filter(self, signal_data, filter_type, params):
        """Applies a specified filter to the signal data."""
        if filter_type == 'none':
            return signal_data
        elif filter_type == 'fir_lowpass':
            coeffs = self.design_fir_filter(
                cutoff=params.get('cutoff', 40),
                numtaps=params.get('numtaps', 101),
                window=params.get('window', 'hamming'),
                pass_zero=True # For lowpass
            )
            return filtfilt(coeffs, 1, signal_data)
        elif filter_type == 'fir_bandpass': # Example for FIR bandpass
            coeffs = self.design_fir_filter(
                cutoff=[params.get('lowcut', 0.5), params.get('highcut', 50)],
                numtaps=params.get('numtaps', 101),
                window=params.get('window', 'hamming'),
                pass_zero=False # For bandpass
            )
            return filtfilt(coeffs, 1, signal_data)
        elif filter_type in ['butter', 'cheby1', 'cheby2', 'ellip']:
            b, a = self.design_iir_filter(
                cutoff_low=params.get('lowcut', 0.5),
                cutoff_high=params.get('highcut', 50),
                order=params.get('order', 5),
                ftype=filter_type,
                btype=params.get('btype', 'bandpass'), # e.g., 'lowpass', 'highpass', 'bandpass'
                rp=params.get('rp', 1), # For Chebyshev Type I, Elliptic
                rs=params.get('rs', 20) # For Chebyshev Type II, Elliptic
            )
            return filtfilt(b, a, signal_data)
        elif filter_type == 'notch':
            b, a = self.design_notch_filter(
                freq=params.get('freq', 50),
                Q=params.get('Q', 30)
            )
            return filtfilt(b, a, signal_data)
        else:
            raise ValueError(f"Unknown filter type for application: {filter_type}")

    def _get_signal_data(self, lead=None, start_sample=0, end_sample=None):
        """Helper to extract signal data and corresponding time range."""
        if self.data is None:
            raise ValueError("No data loaded.")

        if lead is None: # Use the default lead
            lead = self.current_lead
        if lead not in self.data.columns:
            raise ValueError(f"Lead '{lead}' not found in data. Available: {self.get_numeric_columns()}")

        signal_data = self.data[lead].values
        if end_sample is None:
            end_sample = len(signal_data)
        elif end_sample > len(signal_data):
            end_sample = len(signal_data)

        # Ensure start and end are within bounds and start <= end
        start_sample = max(0, start_sample)
        end_sample = min(len(signal_data), end_sample)
        if start_sample >= end_sample:
            # If invalid range, return a default small segment or entire signal
            print(f"Warning: Invalid time range [{start_sample}:{end_sample}], plotting full signal or a default segment.")
            start_sample = 0
            end_sample = min(len(signal_data), 1000) # Plot first few seconds by default

        segment_data = signal_data[start_sample:end_sample]
        segment_time = np.arange(len(segment_data)) / self.fs + (start_sample / self.fs)
        return segment_data, segment_time

    def plot_window_functions(self, windows=['hamming', 'blackman', 'hann', 'kaiser'], numtaps=101, output_filename='window_functions.png'):
        """Plots various window functions."""
        fig, ax = plt.subplots(figsize=(12, 8))
        for window in windows:
            if window == 'kaiser':
                # Kaiser window requires beta parameter. Using a default value (e.g., 14)
                win = signal.get_window((window, 14), numtaps)
                ax.plot(win, label=f'{window.capitalize()} (beta=14)')
            else:
                win = signal.get_window(window, numtaps)
                ax.plot(win, label=window.capitalize())
        ax.set_title(f"Window Functions (NumTaps={numtaps})")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        return output_filename

    def plot_frequency_response(self, filter_design_params, output_filename='frequency_response.png'):
        """Plots frequency response of specified FIR and IIR filters."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # FIR Filter
        fir_params = filter_design_params.get('fir', {})
        if fir_params:
            fir_coeffs = self.design_fir_filter(
                cutoff=fir_params.get('cutoff', 40),
                numtaps=fir_params.get('numtaps', 101),
                window=fir_params.get('window', 'hamming'),
                pass_zero=fir_params.get('pass_zero', True) # Assume lowpass by default
            )
            w_fir, h_fir = signal.freqz(fir_coeffs, 1, worN=8000)
            ax.plot(0.5 * self.fs * w_fir / np.pi, 20 * np.log10(np.abs(h_fir)), 'b-', label='FIR Filter')

        # IIR Filter
        iir_params = filter_design_params.get('iir', {})
        if iir_params:
            b_iir, a_iir = self.design_iir_filter(
                cutoff_low=iir_params.get('lowcut', 0.5),
                cutoff_high=iir_params.get('highcut', 50),
                order=iir_params.get('order', 5),
                ftype=iir_params.get('ftype', 'butter'),
                btype=iir_params.get('btype', 'bandpass'),
                rp=iir_params.get('rp', 1),
                rs=iir_params.get('rs', 20)
            )
            w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=8000)
            ax.plot(0.5 * self.fs * w_iir / np.pi, 20 * np.log10(np.abs(h_iir)), 'r-', label=f'IIR {iir_params.get("ftype", "").capitalize()} Filter')

        # Notch Filter
        notch_params = filter_design_params.get('notch', {})
        if notch_params:
            b_notch, a_notch = self.design_notch_filter(
                freq=notch_params.get('freq', 50),
                Q=notch_params.get('Q', 30)
            )
            w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=8000)
            ax.plot(0.5 * self.fs * w_notch / np.pi, 20 * np.log10(np.abs(h_notch)), 'g--', label='Notch Filter')

        ax.set_title('Frequency Response of Filters')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, self.fs / 2)
        ax.set_ylim(-60, 5) # Typical range for magnitude plots
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        return output_filename

    def plot_ecg_signals(self, lead_name, signals_dict, title, time_range_sec=None, scale=1.0, output_filename='ecg_plot.png'):
        """
        Plots multiple ECG signals (original and filtered) on the same axes.
        signals_dict: { 'label': signal_array, ... }
        time_range_sec: [start_sec, end_sec] or None for full signal.
        """
        if self.data is None:
            raise ValueError("No data loaded for plotting.")
        if lead_name not in self.data.columns:
            raise ValueError(f"Lead '{lead_name}' not found.")

        # Determine plotting range in samples
        start_sample = 0
        end_sample = len(self.data[lead_name])
        if time_range_sec is not None and len(time_range_sec) == 2:
            start_sample = int(time_range_sec[0] * self.fs)
            end_sample = int(time_range_sec[1] * self.fs)

        plot_data_time, _ = self._get_signal_data(lead=lead_name, start_sample=start_sample, end_sample=end_sample)
        plot_time = np.arange(len(plot_data_time)) / self.fs + (start_sample / self.fs) # Recalculate time for the segment

        fig, ax = plt.subplots(figsize=(14, 7))

        for label, signal_array in signals_dict.items():
            segment_signal = signal_array[start_sample:end_sample] * scale
            ax.plot(plot_time, segment_signal, label=label)

        ax.set_title(f"{title} ({lead_name})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        return output_filename

    def plot_subplot_signals(self, lead_name, signals_dict, titles_dict, time_range_sec=None, scale=1.0, output_filename='ecg_subplots.png'):
        """
        Plots multiple ECG signals in separate subplots.
        signals_dict: { 'label': signal_array, ... }
        titles_dict: { 'label': 'Subtitle for plot', ... }
        time_range_sec: [start_sec, end_sec] or None for full signal.
        """
        if self.data is None:
            raise ValueError("No data loaded for plotting.")
        if lead_name not in self.data.columns:
            raise ValueError(f"Lead '{lead_name}' not found.")

        num_plots = len(signals_dict)
        if num_plots == 0:
            return None

        # Determine plotting range in samples
        start_sample = 0
        end_sample = len(self.data[lead_name])
        if time_range_sec is not None and len(time_range_sec) == 2:
            start_sample = int(time_range_sec[0] * self.fs)
            end_sample = int(time_range_sec[1] * self.fs)

        # Get data for time axis
        plot_data_time, _ = self._get_signal_data(lead=lead_name, start_sample=start_sample, end_sample=end_sample)
        plot_time = np.arange(len(plot_data_time)) / self.fs + (start_sample / self.fs)

        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes] # Make it iterable if only one subplot

        for i, (label, signal_array) in enumerate(signals_dict.items()):
            segment_signal = signal_array[start_sample:end_sample] * scale
            axes[i].plot(plot_time, segment_signal, label=label)
            axes[i].set_title(titles_dict.get(label, label))
            axes[i].set_ylabel("Amplitude")
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        return output_filename

    def export_to_pdf(self, image_paths, output_pdf_path='report.pdf'):
        """Combines multiple image files into a single PDF."""
        try:
            with PdfPages(output_pdf_path) as pdf:
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        fig = plt.figure(figsize=(11, 8.5)) # A4 size equivalent
                        img = plt.imread(img_path)
                        plt.imshow(img)
                        plt.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                    else:
                        print(f"Warning: Image file not found: {img_path}")
            print(f"PDF report generated: {output_pdf_path}")
            return output_pdf_path
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None