from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, firwin, filtfilt
import warnings
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile
import json
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Restrict origins in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ECGFilterAnalysis:
    def __init__(self, csv_path, fs=360):
        self.csv_path = csv_path
        self.data = None
        self.fs = fs
        self.fir_coeffs = None
        self.iir_b = None
        self.iir_a = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            self.data.columns = self.data.columns.str.strip("'\"")
            print(f"Loaded {len(self.data)} samples, columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def design_fir_filter(self, cutoff_low=0.5, cutoff_high=40, numtaps=101, window='hamming'):
        nyquist = self.fs / 2
        low = cutoff_low / nyquist
        high = cutoff_high / nyquist
        self.fir_coeffs = firwin(numtaps, [low, high], pass_zero=False, window=window)
        return self.fir_coeffs
    
    def design_iir_filter(self, cutoff_low=0.5, cutoff_high=40, order=4, ftype='butter'):
        nyquist = self.fs / 2
        low = cutoff_low / nyquist
        high = cutoff_high / nyquist
        if ftype == 'butter':
            self.iir_b, self.iir_a = butter(order, [low, high], btype='band')
        elif ftype == 'cheby1':
            self.iir_b, self.iir_a = signal.cheby1(order, 1, [low, high], btype='band')
        elif ftype == 'cheby2':
            self.iir_b, self.iir_a = signal.cheby2(order, 20, [low, high], btype='band')
        elif ftype == 'ellip':
            self.iir_b, self.iir_a = signal.ellip(order, 1, 20, [low, high], btype='band')
        return self.iir_b, self.iir_a
    
    def apply_filters(self, signal_data):
        if self.fir_coeffs is None:
            self.design_fir_filter()
        if self.iir_b is None or self.iir_a is None:
            self.design_iir_filter()
            
        fir_filtered = lfilter(self.fir_coeffs, 1, signal_data)
        iir_filtered = filtfilt(self.iir_b, self.iir_a, signal_data)
        return fir_filtered, iir_filtered
    
    def plot_frequency_response(self, fir_window='hamming', iir_type='butter'):
        if self.fir_coeffs is None:
            self.design_fir_filter(window=fir_window)
        if self.iir_b is None or self.iir_a is None:
            self.design_iir_filter(ftype=iir_type)
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        w_fir, h_fir = signal.freqz(self.fir_coeffs, 1, worN=8000)
        ax1.plot(0.5 * self.fs * w_fir / np.pi, np.abs(h_fir), 'b-', linewidth=2)
        ax1.set_title(f'Frequency Response - FIR ({fir_window})')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.grid(True)
        ax1.set_xlim(0, 50)
        
        w_iir, h_iir = signal.freqz(self.iir_b, self.iir_a, worN=8000)
        ax2.plot(0.5 * self.fs * w_iir / np.pi, np.abs(h_iir), 'r-', linewidth=2)
        ax2.set_title(f'Frequency Response - IIR ({iir_type})')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True)
        ax2.set_xlim(0, 50)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode()
    
    def analyze_and_plot(self, lead='MLII', start=0, end=2000, fir_window='hamming', iir_type='butter'):
        if self.data is None:
            print("No data loaded.")
            return None
            
        available_columns = list(self.data.columns)
        target_column = None
        for col in available_columns:
            if lead.lower() in col.lower():
                target_column = col
                break
        if target_column is None:
            for col in available_columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    target_column = col
                    break
        if target_column is None:
            print("No suitable data column found!")
            return None
            
        signal_data = self.data[target_column].values
        if end > len(signal_data): 
            end = len(signal_data)
        signal_data = signal_data[start:end]
        time = np.arange(len(signal_data)) / self.fs
        
        self.design_fir_filter(window=fir_window)
        self.design_iir_filter(ftype=iir_type)
        
        fir_filtered, iir_filtered = self.apply_filters(signal_data)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        axes[0].plot(time, signal_data, 'k-', linewidth=1.5, label='Original Signal')
        axes[0].set_title(f'Original ECG Signal - {target_column}')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(time, fir_filtered, 'b-', linewidth=1.5, label=f'FIR ({fir_window})')
        axes[1].set_title(f'Signal after FIR ({fir_window})')
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(time, iir_filtered, 'r-', linewidth=1.5, label=f'IIR ({iir_type})')
        axes[2].set_title(f'Signal after IIR ({iir_type})')
        axes[2].set_ylabel('Amplitude (mV)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        axes[3].plot(time, signal_data, 'k-', linewidth=1, alpha=0.7, label='Original')
        axes[3].plot(time, fir_filtered, 'b-', linewidth=1.5, label=f'FIR ({fir_window})')
        axes[3].plot(time, iir_filtered, 'r-', linewidth=1.5, label=f'IIR ({iir_type})')
        axes[3].set_title('Filter Comparison')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Amplitude (mV)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode()
    
    def spectral_analysis(self, lead='MLII', start=0, end=2000, fir_window='hamming', iir_type='butter'):
        if self.data is None:
            print("No data loaded.")
            return None
            
        available_columns = list(self.data.columns)
        target_column = None
        for col in available_columns:
            if lead.lower() in col.lower():
                target_column = col
                break
        if target_column is None:
            for col in available_columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    target_column = col
                    break
        if target_column is None:
            print("No suitable data column found!")
            return None
            
        signal_data = self.data[target_column].values
        if end > len(signal_data): 
            end = len(signal_data)
        signal_data = signal_data[start:end]
        
        self.design_fir_filter(window=fir_window)
        self.design_iir_filter(ftype=iir_type)
        
        fir_filtered, iir_filtered = self.apply_filters(signal_data)
        
        freqs = np.fft.fftfreq(len(signal_data), 1/self.fs)
        freqs = freqs[:len(freqs)//2]
        fft_original = np.abs(np.fft.fft(signal_data))[:len(freqs)]
        fft_fir = np.abs(np.fft.fft(fir_filtered))[:len(freqs)]
        fft_iir = np.abs(np.fft.fft(iir_filtered))[:len(freqs)]
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(freqs, fft_original, 'k-', linewidth=1.5)
        plt.title('Frequency Spectrum - Original Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 50)
        
        plt.subplot(2, 2, 2)
        plt.plot(freqs, fft_fir, 'b-', linewidth=1.5)
        plt.title(f'Frequency Spectrum - FIR ({fir_window})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 50)
        
        plt.subplot(2, 2, 3)
        plt.plot(freqs, fft_iir, 'r-', linewidth=1.5)
        plt.title(f'Frequency Spectrum - IIR ({iir_type})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 50)
        
        plt.subplot(2, 2, 4)
        plt.plot(freqs, fft_original, 'k-', alpha=0.7, label='Original')
        plt.plot(freqs, fft_fir, 'b-', label=f'FIR ({fir_window})')
        plt.plot(freqs, fft_iir, 'r-', label=f'IIR ({iir_type})')
        plt.title('Frequency Spectrum Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, 50)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode()

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        data = request.form.to_dict()
        lead = data.get('lead', 'MLII')
        start = int(data.get('start', 0))
        end = int(data.get('end', 2000))
        fir_window = data.get('fir_window', 'hamming')
        iir_type = data.get('iir_type', 'butter')
        fs = int(data.get('fs', 360))
        
        analyzer = ECGFilterAnalysis(filepath, fs=fs)
        if not analyzer.load_data():
            return jsonify({'error': 'Error loading data'}), 400
            
        freq_response = analyzer.plot_frequency_response(fir_window, iir_type)
        time_domain = analyzer.analyze_and_plot(lead, start, end, fir_window, iir_type)
        freq_domain = analyzer.spectral_analysis(lead, start, end, fir_window, iir_type)
        
        return jsonify({
            'freq_response': freq_response,
            'time_domain': time_domain,
            'freq_domain': freq_domain,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        params = json.loads(request.form['params'])
        freq_response = request.form['freq_response']
        time_domain = request.form['time_domain']
        freq_domain = request.form['freq_domain']
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        elements = []
        
        elements.append(Paragraph("ECG Signal Processing Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Analysis Parameters", styles['Heading2']))
        param_data = [
            ["Parameter", "Value"],
            ["Filename", params['filename']],
            ["Lead", params['lead']],
            ["Start Index", str(params['start'])],
            ["End Index", str(params['end'])],
            ["FIR Window", params['fir_window']],
            ["IIR Type", params['iir_type']],
            ["Sampling Frequency", f"{params['fs']} Hz"],
            ["Analysis Date", params['timestamp']]
        ]
        
        param_table = Table(param_data)
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(param_table)
        elements.append(Spacer(1, 24))
        
        def add_image_to_pdf(img_data, title):
            try:
                img_data_decoded = base64.b64decode(img_data)
                img_io = io.BytesIO(img_data_decoded)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(img_data_decoded)
                    tmp_path = tmp.name
                
                img = Image(tmp_path, width=6*inch, height=3*inch)
                elements.append(Paragraph(title, styles['Heading2']))
                elements.append(Spacer(1, 12))
                elements.append(img)
                elements.append(Spacer(1, 24))
                
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Error adding image to PDF: {e}")
                raise
        
        add_image_to_pdf(freq_response, "Frequency Response of Filters")
        add_image_to_pdf(time_domain, "Time Domain Analysis")
        add_image_to_pdf(freq_domain, "Frequency Domain Analysis")
        
        doc.build(elements)
        
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='ecg_analysis_report.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

@app.route('/download/<image_type>', methods=['GET'])
def download_image(image_type):
    try:
        params = {
            'filename': request.args.get('filename'),
            'lead': request.args.get('lead', 'MLII'),
            'start': int(request.args.get('start', 0)),
            'end': int(request.args.get('end', 2000)),
            'fir_window': request.args.get('fir_window', 'hamming'),
            'iir_type': request.args.get('iir_type', 'butter'),
            'fs': int(request.args.get('fs', 360))
        }
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], params['filename'])
        analyzer = ECGFilterAnalysis(filepath, fs=params['fs'])
        if not analyzer.load_data():
            raise Exception("Error loading data")
        
        if image_type == 'frequency_response':
            img_data = analyzer.plot_frequency_response(params['fir_window'], params['iir_type'])
            filename = f"frequency_response_{params['filename']}.png"
        elif image_type == 'time_domain':
            img_data = analyzer.analyze_and_plot(
                params['lead'], 
                params['start'], 
                params['end'], 
                params['fir_window'], 
                params['iir_type']
            )
            filename = f"time_domain_{params['filename']}.png"
        elif image_type == 'frequency_domain':
            img_data = analyzer.spectral_analysis(
                params['lead'], 
                params['start'], 
                params['end'], 
                params['fir_window'], 
                params['iir_type']
            )
            filename = f"frequency_domain_{params['filename']}.png"
        else:
            raise Exception("Invalid image type")
        
        img_bytes = base64.b64decode(img_data)
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': f'Failed to generate image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)