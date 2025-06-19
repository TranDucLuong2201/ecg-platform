document.addEventListener('DOMContentLoaded', () => {
    const backendUrl = 'http://127.0.0.1:5000/api'; // Thay đổi nếu backend của bạn chạy ở cổng khác

    const ecgFileInput = document.getElementById('ecgFileInput');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileInfo = document.getElementById('fileInfo');

    const signalColumnSelect = document.getElementById('signalColumn');
    const startSampleInput = document.getElementById('startSample');
    const endSampleInput = document.getElementById('endSample');
    const applySegmentButton = document.getElementById('applySegmentButton');

    const firWindowSelect = document.getElementById('firWindow');
    const iirTypeSelect = document.getElementById('iirType');
    const cutoffLowInput = document.getElementById('cutoffLow');
    const cutoffHighInput = document.getElementById('cutoffHigh');
    const applyFilterButton = document.getElementById('applyFilterButton');
    const showFrequencyResponseButton = document.getElementById('showFrequencyResponseButton');

    const signalPlot = document.getElementById('signalPlot');
    const signalPlotStatus = document.getElementById('signalPlotStatus');
    const frequencyPlot = document.getElementById('frequencyPlot');
    const frequencyPlotStatus = document.getElementById('frequencyPlotStatus');

    let currentDataId = null;
    let totalSamples = 0;
    let availableColumns = [];
    let ecgColumns = [];

    // --- Utility Functions ---
    function showStatus(element, message, type = 'info') {
        element.textContent = message;
        element.className = type === 'success' ? 'success' : (type === 'error' ? 'error' : '');
    }

    function setPlotImage(imgElement, statusElement, base64Image) {
        if (base64Image) {
            imgElement.src = `data:image/png;base64,${base64Image}`;
            statusElement.textContent = '';
        } else {
            imgElement.src = '';
            statusElement.textContent = 'No plot available.';
        }
    }

    function enableControls(enable) {
        signalColumnSelect.disabled = !enable;
        startSampleInput.disabled = !enable;
        endSampleInput.disabled = !enable;
        applySegmentButton.disabled = !enable;
        firWindowSelect.disabled = !enable;
        iirTypeSelect.disabled = !enable;
        cutoffLowInput.disabled = !enable;
        cutoffHighInput.disabled = !enable;
        applyFilterButton.disabled = !enable;
        showFrequencyResponseButton.disabled = !enable;
    }

    // Disable controls initially
    enableControls(false);

    // --- Event Listeners ---

    uploadButton.addEventListener('click', async () => {
        const file = ecgFileInput.files[0];
        if (!file) {
            showStatus(uploadStatus, 'Please select a CSV file.', 'error');
            return;
        }

        showStatus(uploadStatus, 'Uploading and analyzing file...', 'info');
        enableControls(false); // Disable controls during upload/initial analysis

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${backendUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                showStatus(uploadStatus, data.message, 'success');
                currentDataId = data.data_id;
                totalSamples = data.total_samples;
                availableColumns = data.columns;
                ecgColumns = data.ecg_columns;

                fileInfo.innerHTML = `
                    <p><strong>File ID:</strong> ${currentDataId}</p>
                    <p><strong>Total Samples:</strong> ${totalSamples}</p>
                    <p><strong>Available Columns:</strong> ${availableColumns.join(', ')}</p>
                    <p><strong>Suggested ECG Columns:</strong> ${ecgColumns.join(', ') || 'None'}</p>
                `;

                // Populate signal column dropdown
                signalColumnSelect.innerHTML = '';
                availableColumns.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;
                    signalColumnSelect.appendChild(option);
                });

                // Select the first suggested ECG column if available, otherwise the first column
                if (ecgColumns.length > 0) {
                    signalColumnSelect.value = ecgColumns[0];
                } else if (availableColumns.length > 0) {
                    signalColumnSelect.value = availableColumns[0];
                }

                // Set initial segment range
                startSampleInput.value = 0;
                endSampleInput.value = Math.min(2000, totalSamples); // Default to first 2000 samples or less if file is smaller

                // Display initial plot
                setPlotImage(signalPlot, signalPlotStatus, data.initial_plot);
                setPlotImage(frequencyPlot, frequencyPlotStatus, ''); // Clear freq plot

                enableControls(true); // Enable controls after successful upload
            } else {
                showStatus(uploadStatus, `Error: ${data.error}`, 'error');
                enableControls(false);
            }
        } catch (error) {
            showStatus(uploadStatus, `Network error: ${error.message}`, 'error');
            console.error('Upload error:', error);
            enableControls(false);
        }
    });

    applySegmentButton.addEventListener('click', () => processSignal());
    applyFilterButton.addEventListener('click', () => processSignal());
    showFrequencyResponseButton.addEventListener('click', () => getFrequencyResponse());

    async function processSignal() {
        if (!currentDataId) {
            showStatus(signalPlotStatus, 'Please upload a file first.', 'error');
            return;
        }

        const selectedColumn = signalColumnSelect.value;
        const start = parseInt(startSampleInput.value);
        let end = parseInt(endSampleInput.value);

        if (isNaN(start) || isNaN(end) || start < 0 || end <= start || end > totalSamples) {
            showStatus(signalPlotStatus, 'Invalid start/end sample range. Please check values.', 'error');
            return;
        }
        
        // Adjust end if it's greater than total samples
        if (end > totalSamples) {
            end = totalSamples;
            endSampleInput.value = totalSamples; // Update UI
        }


        showStatus(signalPlotStatus, 'Processing signal...', 'info');

        const params = {
            data_id: currentDataId,
            column: selectedColumn,
            start: start,
            end: end,
            fir_window: firWindowSelect.value,
            iir_type: iirTypeSelect.value,
            cutoff_low: parseFloat(cutoffLowInput.value),
            cutoff_high: parseFloat(cutoffHighInput.value)
        };

        try {
            const response = await fetch(`${backendUrl}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            const data = await response.json();

            if (data.success) {
                setPlotImage(signalPlot, signalPlotStatus, data.plot_image);
                showStatus(signalPlotStatus, 'Signal processed successfully.', 'success');
            } else {
                setPlotImage(signalPlot, signalPlotStatus, '');
                showStatus(signalPlotStatus, `Error: ${data.error}`, 'error');
            }
        } catch (error) {
            setPlotImage(signalPlot, signalPlotStatus, '');
            showStatus(signalPlotStatus, `Network error: ${error.message}`, 'error');
            console.error('Process signal error:', error);
        }
    }

    async function getFrequencyResponse() {
        if (!currentDataId) {
            showStatus(frequencyPlotStatus, 'Please upload a file first.', 'error');
            return;
        }

        showStatus(frequencyPlotStatus, 'Generating frequency response...', 'info');

        const params = {
            fir_window: firWindowSelect.value,
            iir_type: iirTypeSelect.value,
            cutoff_low: parseFloat(cutoffLowInput.value),
            cutoff_high: parseFloat(cutoffHighInput.value)
        };

        try {
            const response = await fetch(`${backendUrl}/frequency_response`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            const data = await response.json();

            if (data.success) {
                setPlotImage(frequencyPlot, frequencyPlotStatus, data.plot_image);
                showStatus(frequencyPlotStatus, 'Frequency response generated successfully.', 'success');
            } else {
                setPlotImage(frequencyPlot, frequencyPlotStatus, '');
                showStatus(frequencyPlotStatus, `Error: ${data.error}`, 'error');
            }
        } catch (error) {
            setPlotImage(frequencyPlot, frequencyPlotStatus, '');
            showStatus(frequencyPlotStatus, `Network error: ${error.message}`, 'error');
            console.error('Frequency response error:', error);
        }
    }

    // Initial setup when page loads
    setPlotImage(signalPlot, signalPlotStatus, '');
    setPlotImage(frequencyPlot, frequencyPlotStatus, '');
});