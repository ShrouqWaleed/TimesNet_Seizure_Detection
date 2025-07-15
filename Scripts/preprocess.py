import os
import numpy as np
import pandas as pd
import mne
from pyedflib import highlevel
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from datetime import datetime

# Constants
SAMPLING_RATE = 256    # Target sampling rate (Hz)
WINDOW_SIZE = 1024     # 4-second windows (256Hz * 4s)
STEP_SIZE = 512        # 50% overlap (2-second stride)
BANDPASS_RANGE = (0.5, 40)  # Frequency range (Hz)
SEIZURE_BUFFER = 1     # Extra seconds around annotated seizures

def create_bandpass_filter(low_freq, high_freq, fs, order=4):
    """Design Butterworth bandpass filter coefficients."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    return butter(order, [low, high], btype='band')

def apply_filter(data, filter_coeffs):
    """Apply zero-phase bandpass filtering."""
    b, a = filter_coeffs
    return filtfilt(b, a, data)  # filtfilt avoids phase shift

def get_sample_rate(signal_headers, default=256):
    """Robust sampling rate extraction from EDF headers with multiple fallbacks."""
    try:
        # Check standard EDF header fields
        for field in ['sample_rate', 'sample_frequency']:
            if field in signal_headers[0]:
                return signal_headers[0][field]
        
        # Calculate from duration if available
        if 'duration' in signal_headers[0] and signal_headers[0]['duration'] > 0:
            return signal_headers[0]['sample_count'] / signal_headers[0]['duration']
            
    except Exception:
        pass
    
    return default  # Fallback to configured default

def read_edf_with_annotations(edf_path):
    """Load EDF file and corresponding seizure annotations."""
    try:
        # Read EDF using pyedflib (handles various formats)
        signals, signal_headers, header = highlevel.read_edf(edf_path)
        signals = np.array(signals, dtype=np.float32)
        actual_sample_rate = get_sample_rate(signal_headers, SAMPLING_RATE)
        
        print(f"Loaded {os.path.basename(edf_path)} | "
              f"SR: {actual_sample_rate}Hz | "
              f"Shape: {signals.shape} | "
              f"Duration: {signals.shape[1]/actual_sample_rate:.1f}s")
              
    except Exception as e:
        print(f"EDF read error {os.path.basename(edf_path)}: {str(e)}")
        return None, None, None

    # Load seizure annotations from companion CSV
    base_path = os.path.splitext(edf_path)[0]
    annotation_path = f"{base_path}.seizures.csv"
    seizures = []

    if os.path.exists(annotation_path):
        try:
            annotations = pd.read_csv(annotation_path)
            seizures = [(row['start'], row['end']) 
                      for _, row in annotations.iterrows()]
            print(f"   Found {len(seizures)} seizure annotations")
        except Exception as e:
            print(f"Annotation error: {str(e)}")
    
    return signals, actual_sample_rate, seizures

def preprocess_recording(edf_path, filter_coeffs):
    """Process single EDF file into windowed segments with labels."""
    signals, sample_rate, seizures = read_edf_with_annotations(edf_path)
    if signals is None:
        return [], []

    # Skip recordings shorter than window size
    n_channels, n_samples = signals.shape
    if n_samples < WINDOW_SIZE:
        print(f"Skipping short recording ({n_samples} < {WINDOW_SIZE} samples)")
        return [], []

    # Resample if needed (±5% tolerance)
    if abs(sample_rate - SAMPLING_RATE) > 0.05 * SAMPLING_RATE:
        print(f"Resampling {sample_rate}Hz → {SAMPLING_RATE}Hz")
        ratio = SAMPLING_RATE / sample_rate
        signals = mne.filter.resample(signals, down=ratio)
        sample_rate = SAMPLING_RATE

    # Apply preprocessing pipeline
    signals = apply_filter(signals, filter_coeffs)  # Bandpass filter
    signals = (signals - np.mean(signals, axis=1, keepdims=True)) / \
              (np.std(signals, axis=1, keepdims=True) + 1e-8)  # Z-score normalize

    # Convert seizure times to sample indices
    seizure_samples = [
        (int(start * sample_rate), int(end * sample_rate))
        for start, end in seizures
    ]
    
    # Sliding window segmentation
    segments, labels = [], []
    for i in range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE):
        window_start, window_end = i, i + WINDOW_SIZE
        
        # Label 1 if window overlaps with any seizure (+buffer)
        label = 0
        for seizure_start, seizure_end in seizure_samples:
            buffered_start = seizure_start - int(SEIZURE_BUFFER * sample_rate)
            buffered_end = seizure_end + int(SEIZURE_BUFFER * sample_rate)
            
            if not (window_end < buffered_start or window_start > buffered_end):
                label = 1
                break

        segments.append(signals[:, window_start:window_end])
        labels.append(label)

    print(f"   Generated {len(labels)} windows ({sum(labels)} seizure windows)\n")
    return segments, labels

def main(data_dir=None, output_dir=None):
    """Run full preprocessing pipeline."""
    print(f"\n{'='*60}")
    print(f"EEG Preprocessing Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Path handling (Windows compatible)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'chbmit', 'chb01')
    if output_dir is None:
        output_dir = os.path.join(project_root, 'processed')

    # Configuration summary
    print(f"Configuration:")
    print(f"- Data dir: {data_dir}")
    print(f"- Output dir: {output_dir}")
    print(f"- Target SR: {SAMPLING_RATE}Hz")
    print(f"- Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE:.1f}s)")
    print(f"- Step: {STEP_SIZE} samples ({STEP_SIZE/SAMPLING_RATE:.1f}s)\n")

    # Validate inputs
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Find EDF files
    edf_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.lower().endswith('.edf')
    ])
    
    if not edf_files:
        print(f"No EDF files in: {data_dir}")
        return

    print(f"Found {len(edf_files)} EDF files (sample: {edf_files[:3]})\n")

    # Initialize processing
    os.makedirs(output_dir, exist_ok=True)
    filter_coeffs = create_bandpass_filter(*BANDPASS_RANGE, SAMPLING_RATE)
    all_segments, all_labels = [], []

    # Process each file
    for edf_file in tqdm(edf_files, desc="Processing EDFs"):
        edf_path = os.path.join(data_dir, edf_file)
        segments, labels = preprocess_recording(edf_path, filter_coeffs)
        
        if segments:
            all_segments.extend(segments)
            all_labels.extend(labels)

    # Save results
    if all_segments:
        X = np.stack(all_segments)  # [n_windows, channels, time]
        y = np.array(all_labels, dtype=np.int32)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(os.path.join(output_dir, f'eeg_data_{timestamp}.npy'), X)
        np.save(os.path.join(output_dir, f'eeg_labels_{timestamp}.npy'), y)

        print(f"\n{'='*60}")
        print(f"Completed: {X.shape[0]} windows")
        print(f"Seizure ratio: {sum(y)}/{len(y)} ({100*sum(y)/len(y):.1f}%)")
        print(f"Output saved to {output_dir}")
        print(f"{'='*60}")
    else:
        print("No valid data processed!")

if __name__ == "__main__":
    main()