import os
import re
from pathlib import Path

def parse_summary_file(summary_path, output_dir=None):
    """
    Parses CHB-MIT summary file to extract seizure annotations and saves as CSV.
    
    Args:
        summary_path (str): Path to the summary file (e.g., 'chb01-summary.txt')
        output_dir (str): Directory to save CSV files (default: same as summary file)
    
    Returns:
        dict: Mapping of EDF files to their seizure intervals
    """
    # Set output directory
    summary_path = Path(summary_path)
    if output_dir is None:
        output_dir = summary_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Read summary file
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    seizures = []
    all_annotations = {}

    for line in lines:
        # Detect new EDF file section
        if line.startswith('File Name:'):
            # Save previous file's seizures
            if current_file and seizures:
                save_annotations(current_file, seizures, output_dir)
                all_annotations[current_file] = seizures
            
            current_file = line.strip().split()[-1].replace('.edf', '')
            seizures = []

        # Extract seizure times (handles both "Seizure Start Time" and "Start Time")
        if 'Seizure Start Time' in line or ('Start Time' in line and 'Seizure' in lines[lines.index(line)-1]):
            start = extract_seconds(line)
        if 'Seizure End Time' in line or ('End Time' in line and 'Seizure' in lines[lines.index(line)-1]):
            end = extract_seconds(line)
            seizures.append((start, end))

    # Save the last file's annotations
    if current_file and seizures:
        save_annotations(current_file, seizures, output_dir)
        all_annotations[current_file] = seizures

    return all_annotations

def extract_seconds(line):
    """Extracts time in seconds from annotation line (supports hh:mm:ss and seconds formats)."""
    time_str = re.findall(r'[\d:]+', line)[-1]
    
    if ':' in time_str:  # hh:mm:ss format
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
    return int(time_str)  # Assume seconds if no colon

def save_annotations(edf_base, seizures, output_dir):
    """Saves seizure annotations to CSV file."""
    csv_path = output_dir / f'{edf_base}.edf.seizures.csv'
    with open(csv_path, 'w') as out:
        out.write('start,end\n')
        for start, end in seizures:
            out.write(f'{start},{end}\n')
    print(f'Saved: {csv_path}')

if __name__ == "__main__":
    # Example usage
    summary_file = './data/chbmit/chb01/chb01-summary.txt'
    output_directory = './data/chbmit/chb01/'
    
    annotations = parse_summary_file(summary_file, output_directory)
    print(f"\nProcessed {len(annotations)} files with {sum(len(s) for s in annotations.values())} total seizures")