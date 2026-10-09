# Medical Imaging Quality Analyzer

A learning project for exploring image-quality measurements using sample data.

This project is for educational use only. It is not clinically validated and does
not diagnose disease. Use synthetic sample data only. Do not add real patient
information, even if you believe it has been de-identified.

## Project layout

- `src/medical_imaging_quality_analyzer/` — application code
- `tests/` — automated tests
- `data/sample/` — sample images for learning and testing
- `reports/` — generated analysis reports

## Features

- Load regular image files as grayscale and calculate dimensions, minimum and
  maximum pixel values, mean intensity, and standard deviation.
- Display a grayscale image beside its pixel-intensity histogram when requested.
- Calculate an educational estimated SNR as mean intensity divided by population
  standard deviation.
- Apply a Gaussian filter to smooth nearby pixel variations.
- Calculate a 2D FFT and centered log-magnitude frequency spectrum.
- Run a MATLAB example for grayscale statistics, FFT visualization, and
  Gaussian smoothing.
- Run basic quality checks with caller-configurable thresholds.
- Read single-frame grayscale DICOM pixel data, apply rescale slope/intercept,
  and calculate basic measurements without returning identifying metadata.
- Generate text reports without including arbitrary metadata fields.

The DICOM reader returns decoded single-frame grayscale pixel data with
dimensions, modality, and pixel spacing. Rescale slope and intercept are applied
to a floating-point copy for analysis; the stored file is not changed. Patient
names, IDs, birth dates, and other identifying metadata are not returned or
included in reports.

## Installation

From the project root in Windows PowerShell, create and install into a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository includes a small synthetic grayscale pixel-ramp image at
`data\sample\sample.png`. Only this reviewed image and the sample-data note are
allowed by `.gitignore`; other files placed under `data/` are ignored by default.
Do not add real patient information anywhere in the project.

## Run the command-line application

From the project root, set the source directory for this PowerShell session and
analyze the included sample:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer data\sample\sample.png
```

To analyze another **synthetic** image, replace the final path with its path.
Do not use patient images.

Optional image display and Gaussian filtering:

```powershell
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer data\sample\sample.png --show --filter
```

`--show` opens a Matplotlib window and is optional; normal runs and automated
tests do not require a graphical display. Use `--report reports\my_report.txt`
to choose a report path; an existing report file is never overwritten.

The DICOM reader supports single-frame grayscale images. It decodes pixel data,
applies rescale slope and intercept to a floating-point array, then calculates
statistics, SNR, quality checks, and reports without including patient metadata.
Run it like this:

```powershell
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer path\to\sample.dcm --dicom
```

DICOM mode can also apply the existing Gaussian filter with `--filter`. The
current visualizer accepts regular image-file paths, so `--show` is not
supported with DICOM input. Missing pixel data, color images, multi-frame data,
or unsupported/corrupt pixel encodings produce a clear error. DICOM pixel values
remain unchanged in the source file; rescaling is applied only to the array used
for analysis. All output is educational and is not clinically validated or
intended for diagnosis.

To see every available option:

```powershell
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer --help
```

## Run tests

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests generate temporary synthetic images and DICOM files; they do not need
real patient data or a graphical display.

## Frequency-analysis example

The frequency-analysis helper uses NumPy's 2D Fast Fourier Transform (FFT).
`fftshift` moves the zero-frequency component to the center, and the
log-magnitude spectrum is calculated as `log1p(abs(shifted_fft))` for easier
visualization. For example:

```python
import numpy as np

from medical_imaging_quality_analyzer.frequency_analysis import analyze_frequency

sample_image = np.zeros((8, 8), dtype=np.float64)
sample_image[3, 3] = 1.0
frequency_data = analyze_frequency(sample_image)

print(frequency_data["fft"])  # Complex, unshifted FFT values
print(frequency_data["shifted_fft"])  # Zero frequency at the center
print(frequency_data["log_magnitude_spectrum"])  # Visualization-ready values
```

Frequency analysis is included for educational signal processing only; it is
not clinically validated and is not intended for diagnosis.

## MATLAB image and signal analysis

The script `matlab\image_signal_analysis.m` loads the included synthetic
grayscale image, prints its dimensions, mean intensity, and population standard
deviation, then displays the image beside its centered log-magnitude FFT
spectrum. A second figure shows Gaussian smoothing.

From the project root in MATLAB, run:

```matlab
run('matlab/image_signal_analysis.m')
```

Or from Windows PowerShell with MATLAB installed:

```powershell
& 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe' -batch "run('C:\path\to\Medical-Imaging-Quality-Analyzer\matlab\image_signal_analysis.m')"
```

The script uses base MATLAB for image loading, statistics, FFT, plots, and its
Gaussian-filter fallback. If Image Processing Toolbox provides `imgaussfilt`,
that function is used; otherwise, the script builds a normalized Gaussian
kernel and applies it with base MATLAB `conv2`. The fallback keeps the image
dimensions and treats pixels outside the image as zero.

The input is the synthetic `data\sample\sample.png`; do not substitute patient
images. This MATLAB example is for education only and is not clinically
validated or intended for diagnosis.

## Data, reports, and limitations

Generated text reports in `reports/` are ignored by Git by default. Common DICOM
and medical-image formats are also ignored because they may contain patient
identifiers. Local environment files, credential files, and private-data folders
are ignored as well. Keep patient data and credentials outside this project.
Ignore rules are safeguards, not a detector: inspect every file you plan to add
and commit synthetic sample data only.

Image statistics, SNR, Gaussian filtering, and quality checks are basic
educational examples. Their configurable thresholds are not universal standards.
Nothing in this project is clinically validated or suitable for diagnosis or
clinical decision-making.
