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
- Calculate an educational estimated SNR as positive mean intensity divided by
  population standard deviation when that mean is a meaningful signal reference.
- Apply a Gaussian filter to smooth nearby pixel variations.
- Calculate a 2D FFT and centered log-magnitude frequency spectrum.
- Run a MATLAB example for grayscale statistics, FFT visualization, and
  Gaussian smoothing.
- Run basic quality checks with caller-configurable thresholds.
- Read single-frame grayscale DICOM pixel data, apply rescale slope/intercept,
  and calculate basic measurements without returning identifying metadata.
- Flag potentially sensitive DICOM metadata field names and private-tag presence
  without reading or disclosing their values in CLI output or reports.
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
tests do not require a graphical display. It works with PNG and grayscale DICOM
images. DICOM visualization uses the decoded, rescaled pixel array and respects
`MONOCHROME1` grayscale inversion. Use `--report reports\my_report.txt`
to choose a report path; an existing report file is never overwritten.
Add `--frequency` to calculate the centered FFT and log-magnitude spectrum.
The report records only the spectrum dimensions, not the full arrays.

The DICOM reader supports single-frame grayscale images. It decodes pixel data,
applies rescale slope and intercept to a floating-point array, then calculates
statistics, SNR, quality checks, and reports without including patient metadata.
It also flags known potentially identifying field names and the presence of
private tags; it never returns or prints their values. Run it like this:

```powershell
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer data\sample\CT_small.dcm --frequency
```

To display a DICOM image in a Matplotlib window, add `--show`:

```powershell
& .\.venv\Scripts\python.exe -m medical_imaging_quality_analyzer data\sample\CT_small.dcm --show
```

DICOM files ending in `.dcm` are detected automatically; use `--dicom` to force
DICOM reading for files with another extension. DICOM mode can also apply the
existing Gaussian filter with `--filter`. The current visualizer accepts regular
image-file paths; DICOM display uses the decoded pixel array directly. Missing
pixel data, invalid dimensions, color images, multi-frame data, or
unsupported/corrupt pixel encodings produce a clear error.
DICOM pixel values remain unchanged in the source file; rescaling is applied
only to the array used for analysis. The included public sample may contain
potentially identifying metadata: use it only to exercise the privacy checks,
and do not print or copy its metadata values. All output is educational and is
not clinically validated or intended for diagnosis.

The simple SNR estimate is `mean intensity / population standard deviation` and
is reported only when the mean intensity is positive. Signed scales such as CT
values may have a negative mean; in that case the estimate is shown as
unavailable instead of a negative SNR. Also, whole-image standard deviation
contains both image structure and noise, so this ratio is not a physical or
clinically validated SNR measurement. A meaningful SNR requires a defined
signal reference and a suitable noise measurement.

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

The tests generate temporary synthetic images and DICOM files and, when present,
also verify the optional `data\sample\CT_small.dcm` fixture. No graphical display
is required. The DICOM fixture is not needed for the synthetic tests.

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
