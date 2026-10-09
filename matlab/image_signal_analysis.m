% Educational image and frequency analysis using the project's synthetic image.
% This script is not clinically validated and is not intended for diagnosis.

% Find the project root from this script so it works from any current folder.
scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
sampleImagePath = fullfile(projectFolder, 'data', 'sample', 'sample.png');

if ~isfile(sampleImagePath)
    error('Sample image not found: %s', sampleImagePath);
end

% Load the included synthetic grayscale sample.
imageData = imread(sampleImagePath);
if ndims(imageData) ~= 2
    error('Expected a grayscale image, but the sample has multiple channels.');
end

% Convert to double for measurements and image-processing calculations.
grayImage = double(imageData);
[imageHeight, imageWidth] = size(grayImage);
meanIntensity = mean(grayImage(:));
% Use population standard deviation (normalization by the number of pixels).
standardDeviation = std(grayImage(:), 1);

fprintf('Synthetic sample image statistics:\n');
fprintf('  Width: %d pixels\n', imageWidth);
fprintf('  Height: %d pixels\n', imageHeight);
fprintf('  Mean intensity: %.4f\n', meanIntensity);
fprintf('  Standard deviation: %.4f\n', standardDeviation);

% Compute the 2D Fast Fourier Transform and center its zero-frequency value.
fftResult = fft2(grayImage);
centeredFFT = fftshift(fftResult);

% The logarithm compresses the spectrum range to make details easier to view.
logMagnitudeSpectrum = log(1 + abs(centeredFFT));

% Show the original grayscale image and its centered frequency spectrum.
figure('Name', 'Image and Frequency Spectrum');
subplot(1, 2, 1);
imagesc(grayImage);
axis image;
colormap(gca, gray);
colorbar;
title('Synthetic Grayscale Image');
xlabel('Column');
ylabel('Row');

subplot(1, 2, 2);
imagesc(logMagnitudeSpectrum);
axis image;
colormap(gca, gray);
colorbar;
title('Centered Log-Magnitude Spectrum');
xlabel('Horizontal frequency bin');
ylabel('Vertical frequency bin');

% Smooth the image when Image Processing Toolbox is available. Otherwise,
% build a normalized Gaussian kernel and use base MATLAB's conv2 function.
gaussianSigma = 1;
if exist('imgaussfilt', 'file') == 2
    smoothedImage = imgaussfilt(grayImage, gaussianSigma);
    fprintf('Gaussian smoothing: imgaussfilt (Image Processing Toolbox).\n');
else
    kernelRadius = ceil(2 * gaussianSigma);
    [kernelX, kernelY] = meshgrid(-kernelRadius:kernelRadius);
    gaussianKernel = exp( ...
        -(kernelX.^2 + kernelY.^2) / (2 * gaussianSigma^2));
    gaussianKernel = gaussianKernel / sum(gaussianKernel(:));
    % conv2 with 'same' keeps image dimensions; outside pixels are treated as 0.
    smoothedImage = conv2(grayImage, gaussianKernel, 'same');
    fprintf(['Gaussian smoothing: base MATLAB conv2 fallback ' ...
        '(zero-padded image boundaries).\n']);
end

% Display the smoothed image separately from the original/spectrum plots.
figure('Name', 'Gaussian-Smoothed Image');
imagesc(smoothedImage);
axis image;
colormap(gca, gray);
colorbar;
title('Gaussian-Smoothed Image');
xlabel('Column');
ylabel('Row');
