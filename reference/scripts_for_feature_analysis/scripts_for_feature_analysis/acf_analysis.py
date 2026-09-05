#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Image Autocorrelation Analysis Tool

FUNCTIONALITY:
This script performs autocorrelation-based feature size identification on 3D image data.
It loads a 3D TIFF image, allows optional cropping and denoising, calculates the
autocorrelation function, and identifies characteristic length scales at different
correlation thresholds (1/e, 10%, 1%).

USAGE:
1. Run the script: python autocorrelation_analysis.py
2. Select TIFF file through UI dialog
3. Select output directory through UI dialog
4. Choose options for:
   - Layer cropping (extract specific number of layers from center)
   - ROI selection (crop regions of interest from each layer)
   - Denoising (apply noise reduction filters)
5. View autocorrelation analysis results and save plots

DEPENDENCIES:
- numpy, scipy, matplotlib, scikit-image, opencv-python, tifffile, tkinter

ATTRIBUTION: This code was originally developed based on the IMPPY3D framework
             (https://github.com/usnistgov/imppy3d). See IMPPY3D_README.md and
             IMPPY3D_LICENSE.txt in this directory for the original library
             documentation and license terms.

AUTHORS: Zixiang (Zach) Tong, @UT-Austin; Yujie Zhang, @UT-Austin;
         Dr. Alexander K. Landauer, NIST
DATE: 2025.05.25
NOTE: Claude AI (Anthropic) assisted with code formatting, structure optimization,
      and documentation standardization
"""


import numpy as np
import cv2 as cv
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RectangleSelector
from numpy import log10
from scipy import ndimage as ndim
from scipy.interpolate import interp1d
from scipy.fft import rfftn, irfftn, next_fast_len
import tifffile as tiff
import csv
import scipy.io as sio
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import os

# Global constants
EPS_FLOAT = 1e-12


# ---------------------------------------------------------------------------
# Denoising helpers (inlined from imppy3d cv_interactive_processing)
# ---------------------------------------------------------------------------

def _do_nothing(x):
    """Dummy callback required by cv.createTrackbar."""
    pass


def _interact_denoise(img_in):
    """
    Interactive non-local means denoising using OpenCV trackbars.

    Opens a window showing the image and three trackbars (Filter Strength,
    Template Patch Size, Window Size). The user tunes the parameters in
    real time; pressing Enter or Esc closes the window and returns the
    denoised image together with the chosen parameters.

    Args:
        img_in (np.ndarray): Grayscale uint8 image (2-D).

    Returns:
        list: [img_denoise, denoise_params] where denoise_params is
              [cur_h, cur_tsize, cur_wsize].
    """
    img = img_in.copy()
    img_title = "Non-Local Means Denoising"

    print("\nPress [Enter] while the new window is active to continue...")
    print("\nWARNING: This non-local means denoising filter is very "
          "computationally costly!\n    The cost scales approximately linearly"
          " for increasing values of\n    Window Size and Template Patch Size.")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL)
    cv.moveWindow(img_title, 1, 1)

    cv.createTrackbar("Filter Strength",    img_title, 0, 50, _do_nothing)
    cv.createTrackbar("Template Patch Size", img_title, 0, 25, _do_nothing)
    cv.createTrackbar("Window Size",         img_title, 0, 51, _do_nothing)

    cv.setTrackbarPos("Filter Strength",    img_title, 3)
    cv.setTrackbarPos("Template Patch Size", img_title, 7)
    cv.setTrackbarPos("Window Size",         img_title, 21)

    img_denoise = img
    while True:
        cur_h     = cv.getTrackbarPos("Filter Strength",    img_title)
        cur_tsize = cv.getTrackbarPos("Template Patch Size", img_title)
        cur_wsize = cv.getTrackbarPos("Window Size",         img_title)

        if cur_h <= 0:
            cv.imshow(img_title, img)
            img_denoise = img
        else:
            if cur_tsize % 2 == 0:
                cur_tsize -= 1
            if cur_wsize % 2 == 0:
                cur_wsize -= 1
            img_denoise = cv.fastNlMeansDenoising(
                img, h=cur_h,
                templateWindowSize=cur_tsize,
                searchWindowSize=cur_wsize
            )
            cv.imshow(img_title, img_denoise)

        key_pressed = cv.waitKey(200) & 0xFF
        if key_pressed in [27, 10, 13]:
            break

    cv.destroyAllWindows()
    return [img_denoise, [cur_h, cur_tsize, cur_wsize]]


def select_output_directory():
    """
    Select output directory using GUI dialog.
    
    Returns:
        str: Selected directory path
    """
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Force dialog to front

    dir_out_path = askdirectory(title="Select a folder to save the output files")
    print(f"The selected folder path is: {dir_out_path}")
    
    return dir_out_path


def select_tiff_file():
    """
    Select TIFF file using GUI dialog.
    
    Returns:
        tuple: (file_path, file_name_without_extension)
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Force dialog to front

    tif_file_path = askopenfilename(
        title="Select a .tif/tiff file",
        filetypes=[("TIF/TIFF files", "*.tiff *.tif")]
    )
    
    if not tif_file_path:  # If no file is selected, exit the program
        print("No file selected. Exiting...")
        quit()
    
    tif_file_name = os.path.splitext(os.path.basename(tif_file_path))[0]
    
    return tif_file_path, tif_file_name


def load_tiff_stack(tif_file_path):
    """
    Load 3D TIFF stack from file.
    
    Args:
        tif_file_path (str): Path to TIFF file
        
    Returns:
        np.ndarray: 3D numpy array containing image stack
    """
    try:
        with tiff.TiffFile(tif_file_path) as tif:
            # Read all frames from the .tif file
            imgs = [page.asarray() for page in tif.pages]  # Extract each frame into a list
            imgs = np.array(imgs)  # Convert the list of frames into 3D numpy array
            print(f"Loaded {len(imgs)} frames with shape {imgs[0].shape} per frame")
    except FileNotFoundError:
        print(f"File not found: {tif_file_path}")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()
    
    return imgs


def crop_layers_interactive(imgs):
    """
    Interactively crop layers from the image stack.
    
    Args:
        imgs (np.ndarray): Input image stack
        
    Returns:
        np.ndarray: Cropped image stack
    """
    total_layers = imgs.shape[0]
    cropped_layer = total_layers  # Default: use all layers (avoids NameError when user skips cropping)
    print(f"Total available layers: {total_layers}")

    layer_option = input("Crop layers? (Enter=no, 1=yes): ").strip() or '0'

    if layer_option == '1':
        while True:
            cropped_layer_input = input(f"Number of layers to extract (positive integer, max {total_layers}): ").strip()
            if not cropped_layer_input.isdigit() or int(cropped_layer_input) <= 0:
                print("Please enter a positive integer.")
                continue
            cropped_layer = int(cropped_layer_input)
            if cropped_layer > total_layers:
                print(f"Error: {cropped_layer} exceeds total layers ({total_layers}). Please try again.")
                continue
            break

        # Extract center layers
        start = (total_layers - cropped_layer) // 2
        end = start + cropped_layer
        imgs = imgs[start:end]
        print(f"Extracted layers {start} to {end - 1} (center crop). New shape: {imgs.shape}")
    else:
        print(f"No layer cropping applied. Using all {total_layers} layers.")

    return imgs, cropped_layer


def select_roi_interactive(imgs):
    """
    Defines the ROI based on user input for center coordinates and dimensions.
    Displays a preview of the defined ROI on the middle layer.
    
    Args:
        imgs (np.ndarray): Input image stack
        
    Returns:
        tuple: (roi_coords, roi_selected_flag)
               roi_coords is (y_start, y_end, x_start, x_end)
    """
    height, width = imgs.shape[1], imgs.shape[2]

    roi_option = input("Crop each layer to an ROI? (Enter=no, 1=yes): ").strip() or '0'

    if roi_option == '1':
        print(f"Image dimensions: Width={width} px, Height={height} px")

        # 1. Get center coordinates
        while True:
            try:
                center_x = int(input(f"ROI center X (0 to {width - 1}, image width={width}): ").strip())
                center_y = int(input(f"ROI center Y (0 to {height - 1}, image height={height}): ").strip())
                if 0 <= center_x < width and 0 <= center_y < height:
                    break
                else:
                    print(f"Coordinates out of bounds. X must be 0-{width-1}, Y must be 0-{height-1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # 2. Get rectangular dimensions (even integers required for symmetric centering)
        while True:
            try:
                roi_width = int(input("ROI width in pixels (positive even integer): ").strip())
                roi_height = int(input("ROI height in pixels (positive even integer): ").strip())
                if roi_width > 0 and roi_height > 0 and roi_width % 2 == 0 and roi_height % 2 == 0:
                    break
                else:
                    print("Width and height must be positive even integers (required for symmetric centering).")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # 3. Calculate ROI coordinates (y_start, y_end, x_start, x_end)
        half_w = roi_width // 2
        half_h = roi_height // 2

        x_start = max(0, center_x - half_w)
        x_end = min(width, center_x + half_w)
        y_start = max(0, center_y - half_h)
        y_end = min(height, center_y + half_h)

        roi_coords = (y_start, y_end, x_start, x_end)
        print(f"ROI coordinates: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]  "
              f"(actual size: {x_end - x_start} x {y_end - y_start} px)")

        # 4. Display ROI preview using OpenCV (avoids matplotlib/TkAgg conflict on Windows)
        middle_layer = imgs[imgs.shape[0] // 2]
        if middle_layer.dtype == np.uint16:
            display_img = (middle_layer >> 8).astype(np.uint8)
        elif middle_layer.dtype != np.uint8:
            # Normalize to uint8 for any other dtype
            mn, mx = middle_layer.min(), middle_layer.max()
            if mx > mn:
                display_img = ((middle_layer - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                display_img = np.zeros_like(middle_layer, dtype=np.uint8)
        else:
            display_img = middle_layer.copy()

        display_bgr = cv.cvtColor(display_img, cv.COLOR_GRAY2BGR)
        cv.rectangle(display_bgr, (x_start, y_start), (x_end - 1, y_end - 1), (0, 0, 255), 2)
        label = f"ROI {x_end - x_start}x{y_end - y_start} @ ({center_x},{center_y})"
        cv.putText(display_bgr, label, (x_start, max(y_start - 8, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)

        try:
            cv.imshow("ROI Preview (center slice) - press any key to continue", display_bgr)
            cv.waitKey(0)
            cv.destroyAllWindows()
            # On Windows the Enter/Space used to dismiss the OpenCV window can leak
            # into stdin and silently answer the next input() prompt.  Drain the
            # keyboard buffer here to prevent that race condition.
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except ImportError:
                pass  # Not on Windows; no action needed
        except cv.error:
            print(f"[INFO] Cannot open preview window. ROI confirmed: X=[{x_start}:{x_end}], Y=[{y_start}:{y_end}]")
        print("[ROI preview closed]")

        return roi_coords, True, center_x, center_y, roi_width, roi_height

    return None, False, 0, 0, 0, 0

def save_cropped_tiff(imgs, dir_out_path, tif_file_name, layer_count):
    """
    Saves the cropped 3D image stack to a new TIFF file.
    
    Args:
        imgs (np.ndarray): The image stack to save
        dir_out_path (str): Output directory path
        tif_file_name (str): Base filename for naming the output
    """
    output_path = os.path.join(dir_out_path, f"{tif_file_name}_Layer{layer_count}_cropped_ROI.tif")
    try:
        # Use tifffile to save the 3D numpy array
        tiff.imwrite(output_path, imgs)
        print(f"\nSuccessfully saved cropped image stack to: {output_path}")
    except Exception as e:
        print(f"Error saving cropped TIFF file: {e}")


def apply_roi_cropping(imgs, roi_coords, roi_selected, denoise_option):
    """
    Apply ROI cropping to image stack.
    
    Args:
        imgs (np.ndarray): Input image stack
        roi_coords (tuple): ROI coordinates (y_start, y_end, x_start, x_end)
        roi_selected (bool): Whether ROI was selected
        denoise_option (str): Denoising option
        
    Returns:
        np.ndarray: Processed image stack
    """
    if roi_selected and roi_coords:
        y_start, y_end, x_start, x_end = roi_coords
        
        # Ensure even dimensions
        if (y_end - y_start) % 2 == 1:
            y_end = y_end - 1
        if (x_end - x_start) % 2 == 1:
            x_end = x_end - 1
        
        roi_rows = y_end - y_start
        roi_cols = x_end - x_start
        
        cropped_imgs = np.zeros((imgs.shape[0], roi_rows, roi_cols), dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            cropped_imgs[i] = imgs[i, y_start:y_end, x_start:x_end]
    else:
        print("No cropping!")
        height, width = imgs.shape[1], imgs.shape[2]
        
        # Keep dimensions even
        roi_rows = height - 1 if height % 2 == 1 else height
        roi_cols = width - 1 if width % 2 == 1 else width
        
        cropped_imgs = np.zeros((imgs.shape[0], roi_rows, roi_cols), dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            cropped_imgs[i] = imgs[i, :roi_rows, :roi_cols]
    
    # Handle data type conversion for denoising
    if cropped_imgs.dtype == np.uint16 and (denoise_option == '1'):
        imgs = (cropped_imgs >> 8).astype(np.uint8)
    else:
        imgs = cropped_imgs
    
    return imgs


def apply_denoising(imgs, denoise_option):
    """
    Apply denoising to image stack if requested.
    
    Args:
        imgs (np.ndarray): Input image stack
        denoise_option (str): Denoising option ('0' or '1')
        
    Returns:
        np.ndarray: Denoised image stack
    """
    if denoise_option == '1':
        num_imgs, num_rows, num_cols = imgs.shape
        img_denoise = np.zeros((num_cols, num_rows), dtype=np.uint8)
        
        # Interactive denoising parameter selection using middle image
        [img_denoise, denoise_params] = _interact_denoise(imgs[round(num_imgs/2), :, :])
        
        # Apply denoising to all images
        for img_index in range(num_imgs):
            imgs[img_index, :, :] = cv.fastNlMeansDenoising(
                imgs[img_index, :, :], 
                h=denoise_params[0],
                templateWindowSize=denoise_params[1], 
                searchWindowSize=denoise_params[2]
            )
            if np.mod(img_index, 10.0) <= 2 * EPS_FLOAT:
                print(f"\nCurrent denoise image: {img_index}")
    
    return imgs


def compute_autocorrelation(imgs):
    """
    Compute autocorrelation function for 3D image stack.

    Uses the Wiener-Khinchin theorem (ACF = IFFT(|FFT(x)|^2)) with float32
    precision and real-valued FFT to keep peak memory manageable.

    Args:
        imgs (np.ndarray): Input image stack

    Returns:
        tuple: (autocorrelation_volume, coordinate_arrays)
    """
    print('\nRunning cross correlation...')

    # Transpose for proper correlation computation: (layers, rows, cols) -> (rows, cols, layers)
    acf_vol = imgs.transpose([1, 2, 0]).astype(np.float32)

    # Subtract mean for proper autocorrelation
    acf_vol -= acf_vol.mean()

    # Padded shape for linear (non-circular) autocorrelation
    pad_shape = tuple(2 * s - 1 for s in acf_vol.shape)
    fft_shape = tuple(next_fast_len(s) for s in pad_shape)

    print(f'  FFT shape: {fft_shape}  (float32 + rfftn for memory efficiency)')

    # Wiener-Khinchin: ACF = IFFT(|FFT(x)|^2)
    F = rfftn(acf_vol, s=fft_shape)
    del acf_vol

    # Power spectrum (in-place to save memory)
    F *= np.conj(F)

    acf_result = irfftn(F, s=fft_shape)
    del F

    # Crop to valid autocorrelation size
    acf_result = acf_result[:pad_shape[0], :pad_shape[1], :pad_shape[2]]

    # Normalize so zero-lag = 1.0
    acf_result /= acf_result[0, 0, 0]

    # Shift so zero-lag is at the center
    nxcorrcoeff_vol = np.fft.fftshift(acf_result)
    del acf_result

    print('\nCross correlation complete')
    
    # Set up coordinate system
    size_acf = [0, 0, 0]
    for dim in range(3):
        size_acf[dim] = nxcorrcoeff_vol.shape[dim]
    
    px_x_v = np.linspace(1, size_acf[0], size_acf[0])
    px_y_v = np.linspace(1, size_acf[1], size_acf[1])
    px_z_v = np.linspace(1, size_acf[2], size_acf[2])
    
    [X, Y, Z] = np.meshgrid(px_y_v, px_x_v, px_z_v)
    
    X = X - np.fix(size_acf[1]/2) - 1
    Y = Y - np.fix(size_acf[0]/2) - 1
    Z = Z - np.fix(size_acf[2]/2) - 1
    
    # Compute radius for each point in the cartesian grid
    r = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    
    return nxcorrcoeff_vol, r


def compute_radial_autocorrelation(nxcorrcoeff_vol, r):
    """
    Compute radial autocorrelation statistics.
    
    Args:
        nxcorrcoeff_vol (np.ndarray): Autocorrelation volume
        r (np.ndarray): Radius array
        
    Returns:
        tuple: (distances, mean_correlations, std_correlations, max_radius)
    """
    # For each integer radius compute the mean and std dev autocorrelation coefficient
    r_fix = np.fix(r)
    r_max = int(np.max(r_fix))
    corr_coeff_mean = np.zeros([r_max+1, 1])
    corr_coeff_std = np.zeros([r_max+1, 1])
    
    for r_val in range(r_max+1):
        corr_coeff_mean[r_val] = np.mean(nxcorrcoeff_vol[r_fix == r_val])
        corr_coeff_std[r_val] = np.std(nxcorrcoeff_vol[r_fix == r_val])
    
    distance = np.arange(0, r_max+1)
    
    return distance, corr_coeff_mean, corr_coeff_std, r_max


def find_autocorrelation_lengths(corr_coeff_mean, distance, r_max):
    """
    Find characteristic autocorrelation lengths at specific thresholds.
    
    Args:
        corr_coeff_mean (np.ndarray): Mean correlation coefficients
        distance (np.ndarray): Distance array
        r_max (int): Maximum radius
        
    Returns:
        tuple: (characteristic_lengths, target_values, interpolation_data)
    """
    # Ensure positive values for interpolation
    corr_coeff_mean[corr_coeff_mean < 0] = 0.0000000001
    y_values = np.array(corr_coeff_mean).flatten()
    
    x_results = []
    y_targets = []
    
    # Find AC_1/e (≈ 0.3679)
    for i in range(r_max):
        if y_values[i] >= 0.3679 > y_values[i + 1]:
            x1, x2 = distance[i], distance[i+1]
            x_targets1 = [x1, x2]
            y1 = y_values[i]
            y2 = y_values[i+1]
            y_targets1 = [y1, y2]
            break
    
    f1 = interp1d(y_targets1, x_targets1, kind='linear', 
                  bounds_error=False, fill_value="extrapolate")
    x_solution1 = f1(0.3679)
    x_results.append(x_solution1)
    y_targets.append(0.3679)
    
    # Find AC_1/10 (0.1)
    xy_position1 = r_max  # Safe default if threshold is never crossed
    x_targets2 = [distance[r_max - 1], distance[r_max]]
    y_targets2 = [y_values[r_max - 1], y_values[r_max]]
    y3, y4 = y_targets2
    for i in range(r_max):
        if y_values[i] >= 0.1 > y_values[i + 1]:
            x3, x4 = distance[i], distance[i + 1]
            x_targets2 = [x3, x4]
            y3 = y_values[i]
            y4 = y_values[i + 1]
            y_targets2 = [y3, y4]
            xy_position1 = i + 1
            break
    
    f2 = interp1d(y_targets2, x_targets2, kind='linear', 
                  bounds_error=False, fill_value="extrapolate")
    x_solution2 = f2(0.1)
    x_results.append(x_solution2)
    y_targets.append(0.1)
    
    # Find AC_1/100 (0.01) if possible
    interp_data = {}
    if y_values[-1] < 0.01:
        xy_position2 = r_max  # Safe default if threshold is never crossed within range
        x_targets3 = [distance[r_max - 1], distance[r_max]]
        y_targets3 = [y_values[r_max - 1], y_values[r_max]]
        y6 = y_targets3[1]
        for i in range(r_max + 1):
            if y_values[i] >= 0.01 > y_values[i + 1]:
                x5, x6 = distance[i], distance[i + 1]
                x_targets3 = [x5, x6]
                y5 = y_values[i]
                y6 = y_values[i + 1]
                xy_position2 = i + 1
                y_targets3 = [y5, y6]
                break
        
        f3 = interp1d(y_targets3, x_targets3, kind='linear', 
                      bounds_error=False, fill_value="extrapolate")
        x_solution3 = f3(0.01)
        x_results.append(x_solution3)
        y_targets.append(0.01)
        
        x_plot_range = distance[0:xy_position2 + 1]
        y_plot_range = y_values[0:xy_position2 + 1]
        f_plot = interp1d(y_plot_range, x_plot_range, kind='linear', 
                         bounds_error=False, fill_value="extrapolate")
        y_fine = np.linspace(y6, 1, 1500)
        x_interp = f_plot(y_fine)
        
        interp_data = {'x_interp': x_interp, 'y_fine': y_fine}
        
    else:
        x_plot_range = distance[0:xy_position1 + 1]
        y_plot_range = y_values[0:xy_position1 + 1]
        f_plot = interp1d(y_plot_range, x_plot_range, kind='linear', 
                         bounds_error=False, fill_value="extrapolate")
        y_fine = np.linspace(y4, 1, 1500)
        x_interp = f_plot(y_fine)
        
        interp_data = {'x_interp': x_interp, 'y_fine': y_fine}
    
    return x_results, y_targets, interp_data


def create_autocorrelation_plots(distance, corr_coeff_mean, corr_coeff_std, 
                                x_results, y_targets, interp_data, 
                                dir_out_path, tif_file_name, layer_count):
    """
    Create and save autocorrelation analysis plots.
    
    Args:
        distance (np.ndarray): Distance array
        corr_coeff_mean (np.ndarray): Mean correlation coefficients
        corr_coeff_std (np.ndarray): Standard deviation of correlations
        x_results (list): Characteristic length values
        y_targets (list): Target correlation values
        interp_data (dict): Interpolation data for zoomed plot
        dir_out_path (str): Output directory path
        tif_file_name (str): Base filename for saving
    """
    # === Save autocorrelation data to .mat file ===

    # Calculate the +/- Std Dev curves
    mean_minus_std = corr_coeff_mean - corr_coeff_std
    mean_plus_std = corr_coeff_mean + corr_coeff_std

    # Concatenate all data into a matrix
    data_matrix = np.hstack([
        distance.reshape(-1, 1),
        corr_coeff_mean.reshape(-1, 1),
        mean_minus_std.reshape(-1, 1),
        mean_plus_std.reshape(-1, 1)
    ])

    output_path_mat = os.path.join(dir_out_path, f"{tif_file_name}_Layer{layer_count}_autocorrelation.mat")

    try:
        sio.savemat(output_path_mat, {f"AC_data_layer{layer_count}": data_matrix,
                    'Headers': ['Distance','Mean_AC', 'Mean_AC_Minus_Std', 'Mean_AC_Plus_Std']})
        print(f"\nSuccessfully saved autocorrelation plots to {output_path_mat}")
    except Exception as e:
        print(f"Error saving data to .mat file: {e}")

    # =======================================


    # Set default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 12
    
    # Main autocorrelation plot
    fig, ax = plt.subplots()
    line1, = ax.plot(range(len(corr_coeff_mean)), corr_coeff_mean, 
                     label='Mean AC', color='blue')
    line1.set_dashes([2, 2, 10, 2])  # Dashed line style
    
    line2, = ax.plot(range(len(corr_coeff_mean)), 
                     [mean - std for mean, std in zip(corr_coeff_mean, corr_coeff_std)],
                     dashes=[4, 2], color='red', label='Plus/Minus 1 Std')
    line3, = ax.plot(range(len(corr_coeff_mean)), 
                     [mean + std for mean, std in zip(corr_coeff_mean, corr_coeff_std)],
                     dashes=[4, 2], color='red', label=None)
    
    # Plot characteristic points
    for x, y in zip(x_results, y_targets):
        ax.plot(x, y, 'ro', markersize=6)
    
    # Customize main plot
    ax.set_xlabel('Autocorrelation length (vx)', 
                  fontdict={'fontsize': 14, 'fontfamily': 'serif', 'fontname': 'Arial'})
    ax.set_ylabel('Autocorrelation (AC)', 
                  fontdict={'fontsize': 14, 'fontfamily': 'serif', 'fontname': 'Arial'})
    ax.set_title('Mean autocorrelation and standard deviation', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend with characteristic lengths
    for i, (x, y_target) in enumerate(zip(x_results, y_targets)):
        if y_target == 0.3679:
            ax.plot([], [], 'ro', markersize=4, label='1/e AC length: %.1f' % x)
        elif y_target == 0.1:
            ax.plot([], [], 'ro', markersize=4, label='10%% AC length: %.1f' % x)
        elif y_target == 0.01:
            ax.plot([], [], 'ro', markersize=4, label='1%% AC length: %.1f' % x)
    
    ax.legend(fontsize=12, frameon=False)
    
    # Zoomed-in logarithmic plot
    fig2, ax2 = plt.subplots(figsize=(4.5, 6))
    
    if 'x_interp' in interp_data and 'y_fine' in interp_data:
        x_interp = interp_data['x_interp']
        y_fine = interp_data['y_fine']
        
        # Plot the main interpolated line
        line4, = ax2.plot(x_interp, np.log10(y_fine), color='blue', linewidth=2.5)
        
        # Plot points and reference lines
        for x, y_target in zip(x_results, y_targets):
            ax2.plot(x, np.log10(y_target), 'ro', markersize=8)
            ax2.hlines(y=np.log10(y_target), xmin=0, xmax=x, 
                      linestyles='--', colors='gray', alpha=0.5, linewidth=2.5)
            ax2.vlines(x=x, ymin=-3.5, ymax=np.log10(y_target), 
                      linestyles='--', colors='gray', alpha=0.5, linewidth=2.5)
        
        # Set axis limits
        x_max = x_results[-1] * 1.1  # Add 10% margin
        ax2.set_xlim(0, x_max)
        ax2.set_ylim(-2.5, 0)
    
    # Customize zoomed plot
    ax2.set_ylabel('Log (AC)', 
                   fontdict={'fontsize': 20, 'fontfamily': 'serif', 'fontname': 'Arial'})
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Style the plot borders
    for spine in ax2.spines.values():
        spine.set_linewidth(1.75)
    
    # Save plots
    save_path1 = os.path.join(dir_out_path, f"{tif_file_name}_Layer{layer_count}_figure3.1.svg")
    fig.savefig(save_path1, dpi=1200, bbox_inches='tight')
    save_path2 = os.path.join(dir_out_path, f"{tif_file_name}_Layer{layer_count}_figure3.2.svg")
    fig2.savefig(save_path2, dpi=1200, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the 3D autocorrelation analysis workflow.

    Workflow is split into four phases:
      Phase 1 - File selection  : choose input TIFF and output directory
      Phase 2 - Load & info     : load image stack and display metadata
      Phase 3 - Parameters      : collect all processing options, preview ROI,
                                  show parameter summary, and ask for confirmation
      Phase 4 - Processing      : apply all operations and save results
                                  (no interactive prompts after confirmation)
    """
    print("3D Image Autocorrelation Analysis Tool")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Phase 1: File selection
    # ------------------------------------------------------------------
    tif_file_path, tif_file_name = select_tiff_file()
    dir_out_path = select_output_directory()

    # ------------------------------------------------------------------
    # Phase 2: Load image stack and display info
    # ------------------------------------------------------------------
    imgs = load_tiff_stack(tif_file_path)
    print(f"\nLoaded '{tif_file_name}' successfully.")
    print(f"  Shape : {imgs.shape}  (layers x rows x cols)")
    print(f"  dtype : {imgs.dtype}")

    # ------------------------------------------------------------------
    # Phase 3: Collect all parameters (no heavy computation yet)
    # ------------------------------------------------------------------
    print("\n--- Processing Options ---")

    # 3a. Optional layer cropping
    imgs, layer_count = crop_layers_interactive(imgs)

    # 3b. Optional spatial ROI selection (with OpenCV preview)
    roi_coords, roi_selected, center_x, center_y, roi_width, roi_height = select_roi_interactive(imgs)

    # 3c. Optional denoising (asked last because it affects dtype display in summary)
    denoise_option = input(
        "Apply denoising? WARNING: converts 16-bit to 8-bit (Enter=no, 1=yes): "
    ).strip() or '0'

    # 3d. Parameter summary and confirmation before heavy computation
    roi_summary = (
        f"{roi_width} x {roi_height} px  @ center ({center_x}, {center_y})"
        if roi_selected
        else "None (full image extent)"
    )
    denoise_summary = "Yes (16-bit will be converted to 8-bit)" if denoise_option == '1' else "No"

    print("\n" + "=" * 52)
    print("  ANALYSIS PARAMETERS — please review before computing")
    print("=" * 52)
    print(f"  Input file   : {tif_file_name}")
    print(f"  Output dir   : {dir_out_path}")
    print(f"  Layers used  : {layer_count}")
    print(f"  Spatial ROI  : {roi_summary}")
    print(f"  Denoising    : {denoise_summary}")
    print("=" * 52)
    input("Press Enter to begin computation  (Ctrl+C to abort) ...")

    # ------------------------------------------------------------------
    # Phase 4: Processing — no interactive prompts from here on
    # ------------------------------------------------------------------

    # Step 4a: Apply ROI cropping and data type conversion
    imgs = apply_roi_cropping(imgs, roi_coords, roi_selected, denoise_option)
    num_imgs, num_rows, num_cols = imgs.shape
    print(f"\nPost-crop dimensions: {num_imgs} layers x {num_rows} rows x {num_cols} cols")

    # Step 4b: Save cropped stack as TIFF (only when an explicit ROI was chosen)
    if roi_selected:
        save_cropped_tiff(imgs, dir_out_path, tif_file_name, layer_count)

    # Step 4c: Save cropping parameters to CSV
    output_path = os.path.join(dir_out_path, f"{tif_file_name}_Layer{layer_count}_cropping_parameters.csv")
    roi_status = "Used" if roi_selected else "Not Used (Full Layer)"
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Value', 'Unit'])
            writer.writerow(['Original Filename', tif_file_name, 'N/A'])
            writer.writerow(['Layers Used in Analysis', layer_count, 'layers'])
            writer.writerow(['ROI Cropping Status', roi_status, 'N/A'])
            if roi_selected:
                writer.writerow(['ROI Center X', center_x, 'pixels'])
                writer.writerow(['ROI Center Y', center_y, 'pixels'])
                writer.writerow(['ROI Width', roi_width, 'pixels'])
                writer.writerow(['ROI Height', roi_height, 'pixels'])
            else:
                writer.writerow(['ROI Center X', 'N/A', 'N/A'])
                writer.writerow(['ROI Center Y', 'N/A', 'N/A'])
                writer.writerow(['ROI Width', 'N/A', 'N/A'])
                writer.writerow(['ROI Height', 'N/A', 'N/A'])
        print(f"Cropping parameters saved to: {output_path}")
    except Exception as e:
        print(f"Error saving parameters to CSV: {e}")

    # Step 4d: Apply denoising (if requested)
    imgs = apply_denoising(imgs, denoise_option)

    # Step 4e: Compute 3D autocorrelation function
    nxcorrcoeff_vol, r = compute_autocorrelation(imgs)

    # Step 4f: Compute radial autocorrelation statistics
    distance, corr_coeff_mean, corr_coeff_std, r_max = compute_radial_autocorrelation(nxcorrcoeff_vol, r)

    # Step 4g: Find characteristic autocorrelation lengths
    x_results, y_targets, interp_data = find_autocorrelation_lengths(corr_coeff_mean, distance, r_max)

    # Step 4h: Generate and save plots
    create_autocorrelation_plots(distance, corr_coeff_mean, corr_coeff_std,
                                 x_results, y_targets, interp_data,
                                 dir_out_path, tif_file_name, layer_count)

    print("\nAnalysis complete!")
    print(f"Results saved to: {dir_out_path}")
    for x, y_target in zip(x_results, y_targets):
        if y_target == 0.3679:
            print(f"  1/e  autocorrelation length : {x:.2f} voxels")
        elif y_target == 0.1:
            print(f"  10%  autocorrelation length : {x:.2f} voxels")
        elif y_target == 0.01:
            print(f"  1%   autocorrelation length : {x:.2f} voxels")


if __name__ == "__main__":
    main()