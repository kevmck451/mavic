# File to find radiance calibration values

from Band_Correction.correction import band_correction
from Radiance_Calibration.radiance import *
from Radiance_Calibration.radiance_calibration import *
from data_filepaths import *


import matplotlib.pyplot as plt
import numpy as np
import csv


from scipy import stats
from pathlib import Path
# Max Pixel Value is 3947

# Base Process for Data
base_path = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 Imaging/Data/MapIR/Radiance Calibration')

# Brightness Dial In: make sure values are in range
def dial_in_graphs():
    filepath = base_path / 'Brightness Dial In/2.RAW'
    # filepath = base_path / 'Brightness Dial In/2.RAW'
    # filepath = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 Imaging/MapIR/_Radiance Calibration')
    image = MapIR(filepath).dial_in()

# Dark Current Graphs
def dark_current_graphs():
    filepath = base_path / 'Dark'
    sorted_files = sorted(Path(filepath).iterdir())

    image = None
    stats = []
    name = []

    for i, file in enumerate(sorted_files):
        if file.suffix == '.RAW':
            image = MapIR_Radiance(file)
            stats.append(image.dark_current_subtraction(display=False))
            name.append(f'File: {image.path.name} / Stage: {image.stage}')

    rgb_stack = np.zeros((image.img_y, image.img_x, 3), 'uint8')
    text_1 = f'R: {stats[0][6]} | G: {stats[0][7]} | N: {stats[0][8]} (pixels above threshold)'
    text_2 = f'R: {stats[1][6]} | G: {stats[1][7]} | N: {stats[1][8]} (pixels above threshold)'

    plt.figure(figsize=(14, 6))
    plt.suptitle('Dark Current Max Values')
    rows, cols = 1, 2
    s = 20

    plt.subplot(rows, cols, 1)
    plt.imshow(rgb_stack)
    plt.scatter(stats[0][0][1], stats[0][0][0], color='red', label=stats[0][3], s=s)
    plt.scatter(stats[0][1][1], stats[0][1][0], color='green', label=stats[0][4], s=s)
    plt.scatter(stats[0][2][1], stats[0][2][0], color='blue', label=stats[0][5], s=s)
    plt.title(f'{name[0]}: {text_1}')
    plt.axis(False)
    plt.legend(loc='upper right')

    plt.subplot(rows, cols, 2)
    plt.imshow(rgb_stack)
    plt.scatter(stats[1][0][1], stats[1][0][0], color='red', label=stats[1][3], s=s)
    plt.scatter(stats[1][1][1], stats[1][1][0], color='green', label=stats[1][4], s=s)
    plt.scatter(stats[1][2][1], stats[1][2][0], color='blue', label=stats[1][5], s=s)
    plt.title(f'{name[1]}: {text_2}')
    plt.axis(False)
    plt.legend(loc='upper right')

    plt.tight_layout(pad=1)
    plt.show()

# Flat Field Hori vs Vert Graphs
def flat_field_Hori_vs_Vert():
    filepath = base_path / 'Experiments/Exp 1/raw'
    sorted_files = sorted(Path(filepath).iterdir())

    for i, file in enumerate(sorted_files):
        if file.suffix == '.RAW':
            image = MapIR_Radiance(file)
            # image = radiance_calibration(image)
            image.flat_field_hori_vert()

# Flat Field Hori vs Vert OG / Corr Comp Graphs
def flat_field_HV_Comp():
    filepath = base_path / 'Experiments/Exp 1/raw'
    sorted_files = sorted(Path(filepath).iterdir())

    y_mid = 1500
    x_mid = 2000

    for i, file in enumerate(sorted_files):
        if file.suffix == '.RAW':
            image = MapIR(file)
            image_rad = radiance_calibration(image)

            mean_r = np.mean(image.data[:, :, 0])
            mean_g = np.mean(image.data[:, :, 1])
            mean_n = np.mean(image.data[:, :, 2])

            x = [x for x in range(0, 4000)]
            plt.figure(figsize=(14, 8))
            plt.suptitle(f'{image.path.name} Hori vs Vert')
            rows, cols = 2, 2

            plt.subplot(rows, cols, 1)
            plt.scatter(x, image.data[y_mid, :, 0], color='red', s=1)
            plt.axhline(y=mean_r, color='r', linestyle='--')
            plt.scatter(x, image.data[y_mid, :, 1], color='green', s=1)
            plt.axhline(y=mean_g, color='g', linestyle='--')
            plt.scatter(x, image.data[y_mid, :, 2], color='blue', s=1)
            plt.axhline(y=mean_n, color='b', linestyle='--')
            plt.title(f'Horizontal-Original')
            plt.ylim((0, 4096))

            y = [x for x in range(0, 3000)]
            plt.subplot(rows, cols, 2)
            plt.scatter(y, image.data[:, x_mid, 0], color='red', s=1, label='Red Band')
            plt.axhline(y=mean_r, color='r', linestyle='--')
            plt.scatter(y, image.data[:, x_mid, 1], color='green', s=1, label='Green Band')
            plt.axhline(y=mean_g, color='g', linestyle='--')
            plt.scatter(y, image.data[:, x_mid, 2], color='blue', s=1, label='NIR Band')
            plt.axhline(y=mean_n, color='b', linestyle='--')
            plt.title(f'Vertical-Original')
            plt.ylim((0, 4096))

            plt.subplot(rows, cols, 3)
            plt.scatter(x, image_rad.data[y_mid, :, 0], color='red', s=1)
            plt.axhline(y=mean_r, color='r', linestyle='--')
            plt.scatter(x, image_rad.data[y_mid, :, 1], color='green', s=1)
            plt.axhline(y=mean_g, color='g', linestyle='--')
            plt.scatter(x, image_rad.data[y_mid, :, 2], color='blue', s=1)
            plt.axhline(y=mean_n, color='b', linestyle='--')
            plt.title(f'Horizontal-Flat Field')
            plt.ylim((0, 4096))

            y = [x for x in range(0, 3000)]
            plt.subplot(rows, cols, 4)
            plt.scatter(y, image_rad.data[:, x_mid, 0], color='red', s=1, label='Red Band')
            plt.axhline(y=mean_r, color='r', linestyle='--')
            plt.scatter(y, image_rad.data[:, x_mid, 1], color='green', s=1, label='Green Band')
            plt.axhline(y=mean_g, color='g', linestyle='--')
            plt.scatter(y, image_rad.data[:, x_mid, 2], color='blue', s=1, label='NIR Band')
            plt.axhline(y=mean_n, color='b', linestyle='--')
            plt.title(f'Vertical-Flat Field')
            plt.ylim((0, 4096))

            plt.show()

# Heat Map Display of Image for FF Correction
def flat_field_correction_graphs():
    filepath = base_path / 'Experiments/Exp 1/raw'
    sorted_files = sorted(Path(filepath).iterdir())

    for i, file in enumerate(sorted_files):
        if i > 0: continue
        if file.suffix == '.RAW':
            image = MapIR_Radiance(file)
            image.flat_field_correction()

# Test FF Corr on actual image
def flat_field_correction_test():

    file = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 Imaging/Data/MapIR/AC Summer 23/Wheat Field/6-8/raw/081.RAW'
    # file = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 Imaging/Data/MapIR/AC Summer 23/Wheat Field/6-8/raw/177.RAW'
    image = MapIR(file)
    # image.display()
    image = dark_current_subtraction(image)
    # image.display()
    image = band_correction(image)
    image.display()

    red_ff = np.load('flat_field/ff_cor_matrix_red.npy')
    green_ff = np.load('flat_field/ff_cor_matrix_green.npy')
    nir_ff = np.load('flat_field/ff_cor_matrix_nir.npy')

    image.data[:, :, 0] = image.data[:, :, 0] / red_ff
    image.data[:, :, 1] = image.data[:, :, 1] / green_ff
    image.data[:, :, 2] = image.data[:, :, 2] / nir_ff

    print(image.stage)
    # image.data = ((image.data / np.max(image.data)) * image.max_raw_pixel_value).astype('uint16')

    image.display()

# Labsphere Value Graphs
def radiance_plot(directory):
    amp_values_exp1 = {0: 557.2368, 1: 534.7504, 2: 503.9878, 3: 468.3653, 4: 429.8584,
                       5: 390.1801, 6: 349.9428, 7: 308.6331, 8: 265.2019, 9: 220.3712}

    amp_values_exp2 = {0: 527.881, 1: 508.7342, 2: 479.506, 3: 445.5909, 4: 408.9898,
                       5: 371.2294, 6: 332.9773, 7: 293.6617, 8: 252.3409, 9: 209.6933}

    path = Path(directory)
    name = path.parent.name
    if path.parent.name == 'Exp 1': amp_values = amp_values_exp1
    else: amp_values = amp_values_exp2


    sorted_files = sorted(Path(directory).iterdir())

    x_values = []
    R_values = []
    G_values = []
    N_values = []

    for i, file in enumerate(sorted_files):
        if file.suffix == '.RAW':
            image = MapIR_Radiance(file)
            image = dark_current_subtraction(image)
            image = band_correction(image)
            image = flat_field_correction(image)
            # image.display()

            R, G, N = image.radiance_values_center()
            # R, G, N = image.radiance_plot()
            x = int(image.path.stem)
            x = amp_values.get(x)
            x_values.append(x)
            R_values.append(R)
            G_values.append(G)
            N_values.append(N)

    # First, perform linear regression to find the best fit lines
    r_slope, r_intercept, _, _, _ = stats.linregress(x_values, R_values)
    g_slope, g_intercept, _, _, _ = stats.linregress(x_values, G_values)
    n_slope, n_intercept, _, _, _ = stats.linregress(x_values, N_values)

    # Generate x values for regression lines
    x_line = np.linspace(0, 600, 100)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle(name)

    # First subplot
    ax1.set_title('Radiance Plot')
    ax1.plot(x_values, R_values, color='red', marker='s', label='Red')
    ax1.plot(x_values, G_values, color='green', marker='s', label='Green')
    ax1.plot(x_values, N_values, color='blue', marker='s', label='NIR')
    ax1.set_xlabel('LapSphere Amp Values')
    ax1.set_ylabel('Digital Numbers')
    ax1.legend()
    ax1.set_xticks([x for x in amp_values.values()])

    # Second subplot
    ax2.set_title('Linear Regression Lines')
    ax2.plot(x_line, r_slope * x_line + r_intercept, color='red', label='Red Fit')
    ax2.plot(x_line, g_slope * x_line + g_intercept, color='green', label='Green Fit')
    ax2.plot(x_line, n_slope * x_line + n_intercept, color='blue', label='NIR Fit')
    ax2.set_xlabel('LapSphere Amp Values')
    ax2.set_ylabel('Estimated Digital Numbers')
    ax2.axhline(y=0, color='black', linestyle='dotted')
    ax2.axvline(x=0, color='black', linestyle='dotted')
    ax2.legend()

    # Automatically adjust the layout
    plt.tight_layout(pad=1)
    plt.show()

# Function to Retrieve Labsphere values
def get_labsphere_values(filepath):
    lab_bands = []
    lab_rad_values = []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            lab_bands.append(int(row[0]))
            lab_rad_values.append(float(row[1]))

    lab_bands = np.array(lab_bands)
    lab_rad_values = np.array(lab_rad_values)

    np.save(f'labsphere/labsphere_bands.npy', lab_bands)
    np.save(f'labsphere/labsphere_rad_vals.npy', lab_rad_values)

# Function to dot product correction bands
def filter_wavelengths_graph():

    mbands = np.load(MC_Test_Bands)
    mreds = np.load(MC_Test_Reds_Corr)
    mgreens = np.load(MC_Test_Greens_Corr)
    mnir = np.load(MC_Test_NIR_Corr)

    lab_bands = np.load(labsphere_bands)
    lab_rad_values = np.load(labsphere_rad_values)

    # for band, r, g, n in zip(mbands, mreds, mgreens, mnir):
    #     print(f'B: {band} \t |\t R: {r} \t |\t G: {g} \t |\t N: {n}')
    #
    # for band, val in zip(lab_bands, lab_rad_values):
    #     print(f'B: {band} \t |\t V: {val}')

    plt.figure(figsize=(14, 6))
    plt.suptitle('MapIR / Labsphere Filter Wavelengths', fontsize=20)
    row, col = 1, 2

    plt.subplot(row, col, 1)
    plt.plot(mbands, mreds, color='r', linewidth=2, label='Red Values')
    plt.plot(mbands, mgreens, color='g', linewidth=2, label='Green Values')
    plt.plot(mbands, mnir, color='b', linewidth=2, label='NIR Values')
    # plt.axhline(y=0, color='black', linestyle='dotted')
    plt.axvline(x=550, color='black', linestyle='dotted')
    plt.axvline(x=660, color='black', linestyle='dotted')
    plt.axvline(x=850, color='black', linestyle='dotted')
    plt.xlabel('Bands')
    plt.ylabel('Digital Numbers')
    plt.xticks([x for x in range(500, 900, 25)])
    plt.title('MapIR Monochromator Test: RAW', fontsize=12)
    plt.legend(loc='upper right')

    plt.subplot(row, col, 2)
    plt.plot(lab_bands, lab_rad_values, color='black', linewidth=2)
    # plt.axhline(y=0, color='black', linestyle='dotted')
    plt.axvline(x=550, color='black', linestyle='dotted')
    plt.axvline(x=660, color='black', linestyle='dotted')
    plt.axvline(x=850, color='black', linestyle='dotted')
    plt.xlabel('Bands')
    plt.ylabel('Radiance Values')
    plt.xticks([x for x in range(500, 900, 25)])
    plt.title('Labsphere Radiance Values', fontsize=12)

    plt.tight_layout(pad=1)
    plt.show()





if __name__ == '__main__':
    # generate_dark_current_values(base_path / 'Dark')
    # generate_flat_field_correction(base_path / 'Experiments/Exp 1/raw/0.RAW', save=True)

    # dial_in_graphs()
    # dark_current_graphs()

    # flat_field_Hori_vs_Vert()
    # flat_field_HV_Comp()
    # flat_field_correction_graphs()
    # flat_field_correction_test()

    # get_labsphere_values(labsphere_doc)
    # filter_wavelengths_graph()
    filter_wavelengths()


    # generate_radiance_equation_values(base_path / 'Experiments/Exp 1/raw')

    # radiance_plot(labsphere_experiment_1_raw)
    # radiance_plot(labsphere_experiment_2_raw)

    # radiance_calibration_testing()
