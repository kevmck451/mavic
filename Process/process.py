
from Mavic.mavic import Mavic
from Band_Correction.correction import band_correction
from Radiance_Calibration.radiance import radiance_calibration
from Radiance_Calibration.radiance import dark_current_subtraction
from Radiance_Calibration.radiance import flat_field_correction
from Analysis.vegetation_index import NDVI
from data_filepaths import *

from pathlib import Path


def process_single(file, save_directory=''):
    # Create Mavic Object
    image = Mavic(file)
    # image.dial_in()
    # image.display()

    # Dark Current Subtraction
    # image = dark_current_subtraction(image)
    # image.display()

    # Band_Correction
    # image = band_correction(image)
    # image.display()

    # Flat Field Correction
    # image = flat_field_correction(image)
    # image.display()

    # Radiance_Calibration
    # image = radiance_calibration(image)
    # image.display()

    # Reflectance Calibration
    # image = reflectance_calibration(image)
    # image.display()

    # Georectification
    # image.extract_GPS('tiff')
    # image.export_tiff(save_directory)
    # image.display()

    # Analysis
    # NDVI(image)




if __name__ == '__main__':

    # process_single(WF_68_81)
    process_single(WF_68_2)










