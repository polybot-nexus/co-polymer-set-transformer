from skimage.color import lab2rgb
import colormath
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np
import pandas as pd

def lab_to_rgb(lab_values):
    """
    Converts a list of L*a*b* values to RGB.
    
    Parameters:
    - lab_values: List of tuples representing L*a*b* values.
    
    Returns:
    - List of tuples representing RGB values.
    """
    rgb_values = []
    for lab in lab_values:
        # Normalize L*a*b* input for lab2rgb
        l, a, b = lab
        l = l / 100
        a = a / 128 + 0.5
        b = b / 128 + 0.5
        lab_normalized = np.array([[[l, a, b]]])
        
        # Convert L*a*b* to RGB
        rgb = lab2rgb(lab_normalized)[0][0]
        # Ensure RGB values are within [0, 1] range
        rgb_clipped = np.clip(rgb, 0, 1)
        rgb_values.append(tuple(rgb_clipped))
        
    return rgb_values


def get_hex_color_single(lab_values):
    hex_values = []
    for labs in lab_values:
        lab = LabColor(lab_l = labs[0], lab_a = labs[1], lab_b = labs[2], observer='2', illuminant='d65')
        rgb = convert_color(lab, sRGBColor)
        rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
        try:
            hex_color = rgb.get_rgb_hex() 
        except: 
            hex_color = 0 
        hex_values.append(hex_color)
    return hex_values


# # Example L*a*b* values
in_house_data = pd.read_csv('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/closeloop_all_past_experiments_new.csv')

# lab_values = list(zip(in_house_data['L'], in_house_data['a'], in_house_data['b']))

lab_values = list(zip(in_house_data['L_exp'], in_house_data['a_exp'], in_house_data['b_exp']))

# # Convert Lab values to RGB
rgb_values = lab_to_rgb(lab_values)
hex_values = get_hex_color_single(lab_values)
in_house_data['rgb_color'] = rgb_values
in_house_data['hex_color'] = hex_values
in_house_data.to_csv('closeloop_all_past_experiments__new_rgb.csv', index=None)
# print(hex_values)