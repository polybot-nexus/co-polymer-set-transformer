from DECIMER import predict_SMILES
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from IPython.display import SVG
from decimer_segmentation import *
from decimer_segmentation import segment_chemical_structures, segment_chemical_structures_from_file, get_mrcnn_results, complete_structure_mask, visualize
import cv2
import numpy as np
from copy import deepcopy
from itertools import cycle
from multiprocessing import Pool
from typing import List, Tuple
from PIL import Image
from matplotlib import pyplot as plt
import math
import easyocr
import pandas as pd

def get_masked_image(image: np.array, mask: np.array) -> np.array:
    """
    This function takes an image and a masks for this image
    (shape: (h, w)) and returns the masked image where only the
    masked area is not completely white and a bounding box of the
    segmented object

    Args:
        image (np.array): image of a page from a scientific publication
        mask (np.array): masks (shape: (h, w, num_masks))

    Returns:
        List[np.array]: segmented chemical structure depictions
        List[int]: bounding box of segmented object
    """
    for channel in range(image.shape[2]):
        image[:, :, channel] = image[:, :, channel] * mask[:, :]
    # Remove unwanted background
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)

    masked_image = np.zeros(image.shape).astype(np.uint8)
    masked_image = visualize.apply_mask(masked_image, mask, [1, 1, 1])
    masked_image = Image.fromarray(masked_image)
    masked_image = masked_image.convert("RGB")
    return np.array(masked_image), bbox

def apply_mask(image: np.array, mask: np.array) -> np.array:
    """
    This function takes an image and a mask for this image (shape: (h, w))
    and returns a segmented chemical structure depictions (np.array)

    Args:
        image (np.array): image of a page from a scientific publication
        masks (np.array): binary mask (shape: (h, w))

    Returns:
        np.array: segmented chemical structure depiction
        Tuple[int]: (y0, x0, y1, x1)
    """
    # TODO: Further cleanup
    im = deepcopy(image)
    for channel in range(image.shape[2]):
        im[:, :, channel] = im[:, :, channel] * mask
    masked_image, bbox = get_masked_image(deepcopy(image), mask)
    x, y, w, h = bbox
    im_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    _, im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Removal of transparent layer and generation of segment
    _, alpha = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    background = dst[y : y + h, x : x + w]
    trans_mask = background[:, :, 3] == 0
    background[trans_mask] = [255, 255, 255, 255]
    segmented_image = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    return segmented_image, (y, x, y + h, x + w)


def apply_masks(image: np.array, masks: np.array):
    """
    This function takes an image and the masks for this image
    (shape: (h, w, num_structures)) and returns a list of segmented
    chemical structure depictions (np.array)

    Args:
        image (np.array): image of a page from a scientific publication
        masks (np.array): masks (shape: (h, w, num_masks))

    Returns:
        List[np.array]: segmented chemical structure depictions
    """
    masks = [masks[:, :, i] for i in range(masks.shape[2])]
    if len(masks) == 0:
        return [], []
    segmented_images_bboxes = map(apply_mask, cycle([image]), masks)
    segmented_images, bboxes = list(zip(*list(segmented_images_bboxes)))
    return segmented_images, bboxes


def structure_image_segment(image_path):
    page = cv2.imread(image_path)
    image = image = cv2.imread(image_path)
    masks, _, _ = get_mrcnn_results(image)
    segments, bboxes = apply_masks(page, masks)
    return segments, bboxes, page

def visualize_structure_boxes(segments, bboxes, page, name_boxes=None):

    decimer_ai_boxes = []
    for box in bboxes[:]:
        y, x, k, l = box
        h=k-y
        w=l-x
        decimer_coords = x, y, w, h
        decimer_ai_boxes.append(decimer_coords)
        cv2.rectangle(page, (x, y), (x + w, y + h), (250, 0, 50), 2)  # green box, 2 pixels thick

        # Load the image
        #image = cv2.imread('/content/images_large_ma-2018-01789f_0001.jpeg')
        # Define the transparency factor.
        alpha = 0.4  # Transparency factor (between 0 and 1; 0 is fully transparent, 1 is fully opaque)

        # Define the light yellow color in BGR
        light_yellow = (50, 250, 250)

   
    if name_boxes !=None:
        ocr_boxes = []
        for box in name_boxes:
                x = box[0][0]
                y = box[0][1]
                w = box[1][0] - x
                h = box[2][1] - y
                ocr_coords = x, y, w, h
                ocr_boxes.append(ocr_coords)
                # Create a fully opaque rectangle on the overlay
                overlay = page.copy()

                cv2.rectangle(overlay, (x, y), (x + w, y + h), light_yellow, cv2.FILLED)

                # Blend the overlay with the original image
                cv2.addWeighted(overlay, alpha, page, 1 - alpha, 0, page)

                # Draw the rectangle border in a different color, e.g., light blue (optional)
                cv2.rectangle(page, (x, y), (x + w, y + h), (255, 255, 100), 2)

    # Display the image with the detected boxes
    return page, decimer_ai_boxes, ocr_boxes


def ocrc_polymer_names(image_path):
    polymer_markers = ['Ph', 'ProDOT', 'DAT', 'EDOT', 'DOT', 'DMP', 'DMOT', 'ECP', 'P', 'AcDOT']
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(image_path) 

    # Collect polymer names along with their coordinates and confidence levels
    polymer_info = []
    for result in ocr_result:
        if any(marker in result[1] for marker in polymer_markers):
            polymer_info.append(result)

    # Print polymer names with their coordinates and confidence levels
    all_info =[]
    for info in polymer_info:
        print(info)
        all_info.append(info)
    name_boxes = [res[0] for res in all_info]
    polymer_info = [res[1] for res in all_info]
    return name_boxes, polymer_info 


def box_center(box):
    x = box[0] + box[2] / 2
    y = box[1] + box[3] / 2
    return (x, y)

def match_boxes(box_list1, box_list2):
    # Prepare the list to hold the indices of matching boxes
    matched_indices = []

    for i, box1 in enumerate(box_list1):
        center1 = box_center(box1)

        min_distance = float('inf')
        match_idx = None
        for j, box2 in enumerate(box_list2):
            center2 = box_center(box2)

            # Calculate Euclidean distance
            distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

            if distance < min_distance:
                min_distance = distance
                match_idx = j

        matched_indices.append(match_idx)

    return matched_indices


def get_smiles_from_image_segment(image_path):
    SMILES = predict_SMILES(image_path)
    return SMILES

def spatial_matching():
    pass


def moltosvg(mol,molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def render_svg(svg):
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace('svg:',''))



image_path = "monomer_structure_example/images_large_ma-2016-01763s_0008.jpeg"
save_path =  "monomer_structure_example"
segments, bboxes, page = structure_image_segment(image_path)

name_boxes, polymer_info  = ocrc_polymer_names(image_path)

modified_page, decimer_ai_boxes, ocr_boxes = visualize_structure_boxes(segments, bboxes, page, name_boxes)
matched_indices = match_boxes(ocr_boxes, decimer_ai_boxes)
print('matched_indices', matched_indices)
smiles_list = []
polymer_name = []
for i, index in enumerate(matched_indices):
    name = polymer_info[i]
    print(f"Polymer {name}")# matches with molecular image box {decimer_ai_boxes[index]}")
    polymer_name.append(name)
    image_array = segments[index]
    image = Image.fromarray(image_array)
    image_path1 = f"{save_path}/{name}.png"
    image.save(image_path1)
    smiles=get_smiles_from_image_segment(image_path1)
    smiles = smiles.replace('R1', '*')
    smiles = smiles.replace('R', '*')    
    smiles=smiles.split('.')[0]
    smiles_list.append(smiles)
    #image.show()

df = pd.DataFrame(data={'polymer_name': polymer_name, 'smiles': smiles_list})
df.to_csv(f"{save_path}/polymer_smiles.csv")
plt.imshow(cv2.cvtColor(modified_page, cv2.COLOR_BGR2RGB))
plt.savefig(f"{save_path}/images_with_boxes.png", dpi=600)
plt.show()