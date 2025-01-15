import os
import PIL.Image as Image
import sys
import albumentations as A
import numpy as np

sys.path.append(r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\src\mark_2")

import data_generator as dg
from custom_transforms import ConvertDataType, LightenOrDarkenImage, lightenOrDarkenPartsOfWord

if __name__ == "__main__":
    path_to_config = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\tests\jacob_and_joe_configs\joe\one_row_joenelson\paris_data_generator_name_config.yaml"

    config = dg.configLoader(path_to_config).load_config()

    background_color_manager = dg.backgroundColorManager(config, config_key="background_color", number_lower_bound_limit= 0, number_upper_bound_limit= 255)
    font_color_manager = dg.fontColorManager(config, config_key="font_color", number_lower_bound_limit= 0, number_upper_bound_limit= 255)
    font_size_manager = dg.fontSizeManager(config, config_key="font_size", number_lower_bound_limit= 40, number_upper_bound_limit= 120)

    font_color_lower_bound = font_color_manager.get_lower_bound()
    font_color_upper_bound = font_color_manager.get_upper_bound()

    base_image_transforms = None
    word_image_same_transforms = None 
    word_image_different_transforms = [ConvertDataType(dtype=np.uint8, p=1.0)]# A.Affine(shear={"x":(-25, 25), "y":0}, rotate=(0.5), p=0.8, always_apply=False, fit_output=True, cval=255), A.Compose([A.Downscale(scale_range=(0.95, 0.95), p=1), A.GaussNoise(var_limit=(15, 20), per_channel=False, p=1.0), A.GlassBlur(sigma=0.001, max_delta=1, iterations=1, p=1), A.Morphological(p=1, scale=(2, 3), operation='erosion'), A.GaussianBlur(blur_limit=(5, 5), p=1.0), A.Morphological(p=1, scale=(2, 3), operation='dilation')], p=0.8), A.ElasticTransform(p=0.8, always_apply=False), A.GaussNoise(var_limit=(20, 40), per_channel=False, p=0.8, always_apply=False)]
    merged_image_transforms = [A.Rotate(limit=1.3, p=1.0)]

    merge_word_images_on_base_image = dg.mergeWordImagesOnBaseImage(base_image_transforms, word_image_same_transforms, word_image_different_transforms, merged_image_transforms, config["image_generation"]["base_image"], background_color_manager, font_color_manager, font_size_manager)

    merged_images_output_path = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\tests\jacob_and_joe_configs\joe\one_row_joenelson\merged_word_images"
    if not os.path.exists(merged_images_output_path):
        os.makedirs(merged_images_output_path)

    for i in range(100):
        print("index", i)
        result, text = merge_word_images_on_base_image.get_base_image_merged_with_word_images(True)

        result_as_PIL = Image.fromarray(result)

        image_name = f"{i}.png"
        image_path = os.path.join(merged_images_output_path, image_name)
        result_as_PIL.save(image_path)