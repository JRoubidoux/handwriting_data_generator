import os
import PIL.Image as Image
import sys
import albumentations as A
import numpy as np

# Input the path to the data generator here - ie: "grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/main"
new_dir = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator"
os.chdir(new_dir)

# Append the path to the source directory - ie: "grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/main/RLL_handwriting_data_generator/src"
sys.path.append(r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\src\mark_2")
import data_generator as dg
from custom_transforms import ConvertDataType, LightenOrDarkenImage, lightenOrDarkenPartsOfWord


if __name__ == "__main__":
    path_to_config = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\tutorial\iowa_data_generator_occupation_config_example.yaml"

    config = dg.configLoader(path_to_config).load_config()

    background_color_manager = dg.backgroundColorManager(config, config_key="background_color", number_lower_bound_limit= 0, number_upper_bound_limit= 255)
    font_color_manager = dg.fontColorManager(config, config_key="font_color", number_lower_bound_limit= 0, number_upper_bound_limit= 255)
    font_size_manager = dg.fontSizeManager(config, config_key="font_size", number_lower_bound_limit= 40, number_upper_bound_limit= 120)

    font_color_lower_bound = font_color_manager.get_lower_bound()
    font_color_upper_bound = font_color_manager.get_upper_bound()

    base_image_transforms = [A.GaussNoise(var_limit=(5, 10), per_channel=False, p=0.7, always_apply=False)]
    word_image_same_transforms = None 
    word_image_different_transforms = [ConvertDataType(dtype=np.uint8, p=1.0), lightenOrDarkenPartsOfWord(font_color_lower_bound, font_color_upper_bound, 0.9, 0.9, (2, 3), (2, 4)), A.Affine(shear={"x":(-25, 25), "y":0}, rotate=(0.5), p=0.8, always_apply=False, fit_output=True, cval=255), A.Compose([A.Downscale(scale_range=(0.95, 0.95), p=1), A.GaussNoise(var_limit=(15, 20), per_channel=False, p=1.0), A.GlassBlur(sigma=0.001, max_delta=1, iterations=1, p=1), A.Morphological(p=1, scale=(2, 3), operation='erosion'), A.GaussianBlur(blur_limit=(5, 5), p=1.0), A.Morphological(p=1, scale=(2, 3), operation='dilation')], p=0.8), A.ElasticTransform(p=0.8, always_apply=False), A.GaussNoise(var_limit=(20, 40), per_channel=False, p=0.8, always_apply=False)]
    merged_image_transforms = [A.ColorJitter(brightness=(0.8, 1.1), contrast=(0.9999, 1.0001), saturation=(0.9999, 1.0001), hue=(-0.0001, 0.00001), p=0.8, always_apply=False), A.ImageCompression(quality_range=(50, 50), p=0.8, always_apply=False), LightenOrDarkenImage(p=0.8, always_apply=False), A.GaussNoise(var_limit=(30, 50), per_channel=False, p=0.8, always_apply=False)]

    merge_word_images_on_base_image = dg.mergeWordImagesOnBaseImage(base_image_transforms, word_image_same_transforms, word_image_different_transforms, merged_image_transforms, config["image_generation"]["base_image"], background_color_manager, font_color_manager, font_size_manager)

    merged_images_output_path = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\sandbox\merged_word_images_for_tutorial"
    if not os.path.exists(merged_images_output_path):
        os.makedirs(merged_images_output_path)

    for i in range(10000):
        result, text = merge_word_images_on_base_image.get_base_image_merged_with_word_images(True)

        if i % 100 == 0:
            print("index", i)
            print(text)
            print()
            result_as_PIL = Image.fromarray(result)


        # image_name = f"{i}.png"
        # image_path = os.path.join(merged_images_output_path, image_name)
        # result_as_PIL.save(image_path)