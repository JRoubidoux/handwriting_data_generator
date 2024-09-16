import sys
sys.path.append('/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/src')
from image_generator_mark1 import fontWordOnImage, fontHelper
import json
import yaml
import os
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont


if __name__ == '__main__':
    letters = string.ascii_lowercase

    vocab = []

    '''for letter in letters:
        list_of_chars_1 = []
        list_of_chars_2 = []
        for _ in range(3):
            list_of_chars_1.append(letter)
            list_of_chars_2.append(letter.upper())
        text_1 = ''.join(list_of_chars_1)
        text_2 = ''.join(list_of_chars_2)
        vocab.append(text_1)
        vocab.append(text_2)'''

    for i in range(6):
        list_of_numbers = []
        for _ in range(3):
            list_of_numbers.append(str(i))
        sequence = ''.join(list_of_numbers)
        vocab.append(sequence)

    root_directory_for_fonts = '/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts'
    font_helper = fontHelper(root_directory_for_fonts)
    json_dict = font_helper.get_font_and_weight_dictionary_equal_weights()
    config_file_path = '/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/src/font_word_on_image_config_example.yaml'

    output_dir = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/sandbox/fonts'

    for font_path, weight in json_dict.items():
        font_name = os.path.splitext(os.path.basename(font_path))[0]

        font_dir = os.path.join(output_dir, font_name)

        if not os.path.exists(font_dir):
            os.makedirs(font_dir)

        for text in vocab:

            text_to_write = text
            
            font_size = 60
            font_object = ImageFont.truetype(font_path, font_size)

            background_color = 180

            temp_image = Image.new('RGB', (1, 1), color=(background_color, background_color, background_color))
            temp_draw = ImageDraw.Draw(temp_image)

            text_bbox = temp_draw.textbbox((0, 0), text_to_write, font=font_object)

            image_width = 3*(text_bbox[2] - text_bbox[0])
            image_height = 3*(text_bbox[3] - text_bbox[1])

            image = Image.new('RGB', (image_width+5, image_height+5), color=(background_color, background_color, background_color))
            
            temp_p_array = np.array(image)

            font_color = 0

            text_width = image_width
            text_height = image_height

            text_start_y = round(text_height*0.1)
            text_start_x = round(text_width*0.1)

            draw = ImageDraw.Draw(image)
            draw.text((text_start_x, text_start_y), text_to_write, font=font_object, fill=(font_color, font_color, font_color))

            image_file_path = f"{font_dir}/{text}.png"

            temp_post_drawn_array = np.array(image)

            image.save(image_file_path)