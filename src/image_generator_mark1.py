from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import os

class fontHelper():
    def __init__(self, root_directory: str):
        self.acceptable_file_formats = set(['.otf', '.ttf'])
        self.root_directory = root_directory
        self.set_of_directories_to_exlude = set(['amelia-font',
                                                 'caramellia-font', 
                                                 'celinea-font', 
                                                 'barokah-signature-font', 
                                                 'brock-script-font',
                                                 'discipuli-britannica-font',
                                                 'daniela-font',
                                                 'feasibly-single-line-font',
                                                 'hoarsely-single-line-font',
                                                 'homemade-apple-font',
                                                 'may-queen-font',
                                                 'mrs-saint-delafield-font',
                                                 'otto-font',
                                                 'salonica-font',
                                                 'snoorks-font',
                                                 'shimla-la-font',
                                                 'terrible-cursive-font',
                                                 'writing-something-by-hand-free-font',
                                                 'erratic-cursive-font',
                                                 'gunkid-font',
                                                 'milkmoustachio-font',
                                                 'precious-font',
                                                 'internal-font',
                                                 'monsieur-la-doulaise-font',
                                                 'same-sex-marriage-script-ldo-font'])

    def get_font_files(self, root_directory: str):
        list_of_font_file_paths = []

        for sub_dir in os.listdir(root_directory):
            if sub_dir not in self.set_of_directories_to_exlude:
                new_dir = os.path.join(root_directory, sub_dir)
                for item in os.listdir(new_dir):
                    path_to_item = os.path.join(new_dir, item)

                    if os.path.isfile(path_to_item):
                        ext = os.path.splitext(item)[-1]
                        if ext in self.acceptable_file_formats:
                            list_of_font_file_paths.append(path_to_item)
                            break
        
        return list_of_font_file_paths

    def get_font_and_weight_dictionary_equal_weights(self):
        list_of_font_file_paths = self.get_font_files(self.root_directory)

        number_of_fonts = len(list_of_font_file_paths)

        font_and_weight_dictionary = {}
        for font_path in list_of_font_file_paths:
            font_and_weight_dictionary[font_path] = 1 / number_of_fonts

        return font_and_weight_dictionary


class fontWordOnImage():
    def __init__(self, vocabulary: list, fonts_and_weights: dict, config):
        self.vocabulary = vocabulary # This should be a list of lists. Each inner list should contain a character or word that is part of the sequence. 
        self.font_paths, self.list_of_font_indicies = self.get_font_paths_and_indicies(fonts_and_weights)
        self.length_of_font_index_list_minus_1 = len(self.list_of_font_indicies) - 1
        self.config = config
        self.set_variables_for_font_size()
        self.set_background_colors()
        self.set_font_colors()
        self.font_objects = self.get_font_objects()
        self.set_underline_drawing()
        
    def get_font_paths_and_indicies(self, fonts_and_weights: dict):
        list_of_font_indicies = []
        list_of_font_paths = []

        smallest_weight = min(fonts_and_weights.values())
        sum_of_weights = 0

        for index, (font_path, weight) in enumerate(fonts_and_weights.items()):
            
            sum_of_weights += weight

            list_of_font_paths.append(font_path)
            number_of_times_to_add_the_index = round(weight/smallest_weight)

            for _ in range(number_of_times_to_add_the_index):
                list_of_font_indicies.append(index)

        if sum_of_weights > 1.0 or sum_of_weights < .99:
            print("The sum of the weights is greater or less than 1, this should not be the case.")
            exit()

        return list_of_font_paths, list_of_font_indicies
    
    def get_font_objects(self):
        list_of_font_objects = []
        for font_path in self.font_paths:
            dict_of_font_size_to_ImageFont = {}
            for font_size in range(self.font_size_lower_bound, (self.font_size_upper_bound+1)):
                dict_of_font_size_to_ImageFont[font_size] = ImageFont.truetype(font_path, font_size)
            list_of_font_objects.append(dict_of_font_size_to_ImageFont)
        return list_of_font_objects

    def set_variables_for_font_size(self):
        config_key = 'font_size'
        if self.config[config_key]['static']['bool']:
            self.font_size_static = self.config[config_key]['static']['value']
        elif self.config[config_key]['uniform']['bool']:
            self.font_size_lower_bound = self.config[config_key]['uniform']['lower_bound']
            self.font_size_upper_bound = self.config[config_key]['uniform']['upper_bound']
        elif self.config[config_key]['gaussian']['bool']:
            self.mean_font_size = self.config[config_key]['gaussian']['mean']
            self.standard_deviation_font_size = self.config[config_key]['gaussian']['standard_deviation']
            clip_values_at_number_of_std_deviations = self.config[config_key]['gaussian']['clip_values_at_number_of_std_deviations']
            self.font_size_lower_bound = round(self.mean_font_size - self.standard_deviation_font_size*clip_values_at_number_of_std_deviations)
            self.font_size_upper_bound = round(self.mean_font_size + self.standard_deviation_font_size*clip_values_at_number_of_std_deviations)
        else:
            print("One of these settings for font size must be set to True.")
            exit()

    def set_background_colors(self):
        config_key = 'background_color'
        if self.config[config_key]['static']['bool']:
            self.background_color_static = self.config[config_key]['static']['value']
        elif self.config[config_key]['uniform']['bool']:
            self.background_color_lower_bound = self.config[config_key]['uniform']['lower_bound']
            self.background_color_upper_bound = self.config[config_key]['uniform']['upper_bound']
        elif self.config[config_key]['gaussian']['bool']:
            self.mean_background_color = self.config[config_key]['gaussian']['mean']
            self.standard_deviation_background_color = self.config[config_key]['gaussian']['standard_deviation']
            clip_values_at_number_of_std_deviations = self.config[config_key]['gaussian']['clip_values_at_number_of_std_deviations']
            self.background_color_lower_bound = round(self.mean_background_color - self.standard_deviation_background_color*clip_values_at_number_of_std_deviations)
            self.background_color_upper_bound = round(self.mean_background_color + self.standard_deviation_background_color*clip_values_at_number_of_std_deviations)
        else:
            print("One of these settings for background color must be set to True.")
            exit()

    def set_font_colors(self):
        config_key = 'font_color'
        if self.config[config_key]['static']['bool']:
            self.font_color_static = self.config[config_key]['static']['value']
        elif self.config[config_key]['uniform']['bool']:
            self.font_color_lower_bound = self.config[config_key]['uniform']['lower_bound']
            self.font_color_upper_bound = self.config[config_key]['uniform']['upper_bound']
        elif self.config[config_key]['gaussian']['bool']:
            self.mean_font_color = self.config[config_key]['gaussian']['mean']
            self.standard_deviation_font_color = self.config[config_key]['gaussian']['standard_deviation']
            clip_values_at_number_of_std_deviations = self.config[config_key]['gaussian']['clip_values_at_number_of_std_deviations']
            self.font_color_lower_bound = round(self.mean_font_color - self.standard_deviation_font_color*clip_values_at_number_of_std_deviations)
            self.font_color_upper_bound = round(self.mean_font_color + self.standard_deviation_font_color*clip_values_at_number_of_std_deviations)
        else:
            print("One of these settings for font color must be set to True.")
            exit()

    def set_underline_drawing(self):
        list_of_underline_options = []
        smallest_frequency = min([value for value in self.config['draw_underlines'].values() if value != 0])

        for key, value in self.config['draw_underlines'].items():
            proportion_of_options = round(value / smallest_frequency)
            for i in range(proportion_of_options):
                list_of_underline_options.append(key)
        
        self.list_of_draw_underline_options = list_of_underline_options

    def get_font(self, font_size: int):
        font_index = self.list_of_font_indicies[random.randint(0, self.length_of_font_index_list_minus_1)]
        return self.font_objects[font_index][font_size]

    def get_font_size(self):
        config_key = 'font_size'
        if self.config[config_key]['static']['bool']:
            return self.font_size_static
        elif self.config[config_key]['uniform']['bool']:
            return random.randint(self.font_size_lower_bound, self.font_size_upper_bound)
        elif self.config[config_key]['gaussian']['bool']:
            font_size = round(random.gauss(self.mean_font_size, self.standard_deviation_font_size))
            return max(self.font_size_lower_bound, min(self.font_size_upper_bound, font_size))
        else:
            print("One of these must be set. There is no default.")
            exit()

    def get_background_color(self):
        config_key = 'background_color'
        if self.config[config_key]['static']['bool']:
            return self.background_color_static
        elif self.config[config_key]['uniform']['bool']:
            return random.randint(self.background_color_lower_bound, self.background_color_upper_bound)
        elif self.config[config_key]['gaussian']['bool']:
            background_color = round(random.gauss(self.mean_background_color, self.standard_deviation_background_color))
            return max(self.background_color_lower_bound, min(self.background_color_upper_bound, background_color))
        else:
            print("One of these must be set. There is no default.")
            exit()

    def get_font_color(self):
        config_key = 'font_color'
        if self.config[config_key]['static']['bool']:
            return self.background_color_static
        elif self.config[config_key]['uniform']['bool']:
            return random.randint(self.background_color_lower_bound, self.background_color_upper_bound)
        elif self.config[config_key]['gaussian']['bool']:
            font_color = round(random.gauss(self.mean_font_color, self.standard_deviation_font_color))
            return max(self.font_color_lower_bound, min(self.font_color_upper_bound, font_color))
        else:
            print("One of these must be set. There is no default.")
            exit()

    def draw_underline(self, image, text: str, font_size: int):
        draw_underline_selection = random.choice(self.list_of_draw_underline_options)

        if draw_underline_selection == "draw_no_underline_frequency":
            return image
        else:
            y_start_value_to_draw_line = 0
            line_thickness = font_size // 10

            image_as_array = np.array(image)

            if text == 'blank':
                image_height = image.size[1]
                lower_bound = round(image_height*(0.66))

                if (image_height - line_thickness) < lower_bound:
                    return image
                else:
                    y_start_value_to_draw_line = random.randint(lower_bound, (image_height-line_thickness))

            else:
                array_2d = np.sum(image_as_array, axis=2)
                array_1d = np.sum(array_2d, axis=1)
                density_array = np.zeros_like(array_1d)

                for i in range(len(density_array)):
                    if i == 0:
                        density_array[i] = np.mean(array_1d[0:2])
                    elif i == 1:
                        density_array[i] = np.mean(array_1d[0:3])
                    elif i == (len(density_array)-2):
                        density_array[i] = np.mean(array_1d[(i-2):])
                    elif i == (len(density_array)-3):
                        density_array[i] = np.mean(array_1d[(i-2):])
                    else:
                        density_array[i] = np.mean(array_1d[(i-2):(i+2)])

                mean_value_from_density = np.mean(density_array)

                y_start_value_to_draw_line = (image.size[-1] - 1)

                for i in range((len(density_array)-1), -1, -1):
                    if density_array[i] < mean_value_from_density:
                        y_start_value_to_draw_line = i
                        break

            if draw_underline_selection == "draw_full_underline_frequency":

                image_as_array[y_start_value_to_draw_line:(y_start_value_to_draw_line+line_thickness)][:][:] = np.random.randint(0, 2)
                image_with_line = Image.fromarray(image_as_array)
                return image_with_line

            elif draw_underline_selection == "draw_dotted_underlines_frequency":
                line_color = np.random.randint(0, 20)
                skip_line_frequency = np.random.randint(5,8)

                for skip_number, i in enumerate(range(0, image_as_array.shape[1], skip_line_frequency)):
                    increaser = random.randint((skip_line_frequency-1), (skip_line_frequency+1))
                    if skip_number % 2 == 0:
                        image_as_array[y_start_value_to_draw_line:(y_start_value_to_draw_line+line_thickness), i:i+increaser, :] = line_color
                image_with_line = Image.fromarray(image_as_array)
                return image_with_line
            else:
                print("Check that your config file matches the strings listed here.")
                exit()

    def draw_text_on_image(self, image, text_start_x: int, text_start_y: int, text: str, font_object, font_color: int, background_color: int):
        image, add_to_top, add_to_bottom, add_to_left, add_to_right = self.add_padding_to_image(image, background_color)
        text_start_x = text_start_x + round(add_to_left*.8)
        text_start_y = text_start_y + round(add_to_top*.8)
        draw = ImageDraw.Draw(image)
        draw.text((text_start_x, text_start_y), text, font=font_object, fill=(font_color, font_color, font_color))
        
        return image

    def add_padding_to_image(self, image, background_color: int):
        image_as_array = np.array(image)
        image_height, image_width, _ = image_as_array.shape

        height_to_add = round(image_height * (random.random()))
        add_to_top = random.randint(0, height_to_add)
        add_to_bottom = height_to_add - add_to_top

        width_to_add = round(image_width * (.25)*(random.random()))
        add_to_left = random.randint(0, height_to_add)
        add_to_right = width_to_add - add_to_left

        new_image = Image.fromarray(np.full(((image_height+height_to_add), (image_width+width_to_add), 3), background_color, dtype=np.uint8))

        return new_image, add_to_top, add_to_bottom, add_to_left, add_to_right

    def render_word_on_image_and_text_label(self, image_width_multiplier: float, image_height_multiplier: float, start_text_x_fraction_of_width: float, start_text_y_fraction_of_height: float):
        # Assume that the words in the given image will all be the same size and all have the same color. Also that they all have the same font. Font can change from image to image. 
        text = random.choice(self.vocabulary)

        if text == 'blank':
            text_to_write = ''.join([' ' for _ in range(random.randint(4, 20))])
        else:
            text_to_write = text
        
        font_size = self.get_font_size()
        font_object = self.get_font(font_size)
        # print(font_object.path)

        background_color = self.get_background_color()

        temp_image = Image.new('RGB', (1, 1), color=(background_color, background_color, background_color))
        temp_draw = ImageDraw.Draw(temp_image)

        text_bbox = temp_draw.textbbox((0, 0), text_to_write, font=font_object)

        text_width = (text_bbox[2] - text_bbox[0])*((len(text_to_write)+1)/(len(text_to_write)))

        if text == 'blank':
            num_spaces = len(text_to_write)
            text_height = int((random.randint(1, (num_spaces-round(num_spaces*(2/3))))/num_spaces)*text_width)
        else:
            text_height = (text_bbox[3] - text_bbox[1])

        image_width = round((text_width)*image_width_multiplier)
        image_height = round((text_height)*image_height_multiplier)

        image = Image.new('RGB', (image_width, image_height), color=(background_color, background_color, background_color))
        
        text_start_y = round(text_height*start_text_y_fraction_of_height)
        text_start_x = round(text_width*start_text_x_fraction_of_width)

        font_color = self.get_font_color()

        image = self.draw_text_on_image(image, text_start_x, text_start_y, text_to_write, font_object, font_color, background_color)

        image = self.draw_underline(image, text, font_size)

        if text == 'blank':
            return image, ''
        else:
            return image, text