from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

class fontWordOnImage():
    def __init__(self, vocabulary: list, fonts_and_weights: dict, size_images_differently_uniform: bool, size_words_differently_uniform: bool):
        self.vocabulary = vocabulary # This should be a list of lists. Each inner list should contain a character or word that is part of the sequence. 
        self.font_paths, self.list_of_font_indicies = self.get_font_paths_and_indicies(fonts_and_weights)
        self.length_of_font_index_list_minus_1 = len(self.list_of_font_indicies) - 1
        self.size_images_differently_uniform = size_images_differently_uniform
        self.config = {}
        self.size_words_differently_uniform = size_words_differently_uniform
        self.lower_bound_font_size, self.upper_bound_font_size = self.get_lower_and_upper_bounds()
        self.font_objects = self.get_font_objects()
        
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

        if sum_of_weights > 1.0:
            print("The sum of the weights is greater than 1, this should not be the case.")
            exit()

        return list_of_font_paths, list_of_font_indicies
    
    def get_font_objects(self):
        list_of_font_objects = []
        for font_path in self.font_paths:
            dict_of_font_size_to_ImageFont = {}
            for font_size in range(self.lower_bound_font_size, (self.upper_bound_font_size+1)):
                dict_of_font_size_to_ImageFont[font_size] = ImageFont.truetype(font_path, font_size)
            list_of_font_objects.append(dict_of_font_size_to_ImageFont)
        return list_of_font_objects

    def get_lower_and_upper_bounds(self):
        if self.config['size_words']['static_size']['bool']:
            font_size_lower_bound = self.config['size_words']['static_size']['font_size']
            font_size_upper_bound = font_size_lower_bound
            return font_size_lower_bound, font_size_upper_bound

        elif self.config['size_words']['uniformly']['bool']:
            font_size_lower_bound = self.config['size_words']['uniformly']['font_size_lower_bound']
            font_size_upper_bound = self.config['size_words']['uniformly']['font_size_upper_bound']
            return font_size_lower_bound, font_size_upper_bound

        elif self.config['size_words']['gaussian']['bool']:
            mean = self.config['size_words']['gaussian']['mean']
            standard_deviation = self.config['size_words']['gaussian']['standard_deviation']
            clip_values_at_number_of_std_deviations = self.config['size_words']['gaussian']['clip_values_at_number_of_std_deviations']
            font_size_lower_bound = round(mean - standard_deviation*clip_values_at_number_of_std_deviations)
            font_size_upper_bound = round(mean + standard_deviation*clip_values_at_number_of_std_deviations)
            return font_size_lower_bound, font_size_upper_bound
        else:
            print("One of these settings for font size must be set to True.")
            exit()

    def get_font(self, font_size: int):
        font_index = self.list_of_font_indicies[random.randint(0, self.length_of_font_index_list_minus_1)]
        return self.font_objects[font_index][font_size]

    def draw_text_on_image(self, image, text_start_x: int, text_start_y: int, text: str, font_object, font_color: int, pad_left: int, pad_top: int, pad_right: int, pad_bottom: int):
        draw = ImageDraw.Draw(image)
        draw.text((text_start_x, text_start_y), text, font=font_object, fill=(font_color, font_color, font_color))
        
        image_as_array = np.array(image)
        height, width = image_as_array.shape[:2]
        top_of_text = 0
        bottom_of_text = height-1
        left_of_text = 0
        right_of_text = width-1

        for i in range(height):
            row = image_as_array[i, :, :]
            if np.any(row == font_color):
                top_of_text = i
                break
        for i in range((height-1), -1, -1):
            row = image_as_array[i, :, :]
            if np.any(row == font_color):
                bottom_of_text = i
                break
        for i in range(width):
            column = image_as_array[:, i, :]
            if np.any(column == font_color):
                left_of_text = i
                break
        for i in range((width-1), -1, -1):
            column = image_as_array[:, i, :]
            if np.any(column == font_color):
                right_of_text = i
                break

        if left_of_text > (pad_left-1):
            left_of_text -= pad_left
        if top_of_text > (pad_top-1):
            top_of_text -= pad_top
        if right_of_text < (width-pad_right):
            right_of_text += pad_right
        if bottom_of_text < (height-pad_bottom):
            bottom_of_text += pad_bottom

        array_of_interest = image_as_array[top_of_text:bottom_of_text, left_of_text:right_of_text, :]

        return Image.fromarray(array_of_interest)

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

    def sample_pareto_between_0_and_1(alpha, size=1):
        samples = np.random.pareto(alpha, size=size)
        return 1 / (samples + 1)

    def render_word_on_image_and_text_label(self):
        text = random.choice(self.vocabulary)

        if text == 'blank':
            text_to_write = ''.join([' ' for _ in range(random.randint(4, 20))])
        else:
            text_to_write = text
        
        font_size = self.get_font_size()
        font_object = self.get_font(font_size)

        background_color = self.get_background_color()

        temp_image = Image.new('RGB', (1, 1), color=(background_color, background_color, background_color))
        temp_draw = ImageDraw.Draw(temp_image)

        text_bbox = temp_draw.textbbox((0, 0), text_to_write, font=font_object)

        image_width = 5*(text_bbox[2] - text_bbox[0])
        image_height = 5*(text_bbox[3] - text_bbox[1])

        text_width = image_width

        if text == 'blank':
            num_spaces = len(text_to_write)
            text_height = int((random.randint(1, (num_spaces-round(num_spaces*(2/3))))/num_spaces)*text_width)
        else:
            text_height = (text_bbox[3] - text_bbox[1])

        image = Image.new('RGB', (image_width, image_height), color=(background_color, background_color, background_color))
        
        text_start_y = round(text_height*0.2)
        text_start_x = round(text_width*0.2)

        font_color = self.get_font_color()

        image = self.draw_text_on_image(image, text_start_x, text_start_y, text_to_write, font_object, font_color, background_color)

        image = self.draw_underline(image, text, font_size)

        if text == 'blank':
            return image, ''
        else:
            return image, text



class getNumberForParameter():
    def __init__(self):
        pass

    def get_static_number(self, static_number):
        '''
        This function returns the same font size every time.
        '''
        return static_number
    
    def get_number_randomly_uniform(self, lower_bound, upper_bound):
        '''
        This function returns a font size randomly. The font size is sampled uniformly in the range lower_bound and upper_bound.
        '''
        return random.randint(lower_bound, upper_bound)
    
    def get_number_randomly_gaussian(self, mean, standard_deviation, lower_bound_for_font, upper_bound_for_font):
        '''
        This function returns a font size randomly. The font size is sampled from a normal distribution with mean and standard deviation defined by user.
        The user also defines the number of standard deviations from the mean we should clip our font sizes at so that we don't go beyond a reasonable range.
        '''
        font_size = round(random.gauss(mean, standard_deviation))
        return max(lower_bound_for_font, min(upper_bound_for_font, font_size))


class getFontSize(getNumberForParameter):
    def __init__(self, config: dict):
        super().__init__()

        if config['size_words']['static_size']['bool']:
            self.static_font_size = config['size_words']['static_size']['font_size']
            self.get_font = self.get_static_font_size

        elif config['size_words']['static_size']['bool']:
            self.lower_bound_for_font = config['size_words']['uniformly']['font_size_lower_bound']
            self.upper_bound_for_font = config['size_words']['uniformly']['font_size_upper_bound']
            self.get_font = self.get_random_font_size_uniform

        elif config['size_words']['gaussian']['bool']:
            self.mean = config['size_words']['gaussian']['mean']
            self.standard_deviation = config['size_words']['gaussian']['standard_deviation']
            self.clip_values_at_number_of_std_deviations = config['size_words']['gaussian']['clip_values_at_number_of_std_deviations']
            self.lower_bound_for_font = round(self.mean - self.standard_deviation*self.clip_values_at_number_of_std_deviations) 
            self.upper_bound_for_font = round(self.mean + self.standard_deviation*self.clip_values_at_number_of_std_deviations)
            self.get_font = self.get_random_font_size_gaussian

        else:
            print("one of these should be set to true.")
            exit()

    def get_next_font_size(self):
        '''
        This is the main function of this class. It returns a font size when called.
        '''
        return self.get_font()

    def get_static_font_size(self):
        return self.get_static_number(self.static_font_size)
    
    def get_random_font_size_uniform(self):
        return self.get_number_randomly_uniform(self.lower_bound_for_font, self.upper_bound_for_font)

    def get_random_font_size_gaussian(self):
        return self.get_number_randomly_gaussian(self.mean, self.standard_deviation, self.lower_bound_for_font, self.upper_bound_for_font)


class getColor(getNumberForParameter):
    def __init__(self, config: dict):
        super().__init__()

        if config['size_words']['static_size']['bool']:
            self.static_font_size = config['size_words']['static_size']['font_size']
            self.get_font = self.get_static_font_size

        elif config['size_words']['static_size']['bool']:
            self.lower_bound_for_font = config['size_words']['uniformly']['font_size_lower_bound']
            self.upper_bound_for_font = config['size_words']['uniformly']['font_size_upper_bound']
            self.get_font = self.get_random_font_size_uniform

        elif config['size_words']['gaussian']['bool']:
            self.mean = config['size_words']['gaussian']['mean']
            self.standard_deviation = config['size_words']['gaussian']['standard_deviation']
            self.clip_values_at_number_of_std_deviations = config['size_words']['gaussian']['clip_values_at_number_of_std_deviations']
            self.lower_bound_for_font = round(self.mean - self.standard_deviation*self.clip_values_at_number_of_std_deviations) 
            self.upper_bound_for_font = round(self.mean + self.standard_deviation*self.clip_values_at_number_of_std_deviations)
            self.get_font = self.get_random_font_size_gaussian

        else:
            print("one of these should be set to true.")
            exit()

    def get_next_font_size(self):
        '''
        This is the main function of this class. It returns a font size when called.
        '''
        return self.get_font()

    def get_static_font_size(self):
        return self.get_static_number(self.static_font_size)
    
    def get_random_font_size_uniform(self):
        return self.get_number_randomly_uniform(self.lower_bound_for_font, self.upper_bound_for_font)

    def get_random_font_size_gaussian(self):
        return self.get_number_randomly_gaussian(self.mean, self.standard_deviation, self.lower_bound_for_font, self.upper_bound_for_font)
