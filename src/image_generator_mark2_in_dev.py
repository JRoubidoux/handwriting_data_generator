from PIL import Image, ImageDraw, ImageFont
import random

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

    def render_word_on_image_and_text_label(self):
        #TODO This is where you leave off J. Note that you are trying to see if you can create an Image object that is close to the size of the resulting image after adding 
        # text to it. Can I find the right maximum x and y coordinates such that I make this image. 
        text_label_list = random.choice(self.vocabulary)

        max_x_val = 0
        max_y_val = 0

        temp_x_val = 0

        font_sizes = [self.get_font_size() for _ in range(len(text_label_list))]

        for text_label in text_label_list:
            if text_label == '\n':



        temp_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_image)

        # Determine the background color of the image
        None

        # Determine size of words
        if self.config['size_words']['static_size']['bool']:
            font_size = self.config['size_words']['static_size']['font_size']

            # Determine if each word will have a different font
            if self.config['fonts']['different_fonts_per_word']:
                for label in text_label_list:

                    # Determine the font per word
                    font_object = self.get_font(font_size)

                    # Determine the color per word (grayscale)

            else:
                pass

    def get_font_size(self):
        '''
        size_words:
            within_and_across_images: True
            across_images_only: False
            static_size:
                bool: False
                font_size: 12
            uniformly: 
                bool: False
                font_size_lower_bound: None
                font_size_upper_bound: None
            gaussian: 
                bool: True
                mean: 20
                standard_deviation: 3
                clip_values_at_number_of_std_deviations: 1
                
        '''
        if self.config['size_words']['static_size']['bool']:
            return int(self.config['size_words']['static_size']['font_size'])
        elif self.config['size_words']['uniformly']['bool']:
            return random.randint(self.lower_bound_font_size, self.upper_bound_font_size)
        elif self.config['size_words']['gaussian']['bool']:
            pass


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
