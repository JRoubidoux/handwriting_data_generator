import numpy as np
import albumentations as A
import random

class ConvertDataType(A.ImageOnlyTransform):
    def __init__(self, dtype=np.uint8, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.dtype = dtype

    def apply(self, image: np.ndarray, **params):
        return image.astype(self.dtype)

    def get_transform_init_args_names(self):
        return ("dtype", "always_apply", "p")

class lightenOrDarkenPartsOfWord(A.ImageOnlyTransform):
    def __init__(self, background_color: int, font_color_lower_bound: int, font_color_upper_bound: int, horizontal_gradient_prob: float, vertical_gradient_prob: float, range_of_rows: tuple, range_of_columns: tuple, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.background_color = background_color
        self.font_color_lower_bound = font_color_lower_bound
        self.font_color_upper_bound = font_color_upper_bound
        self.horizontal_gradient_prob = horizontal_gradient_prob
        self.vertical_gradient_prob = vertical_gradient_prob
        self.range_of_rows = range_of_rows
        self.range_of_columns = range_of_columns

    def randomLightenOrDarken(self, image: np.array):
        font_color = np.min(image)
        light_value_add = font_color - np.random.randint(self.font_color_lower_bound, font_color)
        darken_value_add = np.random.randint(font_color, self.font_color_upper_bound) - font_color

        selection = [light_value_add, 0, darken_value_add]

        horizontal_value = np.random.random()
        vertical_value = np.random.random()

        mask = (image == font_color)
        light_and_dark_add_array = np.zeros_like(mask)

        image_height, image_width, _ = image.shape

        number_of_rows = 1
        number_of_columns = 1

        if horizontal_value < self.horizontal_gradient_prob:
            number_of_rows = np.random.randint(self.range_of_rows[0], self.range_of_rows[1])

        if vertical_value < self.vertical_gradient_prob:
            number_of_columns = np.random.randint(self.range_of_columns[0], self.range_of_columns[1])

        divisor = 1
        if number_of_columns > 1 and number_of_rows > 1:
            divisor = 2

        row_values = []
        column_values = []
        row_coordinates = []
        column_coordinates = []

        if number_of_rows == 1:
            row_coordinates = [0, (image_height-1)]
            row_values.append(0)
        else:
            row_coordinates = [0]
            mean_row_height = image_height / number_of_rows

            for i in range(number_of_rows):
                value = np.random.choice(selection)
                row_values.append(value)
                if i == (len(number_of_rows)-1):
                    row_coordinates.append(image_height-1)
                else:
                    sample_value = np.random.randint(-mean_row_height, mean_row_height)
                    row_coordinates.append(round(mean_row_height*i + sample_value))

        if number_of_columns == 1:
            column_coordinates = [0, (image_width-1)]
            column_values.append(0)
        else:
            column_coordinates = [0]
            mean_column_width = image_width / number_of_columns

            for i in range(number_of_columns):
                value = np.random.choice(selection)
                column_values.append(value)
                if i == (len(number_of_columns)-1):
                    column_coordinates.append(image_width-1)
                else:
                    sample_value = np.random.randint(-mean_column_width, mean_column_width)
                    column_coordinates.append(round(mean_column_width*i + sample_value))

        grid_values = {}

        for i in range(0, (len(row_coordinates)-1)):
            if i == 0:
                correct_row_coordinate = 0
            elif i == (len(row_coordinates)-2):
                correct_row_coordinate = image_height - 1
            else:
                correct_row_coordinate = round((row_coordinates[i]+row_coordinates[i+1])/2)

            for j in range(0, (len(column_coordinates)-1)):
                if j == 0:
                    correct_column_coordinate = 0
                elif j == (len(column_coordinates)-2):
                    correct_column_coordinate = image_width - 1
                else:
                    correct_column_coordinate = round((column_coordinates[i]+column_coordinates[i+1])/2)

                row_value = row_values[i]
                column_value = column_values[j]

                result_value = row_value+column_value

                grid_values[(i, j)] = {"coordinates": (correct_row_coordinate, correct_column_coordinate), "color_add_value": (result_value)}

        for i in range(number_of_rows):
            for j in range(number_of_columns):
                top_left_dict = grid_values[(i, j)]
                top_right_dict = grid_values[(i, j+1)]
                bottom_left_dict = grid_values[(i+1, j)]
                bottom_right_dict = grid_values[(i+1, j+1)]

                top_left_coordinate = top_left_dict["coordinates"]
                top_right_coordinate = top_right_dict["coordinates"]
                bottom_left_coordinate = bottom_left_dict["coordinates"]
                bottom_right_coordinate = bottom_right_dict["coordinates"]
                
                top_left_value = top_left_dict["value"]
                top_right_value = top_right_dict["value"]
                bottom_left_value = bottom_left_dict["value"]
                bottom_right_value = bottom_right_dict["value"]

                width_of_square = (top_right_coordinate[0] - top_left_coordinate[0] + 1)
                height_of_square = (bottom_left_coordinate[1] - top_left_coordinate[1] + 1)

                column_probabilities = np.array([(i/width_of_square) for i in range(width_of_square)])
                row_probabilities = np.array([(i/height_of_square) for i in range(height_of_square)])

                

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return self.randomLightenOrDarken(image)

    def get_transform_init_args_names(self):
        return ("background_color", "always_apply", "p")

class LightenOrDarkenImage(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def augment_array(self, image: np.array):
        '''
        This function allows us to lighten or darken image data following a polynomial line bounded between [0,1] and [0,1] in R2.
        '''
        max_value = np.max(image)

        polynomial_degree = None
        
        y_intercept = random.uniform(0, 0.6)
        lower_bound_for_max_y = min((y_intercept + 0.4), 1)
        max_y = random.uniform(lower_bound_for_max_y, 1)

        if random.choice([0, 1]) == 0:
            polynomial_degree = 1 - random.random()/2
        else:
            polynomial_degree = 1 + random.random()

        if max_value > 1:
            data_type = image.dtype
            max_y = max_y * 255
            y_intercept = y_intercept * 255

            slope = (max_y - y_intercept) / (255.0 ** polynomial_degree)

            new_image = slope * (image ** polynomial_degree) + y_intercept
            new_image = new_image.astype(data_type)

            return np.clip(new_image, 0, 255)

        else:
            slope = (max_y - y_intercept)

            new_image = slope * (image ** polynomial_degree) + y_intercept

            return np.clip(new_image, 0, 1.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Perform your custom transformation on the image
        return self.augment_array(image)

    def get_transform_init_args_names(self):
        return ("always_apply", "p")

class TrimPadding(A.ImageOnlyTransform):
    def __init__(self, background_color: int = 255, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.background_color = background_color

    def apply(self, image: np.array, **params):
        """
        Trim the padding around the non-background area of the image.
        """
        # Detect non-background area
        mask = np.any(image != self.background_color, axis=-1)
        y_indices, x_indices = np.where(mask)
        
        # Find bounding box coordinates
        true_y_pos_start, true_y_pos_end = y_indices.min(), y_indices.max()
        true_x_pos_start, true_x_pos_end = x_indices.min(), x_indices.max()
        
        # Trim padding from image and return result
        trimmed_image = image[true_y_pos_start:true_y_pos_end+1, true_x_pos_start:true_x_pos_end+1, :]
        return trimmed_image

    def get_transform_init_args_names(self):
        return ("background_color",)
