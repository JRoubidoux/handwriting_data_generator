import numpy as np
import albumentations as A
import random
import matplotlib.pyplot as plt
from PIL import Image

# Future things to do
#TODO: In the lightenOrDarkenPartsOfWord transform, make the logic more robust
#TODO: Create a transform that allows

class ConvertDataType(A.ImageOnlyTransform):
    """
    This class takes a numpy array and converts it to a given data type. This is useful when wanting to compose transforms.
    """
    def __init__(self, dtype=np.uint8, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.dtype = dtype
        self.always_apply = always_apply
        self.p = p

    def apply(self, image: np.ndarray, **params):
        probability_value = self.p
        if self.always_apply:
            probability_value = 1.0

        if np.random.random() <= probability_value:
            return image.astype(self.dtype)
        else:
            return image

    def get_transform_init_args_names(self):
        return ("dtype", "always_apply", "p")

class lightenOrDarkenPartsOfWord(A.ImageOnlyTransform):
    """
    This class creates a grid and randomly determines portions of this grid to be lightened or darkened or to remain the same.
    This attempts to mimick how parts of words are darker than others based on how hard people write while writing certain parts of a word. 
    
    As a note, it is not recommended that transforms that would affect the background color be applied before this one as
        it only affect the pixels in the image that aren't the background color. In other words, the images in the pixel that correspond to text.
    """
    def __init__(self, font_color_lower_bound: int, font_color_upper_bound: int, horizontal_gradient_prob: float, vertical_gradient_prob: float, range_of_rows: tuple, range_of_columns: tuple, always_apply=False, p=1.0):
        """
        Constructor:

        Args:
            font_color_lower_bound: The darkest a given font_color gets.
            font_color_upper_bound: The lightest a given font_color gets.
            horizontal_gradient_prob: The probability that the font color gets lighter or darker along the x-axis
            vertical_gradient_prob: The probability that the font color gets lighter or darker along the y-axis
        """
        super().__init__(always_apply, p)
        self.font_color_lower_bound = font_color_lower_bound
        self.font_color_upper_bound = font_color_upper_bound
        self.horizontal_gradient_prob = horizontal_gradient_prob
        self.vertical_gradient_prob = vertical_gradient_prob
        self.range_of_rows = range_of_rows
        self.range_of_columns = range_of_columns
        self.always_apply = always_apply
        self.p = p

    def randomLightenOrDarken(self, image: np.array):
        """
        This function employs the main logic to systematically lighten, darken or allow parts of the image to remain the same.
        
        Args:
            image: A np.array of our image.
        """
        dtype = image.dtype # Track the original datatype of the image so we can convert our image back to that dtype.
        image = image.astype(np.float32) # Convert the image to float to avoid unsigned overflow issues.
        font_color = np.min(image) # The smallest value in the image should represent the font color
        background_color = np.max(image) # The largest value in the image should represent the background color.
        darken_value_add = self.font_color_lower_bound - font_color # How dark our image can get. 
        light_value_add = self.font_color_upper_bound - font_color # How light our image can get. 
        horizontal_value = np.random.random() # Value that determines if we apply lightening or darkening along the x-axis
        vertical_value = np.random.random() # Ditto for y-axis 

        mask = (image != background_color) # Get a boolean mask of all the indicies in the image that correspond to text. 
        light_and_dark_add_array = np.zeros_like(image) # Get an array of zeros the same shape as the image. This will be used to add or subtract values to the pixels of the image to lighten or darken the text.

        image_height, image_width, _ = image.shape

        number_of_rows = 1 # To lighten or darken along the y-axis, we select a number of rows that each represent a value: lighten, normal or darken. If just one row, then the value is left as normal. 
        number_of_columns = 1 # Ditto for the x-axis

        if horizontal_value <= self.horizontal_gradient_prob: # If horizontal_value is less than the probability the user selected, select a given number of columns based on the range the user inputted.
            number_of_columns = np.random.randint(self.range_of_columns[0], self.range_of_columns[1]+1)

        if vertical_value <= self.vertical_gradient_prob: # Ditto for number of rows. 
            number_of_rows = np.random.randint(self.range_of_rows[0], self.range_of_rows[1]+1)

        if number_of_columns == 1 and number_of_rows == 1: # If only 1 row and column are selected, return the original image. 
            return image.astype(dtype)

        if number_of_rows > 1 and number_of_columns > 1: # If more than 1 row and more than 1 column are selected, divide the previously selected lighten and darken values by 2 as to not lighten or darken too much.  
            light_value_add = round(light_value_add/2)
            darken_value_add = round(darken_value_add/2)

        selection = [light_value_add, 0, darken_value_add] # List of values that we can sample from to determine which row or column correspond to which value. 

        # We need to determine the coordinates for the rows and columns as well as whether the value for the rows or columns will be a lighten, normal or darken value.
        row_values = []
        column_values = []
        row_coordinates = []
        column_coordinates = []

        if number_of_rows == 1:
            row_coordinates = [0, (image_height-1)] # Get row coordinates
            row_values = [0, 0] # Get row values
        else:
            # If there are more than 1 rows, the expected value of each row coordinate is a multiple of the mean row height. For a given sample this value varies uniformly between 1/3 of the row height. 
            row_coordinates = [0] 
            mean_row_height = image_height / number_of_rows
            third_of_row_height = round(mean_row_height/3)
            negative_third_of_row_height = -third_of_row_height

            for i in range(number_of_rows):
                value = np.random.choice(selection) # Select a lighten, normal or darken value for the row. 
                row_values.append(value)
                if i == (number_of_rows-1):
                    row_coordinates.append(image_height-1)
                else:
                    sample_value = np.random.randint(negative_third_of_row_height, third_of_row_height)
                    row_coordinates.append(round(mean_row_height*(i+1) + sample_value))

        if number_of_columns == 1: # Same logic as rows but for columns. 
            column_coordinates = [0, (image_width-1)]
            column_values = [0, 0]
        else:
            column_coordinates = [0]
            mean_column_width = round(image_width / number_of_columns)
            third_of_column_width = round(mean_column_width/3)
            negative_third_of_column_width = -third_of_column_width

            for i in range(number_of_columns):
                value = np.random.choice(selection)
                column_values.append(value)
                if i == (number_of_columns-1):
                    column_coordinates.append(image_width-1)
                else:
                    sample_value = np.random.randint(negative_third_of_column_width, third_of_column_width)
                    column_coordinates.append(round(mean_column_width*(i+1) + sample_value))

        # Store the information we gained above and determine grid point coordinates and their corresponding values to compute for the image later on. 
        grid_values = {}

        for i in range(number_of_rows):
            for j in range(number_of_columns):
                if i == 0:
                    if j == 0:
                        correct_row_coordinate = 0
                        correct_column_coordinate = 0
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if j == (number_of_columns - 1):
                        correct_row_coordinate = 0
                        correct_column_coordinate = image_width -1
                        if j == 0:
                            j += 1
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                            j -= 1
                        else:
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if (j != 0 and j != (number_of_columns - 1)):
                        correct_row_coordinate = 0
                        correct_column_coordinate = round((column_coordinates[j]+column_coordinates[j+1])/2)
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                if i == (number_of_rows-1):
                    i_is_zero = False
                    if i == 0:
                        i += 1
                        i_is_zero = True
                    if j == 0:
                        correct_row_coordinate = image_height - 1
                        correct_column_coordinate = 0
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if j == (number_of_columns - 1):
                        correct_row_coordinate = image_height - 1
                        correct_column_coordinate = image_width -1
                        if j == 0:
                            j += 1
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                            j -= 1
                        else:
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if (j != 0 and j != (number_of_columns - 1)):
                        correct_row_coordinate = image_height - 1
                        correct_column_coordinate = round((column_coordinates[j]+column_coordinates[j+1])/2)
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if i_is_zero:
                        i -= 1
                if (i != 0 and i != (number_of_rows - 1)):
                    if j == 0:
                        correct_row_coordinate = round((row_coordinates[i]+row_coordinates[i+1])/2)
                        correct_column_coordinate = 0
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                    if j == (number_of_columns - 1):
                        correct_row_coordinate = round((row_coordinates[i]+row_coordinates[i+1])/2)
                        correct_column_coordinate = image_width -1
                        if j == 0:
                            j += 1
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)
                            j -= 1
                        else:
                            self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)   
                    if (j != 0 and j != (number_of_columns - 1)):
                        correct_row_coordinate = round((row_coordinates[i]+row_coordinates[i+1])/2)
                        correct_column_coordinate = round((column_coordinates[j]+column_coordinates[j+1])/2)
                        self.add_grid_value(correct_row_coordinate, correct_column_coordinate, i, j, row_values, column_values, grid_values)

        # For each square in our grid determine the add or subtract values in our light_and_dark_add_array to be applied to the image later on.
        for i in range(max(number_of_columns-1, 1)):
            for j in range(max(number_of_rows-1, 1)):
                top_left_dict = grid_values[(i, j)]
                top_right_dict = grid_values[(i+1, j)]
                bottom_left_dict = grid_values[(i, j+1)]
                bottom_right_dict = grid_values[(i+1, j+1)]

                # Get the coordinates of a given square in our grid. 
                top_left_coordinate = top_left_dict["coordinates"]
                top_right_coordinate = top_right_dict["coordinates"]
                bottom_left_coordinate = bottom_left_dict["coordinates"]
                
                top_left_value = top_left_dict["color_add_value"]
                top_right_value = top_right_dict["color_add_value"]
                bottom_left_value = bottom_left_dict["color_add_value"]
                bottom_right_value = bottom_right_dict["color_add_value"]

                width_of_square = (top_right_coordinate[0] - top_left_coordinate[0])
                height_of_square = (bottom_left_coordinate[1] - top_left_coordinate[1])

                column_probabilities = np.array([(k/width_of_square) for k in range(width_of_square+1)])
                row_probabilities = np.array([(k/height_of_square) for k in range(height_of_square+1)])

                # Compute the values in the square to lighten or darken certain values. This function determines this such that the lightening or darkening happen gradually rather than having blocks of lighter and darker words. This make the word look more natural. 
                determine_lighten_normal_or_darken_values = determineLightenNormalOrDarkenValues(top_left_value, top_right_value, bottom_right_value, bottom_left_value)

                add_array = np.array([[[determine_lighten_normal_or_darken_values.get_value(vertical_prob, horizontal_prob)]*3 for horizontal_prob in column_probabilities] for vertical_prob in row_probabilities])

                light_and_dark_add_array[top_left_coordinate[1]:(bottom_left_coordinate[1]+1), top_left_coordinate[0]:(top_right_coordinate[0]+1), :] = add_array

        # Lighten and darken parts of the image. 
        image[mask] += light_and_dark_add_array[mask]
        image = np.clip(image, 0, 255)
        return image.astype(dtype)

    def add_grid_value(self, correct_row_coordinate: int, correct_column_coordinate: int, i: int, j: int, row_values: list, column_values: list, grid_values: dict):
        """
        This is a helper function to the randomLightenOrDarken function. 

        Args:
            correct_row_coordinate: The coordinate of the row.
            correct_column_coordinate: The coordinate of the column.
            i: A given index in our grid
            j: A given index in our grid
            row_values: list of lighten, normal, or darken values.
            column_values: same.
            grid_values: This is a dict that contains the grid information for darkening or lightening the image.
        """
        row_value = row_values[i]
        column_value = column_values[j]
        result_value = row_value+column_value
        grid_values[(j, i)] = {"coordinates": (correct_column_coordinate, correct_row_coordinate), "color_add_value": (result_value)}

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        probability_value = self.p
        if self.always_apply:
            probability_value = 1.0

        if np.random.random() <= probability_value:
            return self.randomLightenOrDarken(image)
        else:
            return image

    def get_transform_init_args_names(self):
        return ("background_color", "always_apply", "p")


class determineLightenNormalOrDarkenValues():
    """
    This class is meant to help the lightenOrDarkenPartsOfWord class by assisting in computing the values for the given rectangles of a grid such that the image can be lightened or darkened. 
    """
    def __init__(self, top_left: int, top_right: int, bottom_right: int, bottom_left: int):
        """
        Constructor:

        Args:
            top_left: This represents the top_left lighten, darken, or normal value in the rectangle 
            top_right: Ditto for top_right
            bottom_right: Ditto for bottom_right
            bottom_left: Ditto for bottom_left
        """
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left
        self.top_right_minus_top_left = top_right - top_left

    def get_value(self, vertical_prob: float, horizontal_prob: float):
        """
        This function returns the average value of a lighten, normal, or darken value for a given place in a rectangle

        The below logic is unintuitive as it represnts a more intuitive operation that has been optimized to run faster. 

        Say that we have a given rectangle in our grid where the values for top_left, top_right, bottom_right, bottom_right are: -6, 0, 10, -3.
        This means that the value of the top left pixel in this array is -6, the value in the top right is 0, the value for the bottom right is 10 and the value for the bottom left is -3. 
        Also, say that the rectangle is 100 pixels wide and 50 pixels tall (We'll ignore the depth for now).

        If we want to know what the value of the pixel at row 30 and column 80 should be some average of the values in each of the corners that way the lightening or darkening of the image is gradual across different axes. 

        This average can be computed as the following intuitive formula: 
            The value of the pixel at the top row (y_index: 0) and at x_index: 79 (zero indexed) would be computed as (-6 + (79 / 99)*(0-(-6))). 
                Or written more generally: (top_left + probability_along_x-axis*(top_right - top_left)) (Note the probability_along_x-axis really represents the distance between the left and right values. 0 represents being on the left value, 0.5 is halfway between the values, and 1 is on the right value)
            Similarly the value of the pixel at the bottom row (y-index: 49) and at x_index: 79 would be: (bottom_left + probability_along_x-axis*(bottom_right - bottom_left))

            Now to our interest is in getting the average in a 2D space rather than 1D, so we can do that by taking the two weighted averages along the x-axis we found for the top and bottom rows and then compute the weighted average along
                the y-axis as well. Let A = (the weighted average for the top row) and let B = (the weighted average for the bottom row)

            Then our new weighted average between these two would be: (A - probability_along_y-axis*(B - A)) or ((top_left + probability_along_x-axis*(top_right - top_left)) - probability_along_y-axis*((bottom_left + probability_along_x-axis*(bottom_right - bottom_left)) - (top_left + probability_along_x-axis*(top_right - top_left))))
                The formula in the code below is very similar to this one but uses some algebra to merge two of the multiplication operations into one to optimize time complexity of the code. 
        """
        
        return round((self.top_left) + (horizontal_prob*self.top_right_minus_top_left) + (vertical_prob*(self.bottom_left - self.top_left + horizontal_prob*(self.bottom_right - self.bottom_left + self.top_right_minus_top_left))))

class LightenOrDarkenImage(A.ImageOnlyTransform):
    """
    This transform will lighten or darken all pixels in the image preserving the monotonic nature of the pixels in the 
        image so that the contrast features are preserved but the lightness or darkness of the image is changed. 
    """
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.always_apply = always_apply
        self.p = p

    def augment_array(self, image: np.array):
        '''
        This function allows us to lighten or darken image data following a polynomial line bounded between x: [0,1] and y: [0,1] in R2.

        Assuming that the pixels in our image fall within the range [0-255] we can find a line that is monotonic in nature and determine new values to force our 
            pixels to that will either lighten or darken our whole image. Say for example that the darkest pixels of our image are represented by the value 0 (black), and that
            the lightest pixels in the image are represented by the value 255 (white). We can define a function for the pixels in this image as resulting_pixel_value = original_pixel_value.
            Thus this function just returns the original pixel. If however we wanted to darken the lighter pixels and lighten the darker pixels by some degree we can 
            construct a function to do this. The function constructed is of the form: 'new_pixel_value = slope * (old_pixel_value ** polynomial_degree) + lowest_pixel_value_randomly_selected'
            This function has the property such that the new_pixel_value cannot exceed the value 255 (our highest pixel value) and our lowest_pixel_value cannot be lower than 0 (lowest pixel value). 

        Args:
            image: A np.array object of our image. 
        '''
        data_type = image.dtype # Get the original data type of our image.
        image = image.astype(np.float32)

        max_value = np.max(image) # Get the lightest colored pixel in the image. 

        polynomial_degree = None
        
        y_intercept = random.uniform(0, 0.6) # Choose a y-intercept for our new function that would represent the darkest pixel value in our new image.
        max_y = random.uniform((y_intercept + 0.4), 1) # Choose the highest value our function would take that would represent the lightest pixel value in our new image.

        if random.choice([0, 1]) == 0: # Determine the degree of the polynomial in our function.
            polynomial_degree = 1 - random.random()/2
        else:
            polynomial_degree = 1 + random.random()

        if max_value > 1: # If the pixels in our image are in the assumed range: [0, 255]
            max_y = max_y * 255 # Get the correct max_y
            y_intercept = y_intercept * 255 # Get the correct y-intercept

            slope = (max_y - y_intercept) / (255.0 ** polynomial_degree) # Calculate the slope

            new_image = slope * (image ** polynomial_degree) + y_intercept # Using array broadcasting, calculate the values of the new image.
            new_image = np.clip(new_image, 0.0, 255.0) # Clip the values of our new array to 0 and 255, round errors may lead some values in our image to exceed these numbers. 

            return new_image.astype(data_type) # Convert the image back to its original dtype and return it.

        else: # If the pixels in our image are in the assumed range: [0, 1]
            slope = (max_y - y_intercept) # Calculate our slope

            new_image = slope * (image ** polynomial_degree) + y_intercept # Using array broadcasting, calculate the values of the new image.
            new_image = np.clip(new_image, 0.0, 1.0) # Clip our image to our given bounds.
            return new_image.astype(data_type) # Convert the image back to its original datatype and return it. 

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Perform your custom transformation on the image
        probability_value = self.p
        if self.always_apply:
            probability_value = 1.0
        
        if np.random.random() <= probability_value:
            return self.augment_array(image)
        else:
            return image

    def get_transform_init_args_names(self):
        return ("always_apply", "p")

class TrimPadding(A.ImageOnlyTransform):
    """
    This transform removes padding from an image where padding is assumed to have the value: background_color. 
    If the only values in the image are: background_color, then the no trimming is done and the image is returned as it entered.
    """
    def __init__(self, background_color: int = 255, always_apply: bool = False, p: float = 1.0):
        """
        Constructor:

        Args:
            background_color: This value is the color in our that text is placed on top of. As such if we want to trim our image down to text only, 
            we can trim the background_color of the image. 
        """
        super().__init__(always_apply, p)
        self.background_color = background_color
        self.always_apply = always_apply
        self.p = p

    def trim(self, image: np.array):
        """
        Trim the padding around the non-background area of the image.

        Args:
            image: A np.array of our image.
        """
        # Detect non-background area
        mask = np.any(image != self.background_color, axis=-1)
        y_indices, x_indices = np.where(mask)

        if y_indices.size == 0 or x_indices.size == 0:
            return image
        else:
            # Find bounding box coordinates
            true_y_pos_start, true_y_pos_end = y_indices.min(), y_indices.max()
            true_x_pos_start, true_x_pos_end = x_indices.min(), x_indices.max()
            
            # Trim padding from image and return result
            trimmed_image = image[true_y_pos_start:true_y_pos_end+1, true_x_pos_start:true_x_pos_end+1, :]
            return trimmed_image

    def apply(self, image: np.array, **params):
        """
        Apply our transform.
        """
        probability_value = self.p
        if self.always_apply:
            probability_value = 1.0
            
        if np.random.random() <= probability_value:
            return self.trim(image)
        else:
            return image

    def get_transform_init_args_names(self):
        return ("background_color",)
