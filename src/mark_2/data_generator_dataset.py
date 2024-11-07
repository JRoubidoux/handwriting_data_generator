import sys
from torch.utils.data import Dataset
sys.path.append(r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\src\mark_2")
import data_generator as dg
print("Python interpreter being used:")
print(sys.executable)

# Current:
# TODO: make another class for non-merging sythetic data.

class syntheticGeneratorDataset(Dataset):
    """
    This class inherits from the pytorch Dataset class allowing it to integrate into a pytorch Dataloader object. Since the data generator can generate endless amount of 
        data, the necessary index parameter in the __getitem__ function isn't used, and the __len__ function always returns the batch_size. 
    """
    def __init__(self, config: dict, background_color_manager, font_color_manager, font_size_manager, base_image_transforms: list, word_image_same_transforms: list, word_image_different_transforms: list, merged_image_transforms: list, batch_size: int):
        self.merge_word_images_on_base_image = dg.mergeWordImagesOnBaseImage(base_image_transforms, word_image_same_transforms, word_image_different_transforms, merged_image_transforms, config["image_generation"]["base_image"], background_color_manager, font_color_manager, font_size_manager)
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        image, text_label = self.merge_word_images_on_base_image.get_base_image_merged_with_word_images(True)
        
        if self.transforms is not None:
            tensor = self.transforms(image)
        
        return tensor, text_label