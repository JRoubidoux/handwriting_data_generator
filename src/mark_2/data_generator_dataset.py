from torch.utils.data import Dataset
from torchvision.transforms import v2

# Current:
# TODO: make another class for non-merging sythetic data.

class syntheticGeneratorDataset(Dataset):
    """
    This class inherits from the pytorch Dataset class allowing it to integrate into a pytorch Dataloader object. Since the data generator can generate endless amount of 
        data, the necessary index parameter in the __getitem__ function isn't used, and the __len__ function always returns the batch_size. 
    """
    def __init__(self, merge_word_images_on_base_image, transforms: list, batch_size: int):
        self.merge_word_images_on_base_image = merge_word_images_on_base_image
        self.transforms = v2.Compose(transforms)
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index: int):
        image, text_label = self.merge_word_images_on_base_image.get_base_image_merged_with_word_images(True)
        
        if self.transforms is not None:
            tensor = self.transforms(image)
        
        return tensor, text_label