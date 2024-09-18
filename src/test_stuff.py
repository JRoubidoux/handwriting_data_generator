import sys
import yaml
import os
import json
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torchvision.transforms import v2
import time
import queue
import threading
sys.path.append('/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/src')
sys.path.append('/grphome/fslg_census/nobackup/archive/common_tools/deep_learning_training_helper/branches/hot_fix/deep_learning_training_helper/src')
from image_generator_mark1 import fontWordOnImage, fontHelper
from custom_transforms import lighten_or_darken_image, minMaxScale, scale255To0And1
print("Python interpreter being used:")
print(sys.executable)

class syntheticGeneratorDataset(Dataset):
    def __init__(self, vocabulary: list, json_dict: dict, config_file_path: str, transforms: list, batch_size: int):
        self.config = self.load_config(config_file_path)
        self.image_maker = fontWordOnImage(vocabulary, json_dict, self.config)
        self.transforms = v2.Compose(transforms)
        self.image_width_multiplier = self.config['image_maker_parameters']['image_width_multiplier']
        self.image_height_multiplier = self.config['image_maker_parameters']['image_height_multiplier']
        self.start_text_x_fraction_of_width = self.config['image_maker_parameters']['start_text_x_fraction_of_width']
        self.start_text_y_fraction_of_height = self.config['image_maker_parameters']['start_text_y_fraction_of_height']
        self.batch_size = batch_size

    def load_config(self, config_file: str):
        '''
        This function loads a .yaml file for the script to use. 
        '''
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(exc)
                exit()

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        image, text_label = self.image_maker.render_word_on_image_and_text_label(self.image_width_multiplier, self.image_height_multiplier, self.start_text_x_fraction_of_width, self.start_text_y_fraction_of_height)

        if self.transforms is not None:
            tensor = self.transforms(image)
        
        return tensor, text_label


class syntheticGeneratorDataset2(IterableDataset):
    def __init__(self, transforms: list, batch_size: int, data_queue):
        self.batch_size = batch_size
        self.transforms = v2.Compose(transforms)
        self.data_queue = data_queue

    def __iter__(self):
        while True:
            try:
                image, text_label = self.data_queue.get(timeout=2)
                if self.transforms is not None:
                    tensor = self.transforms(image)
                
                yield tensor, text_label
            except queue.Empty:
                return


class queueFiller():
    def __init__(self, vocabulary: list, json_dict: dict, config_file_path: str, data_queue, needed_length: int):
        self.config = self.load_config(config_file_path)
        self.image_maker = fontWordOnImage(vocabulary, json_dict, self.config)
        
        self.image_width_multiplier = self.config['image_maker_parameters']['image_width_multiplier']
        self.image_height_multiplier = self.config['image_maker_parameters']['image_height_multiplier']
        self.start_text_x_fraction_of_width = self.config['image_maker_parameters']['start_text_x_fraction_of_width']
        self.start_text_y_fraction_of_height = self.config['image_maker_parameters']['start_text_y_fraction_of_height']
        self.data_queue = data_queue
        self.needed_length = needed_length

    def load_config(self, config_file: str):
        '''
        This function loads a .yaml file for the script to use. 
        '''
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(exc)
                exit()

    def fill_queue(self):
        while self.data_queue.qsize() < self.needed_length:
            image, text_label = self.image_maker.render_word_on_image_and_text_label(self.image_width_multiplier, self.image_height_multiplier, self.start_text_x_fraction_of_width, self.start_text_y_fraction_of_height)
            data_queue.put((image, text_label))


def save_tensor_as_image(tensor, path):
    # Rescale from [0, 1] to [0, 255]
    tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255)  # Ensure values are in range [0, 255]
    
    # Convert to uint8
    tensor = tensor.to(torch.uint8)
    
    # If it's a single-channel image, remove the extra dimension
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert the tensor back to a PIL image
    pil_image = v2.ToPILImage()(tensor)
    
    # Save the image
    pil_image.save(path)



if __name__ == '__main__':
    # vocabulary = ['farmer', 'The cheeky \nFarmer', 'yeet', 'poot \nput \npooterut \npoom']

    path_to_vocab = '/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/vocabularies.json' # r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\vocabularies_of_interest\vocabularies.json" #
    fields = ['occupation']

    vocabulary = []

    with open(path_to_vocab, 'r') as json_in:
        data = json.load(json_in)
        for field in fields:
            vocabulary += data[field]

    '''json_dict = {'/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/a-accountant-signature-font/AccountantSignature-1GXdB.otf': .33,
    '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/alex-brush-font/AlexBrush-7XGA.ttf': .33,
    '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/san-angel-font/SanAngelRegular-WpWyV.otf': .33}

    # '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/hannahfont-font/Hannahfont-xqmj.ttf': .33,
    '''

    root_directory_for_fonts = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts' # r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\fonts" #

    font_helper = fontHelper(root_directory_for_fonts)

    json_dict = font_helper.get_font_and_weight_dictionary_equal_weights()

    config_file_path = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/src/font_word_on_image_config_example.yaml' # r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\src\font_word_on_image_config_example.yaml" #

    output_directory = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/sandbox/test_image_generation' # r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\sandbox\fonts_mark_1" #

    transforms = [v2.PILToTensor(), scale255To0And1(), v2.Resize(size=(64, 256), antialias=True)] #, v2.ColorJitter(brightness=(.9, 1.1),  contrast=(1, 1)), lighten_or_darken_image(), v2.ElasticTransform(alpha=50.0, sigma=10.0), v2.RandomPerspective(distortion_scale=0.2, p=0.5), v2.GaussianNoise(mean=0, sigma=0.025, clip=True), v2.Grayscale(3)]

    print("before data queue", flush=True)

    dataset = syntheticGeneratorDataset(vocabulary, json_dict, config_file_path, transforms, 10000)
    '''data_queue = queue.Queue(maxsize=10000)

    queue_filler = queueFiller(vocabulary, json_dict, config_file_path, data_queue, 1000)

    print("before threads", flush=True)

    thread = threading.Thread(target=queue_filler.fill_queue)
    thread.daemon = True  # Daemon threads exit when the main thread exits
    thread.start()

    print("Thread started", flush=True)

    dataset = syntheticGeneratorDataset2(transforms, 100, data_queue)'''

    dataloader = DataLoader(dataset, batch_size=10, num_workers=8)
    to_PIL = v2.ToPILImage()

    start_time = time.time()
    for i, (tensors, text_labels) in enumerate(dataloader):
        images = [to_PIL(tensor) for tensor in tensors]
        print(i)
        for j, (tensor, text_label) in enumerate(zip(tensors, text_labels)):
            print(text_label)
            tensor = torch.squeeze(tensor, dim=0)
            new_image_name = f"{i}.png"
            new_image_path = os.path.join(output_directory, new_image_name)
            #save_tensor_as_image(tensor, new_image_path)
        if i == 50:
            break
    print("Number of seconds: ", time.time() - start_time)
    