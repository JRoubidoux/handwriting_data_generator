from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

print("model loaded")