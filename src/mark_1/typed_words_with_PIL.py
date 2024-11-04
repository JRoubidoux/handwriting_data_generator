from PIL import Image, ImageDraw, ImageFont
import random



# Load different fonts for "Hello," and "World"
font_hello = ImageFont.truetype("/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/shimla-la-font/ShimlaLa-jyGG.otf", 100)  # Font for "Hello,"
# font_world = ImageFont.truetype("/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/fonts/alex-brush-font/AlexBrush-7XGA.ttf", 100)    # Font for "World"

# Set positions for the text
position_hello = (100, 75)
position_world = (300, 75)  # Adjust this position based on the width of "Hello,"



for i in range(1):
    # Add "Hello," to the image with its font
    val1 = random.randint(0, 150)
    val2 = random.randint(0, 150)

    print(val1)
    print(val2)

    # Create a blank image
    image = Image.new('RGB', (800, 400), color=(255, 255, 255))

    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)

    draw.text(position_hello, "Farm Laborer \n Apple farm", font=font_hello, fill=(val1, val1, val1))

    # Add "World" to the image with its font
    #draw.text(position_world, "\nLaborer", font=font_world, fill=(val2, val2, val2))

    # Save the image
    image.save(f"/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/sandbox/hello_world_different_fonts_{3}.png")
