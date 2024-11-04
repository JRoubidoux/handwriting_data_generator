mark_1: Proof of concept.  (This is working quite well on the iowa census field for occupation. It was acheiving around 80% word accuracy on that dataset.)
    features include:
        - Render a word on an image from a font file. 
            - This is done naively, basically eyeballing that everything is working as intended. 
        - control font, font size and color of font
            - User can control which fonts are used more than other via weighting. 
            - User can choose one of the following options to select the font size and color:
                - Static value: never changes
                - Value sampled from uniform distribution where user defines the range. 
                - Value sampled from Gaussian distribution where user defines mean, std deviation and range to keep values in. 
        - control background color:
            - static, uniform or gaussian
        - control how an underline is drawn on image:
            - no underline
            - full underline
            - dotted underline


mark_2:
    - new_features:
        - artifacts randomly placed in image to add noise. (letters leaking into the top or bottom, dashes through images)
        - images can be padded in different ways to allow for words to be places differently.
        - transform (Add blotting to lettering to simulate how ink might blot)
        - transform (Give option to adjust the shading of the font color based on location on the image.)
    - Changes to existing features:
        - Images with multiple lines of text can have multiple underlines for each section of text. (This will be determined based on the given font.)
        - Default vocabularies will be expanded to include more words.
        - underlines will be drawn with some variation to the slope. Slope won't always be zero.
        - fonts that are by default apart of the library will be stored in a better way.
        - render text better:
            - have a map stored that for a given font file and font size, will show the optimal image size and where to start drawing the text as to not unnecessarily exhaust time during program execution.
            - have a map stored that defines the optimal place to put an underline under a given font depending on the font type and size.  
            - '''json
                {
                    'font_path': {
                        font_size: {
                            'a': {
                                'plot_coordinates': (offset_x, offset_y),
                                'height': height
                                'width': width
                                 }
                                   }
                                 }
                }'''

    Transforms_for_mark2:
        - (custom transform) different shading applied throughout the image.
        - (custom transform) "coffee" spills applied to different parts of the image. 
        - (custom transform) Give images an JPEG compression type look. 
        - (custom transform) text warping
        - (pytorch) text stretching
        - (custom transform) section shading randomly on image.


mark_3:
    - new_features:
        - Begin to build out support for other languages.
        - Define a custom dataset that can be used that generates images on a separate thread to allow for near zero latency during model training. 


mark_4:
    - Changes to existing features:
        - Given an existing image, have a function that allows text to be written starting from a given (x, y) coordinate where the x coordinate will be the left part of the text and the y coordinate will correspond to the 
            optimal underline position of the text. (This feature will allow handwriting to be synthetically placed onto template images such as census records to help boost the performance of a variety of different models)