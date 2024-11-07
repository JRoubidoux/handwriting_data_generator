mark_1: Proof of concept.  (This is working quite well on the iowa census field for occupation. It was acheiving around 80% word accuracy on that dataset.)
    features include:
        - Render a word on an image from a font file. (Completed)
            - This is done naively, basically eyeballing that everything is working as intended.
        - control font, font size and color of font (Completed)
            - User can control which fonts are used more than other via weighting. 
            - User can choose one of the following options to select the font size and color:
                - Static value: never changes
                - Value sampled from uniform distribution where user defines the range. 
                - Value sampled from Gaussian distribution where user defines mean, std deviation and range to keep values in. 
        - control background color: (Completed)
            - static, uniform or gaussian
        - control how an underline is drawn on image: (Completed)
            - no underline
            - full underline
            - dotted underline

mark_2:
    - new_features:
        - artifacts randomly placed in image to add noise. 
            - letters leaking into the top or bottom. (Completed)
            - dashes on images. (Completed)
        - images can be padded in different ways to allow for words to be places differently. (Completed)
        - transform
            - Adjust the lightness and darkness of a font color across a word. (Completed)
    - Changes to existing features:
        - render text better: (Completed)
            - have a map stored that for a given font file and font size, will show the optimal image size and where to start drawing the text as to not unnecessarily exhaust time during program execution. (Completed)
            - have a map stored that defines the optimal place to put an underline under a given font depending on the font type and size. (Completed via using the undercase c for that word as the underline start pos.)
        - Merge synthetic images on a given image: (Completed)
            - Given an existing image, have a function that allows text to be written starting from a given (x, y) coordinate where the x coordinate will be the left part of the text and the y coordinate will correspond to the 
            optimal underline position of the text. This feature will allow handwriting to be synthetically placed onto template images such as census records to help boost the performance of a variety of different models.

    Transforms_for_mark2:
        - (Albumentations) Give images an JPEG compression type look. 
        - (Albumentations) text warping.
        - (Albumentations) text stretching.
        
mark_3:
    - new_features:
        - Begin to build out support for other languages.
        - Define a custom dataset that can be used that generates images on a separate thread to allow for near zero latency during model training. 
        - dashes through words.
        - transform 
            - Add blotting to lettering to simulate how ink might blot.
        - Images with multiple lines of text can have multiple underlines for each section of text. This will be determined based on the given font.
    - Changes to existing features:
        - underlines will be drawn with some variation to the slope. Slope won't always be zero.
        - fonts that are by default apart of the library will be stored in a better way. (Perhaps pickled)

    Transforms for mark_3:
        - (custom transform) different shading applied throughout the image.
        - (custom transform) "coffee" spills applied to different parts of the image. 
        - (custom transform) section shading randomly on image.
