"""
Create tests for each function in each class.
"""

import unittest
import sys
sys.append(r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\handwriting_data_generator\src\mark_2")
import os
import data_generator as dg
from PIL.ImageFont import FreeTypeFont

class DataGeneratorTests(unittest.TestCase):
    """
    This class implements tests for the data generator code.
    """

    def setUp(self):
        pass

    def test_customError_class(self):
        try:
            ce = dg.CustomError("String")
            raise ce
        except Exception as e:
            assert e.__str__ == "String", "Failed the instantiation of the CustomError class."
        try:
            var = 123
            ce = dg.CustomError(f"F-String {var}")
            raise ce
        except Exception as e:
            assert e.__str__ == "F-String 123", "Failed the instantiation of the CustomError class."

    def test_vocabManager_class(self):
        path_to_vocab = r""
        vm = dg.vocabManager(path_to_vocab)
        valid_words = set(["apple", "orange", "banana", "pear"])
        word = vm.get_next_text_from_vocab()
        assert word == "apple", f"The first word in the vocab is apple. The first word returned is {word}."
        word = vm.get_random_text_from_vocab()
        assert word in valid_words, f"The word:{word} is not in the set of valid words in this test. Check the vocab and this function."
        word = vm.get_text(False)
        assert word == "orange", f"The second word in the vocabulary is orange, the word returned was {word}."

    def test_fontObjectManager_class(self):
        fonts_and_weights_path = ""
        font_size_lower_bound = 50
        font_size_upper_bound = 60
        fom = dg.fontObjectManager(fonts_and_weights_path, font_size_lower_bound, font_size_upper_bound)
        smallest_font_weight = 0.1
        assert fom.font_size_lower_bound == font_size_lower_bound, "The fontObjectManager didn't select the correct lower bound."
        assert fom.font_size_upper_bound == font_size_upper_bound, "The fontObjectManager didn't select the correct upper bound."
        assert smallest_font_weight == fom.smallest_weight, f"The smallest weight selected in the fonts and weights file should have been 0.1, the smallest weight selected was: {fom.smallest_weight}."
        assert len(fom.font_dictionaries) == 10, "Based on the weights in the font and weights file, there should be 10 pointers in the list of font_dictionaries."
        assert isinstance(fom.font_dictionaries[0][font_size_lower_bound], FreeTypeFont), "The dictionary should map to a freeTypeFont object."

    def test_fontObjectManagerGivenCharacters_class(self):
        # Test this with a vocabulary where at all of the fonts support the given characters.
        # Test this with a vocabulary where only one of the fonts support the given characters.
        # Test this with a vocabulary where none of the fonts support all of the characters.
        # Test get reduced text.

        pass

    def test_fontLetterPlotDictionaryInstantiator_class(self):
        # Test a font that renders some characters well.
        # Test a font that doesn't render characters.
        pass

    def test_fontObjectManagerGivenVocabulary_class(self):
        # Test a font that supports all characters.
        # Test a font that doesn't support all characters.
        # Test other functions.
        pass

    def test_numberManager_class(self):
        pass

    def test_backgroundColorManager_class(self):
        pass

    def test_fontColorManager_class(self):
        pass

    def test_underlineColorManager_class(self):
        pass

    def test_fontSizeManager_class(self):
        pass

    def test_padImage_class(self):
        pass

    def test_configLoader_class(self):
        pass

    def test_loadFontsAndWeights_class(self):
        pass

    def test_cubicBezierCurve_class(self):
        pass

    def test_drawDashesWithBezier_class(self):
        pass

    def test_drawWordOnImage_class(self):
        pass

    def test_drawWordOnImageInstantiator_class(self):
        pass

    def test_Point_class(self):
        pass

    def test_Quadrilateral_class(self):
        pass

    def test_getBoundsForWindowOnBaseImageFromQuadrilateral_class(self):
        pass

    def test_determineNewBaseImageBounds_class(self):
        pass

    def test_getNewBaseImage_class(self):
        pass

    def test_determineNewWindowBounds_class(self):
        pass

    def test_Window_class(self):
        pass

    def test_mergeWordImageOnBaseImage_class(self):
        pass

    def test_mergeWordImageOnBaseImageInstantiator_class(self):
        pass

    def test_cropMergedImageToViewSize_class(self):
        pass

    def test_cropMergedImageToViewSizeInstantiator_class(self):
        pass

    def test_transformManager_class(self):
        pass

    def test_transformWordImagesForBaseImage_class(self):
        pass

    def test_transformBaseImage_class(self):
        pass

    def test_transfromMergedImage_class(self):
        pass

    def test_mergeWordImagesOnBaseImage(self):
        pass

if __name__ == "__main__":
    unittest.main()