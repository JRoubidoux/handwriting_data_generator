import string

if __name__ == '__main__':

    letters = letters = string.ascii_lowercase

    list_of_text = []

    for letter in letters[:3]:
        for i in range(1000):
            str_i = str(i)
            letter_in_front = letter + str_i
            letter_in_back = str_i + letter
            list_of_text.append(str_i)
            list_of_text.append(letter_in_front)
            list_of_text.append(letter_in_back)


    output_file = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/cardNo.txt'

    with open(output_file, 'w') as f:
        for text in list_of_text:
            f.write(f"{text}\n")