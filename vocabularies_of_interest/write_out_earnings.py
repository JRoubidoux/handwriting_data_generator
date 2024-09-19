

if __name__ == '__main__':

    superscript = '⁰⁰'

    list_of_text = []
    for i in range(3000):
        if i % 100 == 0:
            str_i = str(i)
            str_i_with_super = str_i + superscript
            str_i_with_dash = str_i + '-'
            str_i_with_super_and_dash = str_i_with_super + '-'
            list_of_text.append(str_i)
            list_of_text.append(str_i_with_super)
            list_of_text.append(str_i_with_dash)
            list_of_text.append(str_i_with_super_and_dash)
        else:
            str_i = str(i)
            list_of_text.append(str_i)

    output_file = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/earnings.txt'

    other_text_to_add = ["no earnings"]

    list_of_text = list_of_text + other_text_to_add

    with open(output_file, 'w') as f:
        for text in list_of_text:
            f.write(f"{text}\n")