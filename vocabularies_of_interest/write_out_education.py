import json

if __name__ == '__main__':

    one_half = "½"

    list_of_text = []
        
    # total text rendered: 
    for yes in ['yes ', '', '-']: 
        if yes == '-':
            for newline in ['', '\n']:
                if newline=='': 
                    list_of_text.append(f"{yes}") # 1
                else:
                    for yes_2 in ['yes ', '', '-']:
                        if yes_2 == '-':
                            list_of_text.append(f"{yes}{newline}{yes_2}") # 1
                        else:
                            for j in range(13):
                                for half_2 in ['', one_half]:
                                    for y_2 in ['', 'y', ' y']:
                                        list_of_text.append(f"{yes}{newline}{yes_2}{j}{half_2}{y_2}") # 1*4*12*3 = 
        else:
            for i in range(13): 
                for half in ['', one_half]: 
                    for y in ['', 'y', ' y']: 
                        for newline in ['', '\n']:
                            if newline=='': 
                                list_of_text.append(f"{yes}{i}{half}{y}")
                            else:
                                for yes_2 in ['yes ', '', '-']: 
                                    if yes_2 == '-':
                                        list_of_text.append(f"{yes}{i}{half}{y}{newline}{yes_2}")
                                    else:
                                        for j in range(13): 
                                            for half_2 in ['', one_half]: 
                                                for y_2 in ['', 'y', ' y']: 
                                                    list_of_text.append(f"{yes}{i}{half}{y}{newline}{yes_2}{j}{half_2}{y_2}")



    output_file = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/education.json'

    dict_object = {"text": list_of_text}

    with open(output_file, 'w') as f:
        json.dump(dict_object, f)