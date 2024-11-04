import json

if __name__ == "__main__":
    input_file = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\fonts_and_weights\fonts_and_weights_handwriting_for_iowa_2.json"
    output_file = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\fonts_and_weights\fonts_and_weights_handwriting_for_iowa_2_float.json"

    with open(input_file, "r") as json_in:
        dict_in = json.load(json_in)

    total_weight = 0

    for value in dict_in.values():
        total_weight += value

    for key in dict_in.keys():
        dict_in[key] /= total_weight

    with open(output_file, "w") as json_out:
        json.dump(dict_in, json_out)