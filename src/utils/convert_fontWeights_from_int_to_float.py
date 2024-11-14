import json
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, "r") as json_in:
        dict_in = json.load(json_in)

    total_weight = 0

    for value in dict_in.values():
        total_weight += value

    for key in dict_in.keys():
        dict_in[key] /= total_weight

    with open(output_file, "w") as json_out:
        json.dump(dict_in, json_out)