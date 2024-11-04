import csv
import json

class sampling_function():
    def __init__(self, min: int, max: int):
        self.slope = self.get_slope(min, max)

    def get_slope(self, min: int, max: int):
        return (999)/(max-min)

    def get_new_value(self, frequency):
        return round(1 + self.slope*frequency)

if __name__ == "__main__":
    input_file = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\vocabularies_of_interest\create_occupations.tsv"
    output_file = r"C:\Users\Jackson Roubidoux\RLL\repos\data_generator\RLL_handwriting_data_generator\vocabularies_of_interest\occupations_2.json"

    output_dict = {}
    min = float("inf")
    max = -min

    with open(input_file, 'r') as tsv_in:
        reader = csv.reader(tsv_in, delimiter='\t')
        for row in reader:
            value = int(row[1])
            output_dict[row[0]] = int(value)
            if value < min:
                min = value
            if value > max:
                max = value

    sample_function = sampling_function(min, max)

    for key in output_dict.keys():
        value = output_dict[key]
        output_dict[key] = sample_function.get_new_value(value)

    with open(output_file, 'w') as json_out:
        json.dump(output_dict, json_out)