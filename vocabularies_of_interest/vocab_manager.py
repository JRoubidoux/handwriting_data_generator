import json

class vocabManager():
    def __init__(self, current_vocabularies_file_path: str):
        self.current_vocabularies_file_path = current_vocabularies_file_path
        self.vocabulary = self.load_vocab_file()

    def load_vocab_file(self, ):
        with open(self.current_vocabularies_file_path, 'r') as json_in:
            vocab = json.load(json_in)
        return vocab
        
    def load_in_data_and_write_to_vocab(self, field, data_file_path: str):
        if field not in self.vocabulary:
            print("Field must be entered manually prior to appending data to it.")
        else:
            text_in_field = set(self.vocabulary[field])

            with open(data_file_path, 'r') as f_in:
                for line in f_in:
                    text = line.strip().lower()
                    if text not in text_in_field:
                        self.vocabulary[field].append(text)
                        self.vocabulary[field].append(text.title())
                        self.vocabulary[field].append(text.upper())
            with open(self.current_vocabularies_file_path, 'w') as json_out:
                json.dump(self.vocabulary, json_out, indent=2)

    def load_in_data_and_write_to_vocab_from_json(self, field, data_file_path: str):
        if field not in self.vocabulary:
            print("Field must be entered manually prior to appending data to it.")
        else:
            text_in_field = set(self.vocabulary[field])

            with open(data_file_path, 'r') as f_in:
                dict_object = json.load(f_in)
                for text in dict_object["text"]:
                    text = text.lower()
                    if text not in text_in_field:
                        self.vocabulary[field].append(text)
                        self.vocabulary[field].append(text.title())
            with open(self.current_vocabularies_file_path, 'w') as json_out:
                json.dump(self.vocabulary, json_out, indent=2)
            

if __name__ == '__main__':
    current_vocabularies_file_path = '/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/vocabularies.json'
    text_in = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/vocabularies_of_interest/earnings.txt'
    vocab_manager = vocabManager(current_vocabularies_file_path)
    vocab_manager.load_in_data_and_write_to_vocab("earnings", text_in)