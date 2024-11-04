import os

path_to_subdirectories = '/grphome/fslg_census/nobackup/archive/common_tools/handwriting_data_generator/branches/init/RLL_handwriting_data_generator/new_fonts'

for sub_dir in os.listdir(path_to_subdirectories):
    sub_dir_strip_zip = sub_dir.replace(".zip_font", "")
    new_path = os.path.join(path_to_subdirectories, sub_dir_strip_zip)
    old_path = os.path.join(path_to_subdirectories, sub_dir)
    os.rename(old_path, new_path)