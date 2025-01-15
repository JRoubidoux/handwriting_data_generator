import os

def get_font_file_names(root_dir):
    font_files = []
    
    for subdir, _, files in os.walk(root_dir):
        ttf_file = None
        otf_file = None
        
        for file in files:
            if file.lower().endswith('.ttf'):
                ttf_file = file
            elif file.lower().endswith('.otf'):
                otf_file = file
        
        if ttf_file:
            font_files.append(subdir + '/' + ttf_file)
        elif otf_file:
            font_files.append(subdir + '/' + otf_file)
        else:
            print("no file for " + subdir + "!!!")
    
    return font_files

def format_font_names(font_files):
    formatted_files = []
    for idx, file in enumerate(font_files, 1):
        base_name, ext = os.path.splitext(file)
        formatted_name = f"{base_name}-{idx}{ext}"
        formatted_files.append(formatted_name)
    
    return formatted_files

# Change this to the path of your root directory
root_dir = "/home/jl988/fsl_groups/fslg_census/nobackup/archive/projects/paris_french_census/branches/jacob/RLL_paris_french_census/src/french_census/french_fonts"
font_files = get_font_file_names(root_dir)
#formatted_font_names = format_font_names(font_files)

# Print the formatted file names
print("\",\n\"".join(sorted(font_files)))