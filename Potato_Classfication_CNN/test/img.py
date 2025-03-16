import os

def rename_files_in_folder(folder_path, prefix="Healthy_Potato_"):
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort the files to ensure consistent renaming order
    files.sort()
    
    # Iterate through the files and rename them
    for index, file_name in enumerate(files, start=1):
        # Get the file extension
        file_extension = os.path.splitext(file_name)[1]
        
        # Create the new file name
        new_name = f"{prefix}{index}{file_extension}"
        
        # Get the full paths
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

# Specify the folder containing the files
folder_path = r"C:/Users/divya/Downloads/PH/train"
rename_files_in_folder(folder_path)
