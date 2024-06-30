from scripts import fid_score
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import glob
import subprocess

base_directory = "/home/admin/workspace/user/improved-diffusion"
pattern = os.path.join(base_directory, "sampled*/")  # Pattern to match directories starting with 'sampled'

# Use glob.glob with the pattern and the os.DIR flag to match directories
folders = glob.glob(pattern)
folders = ["/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim_ddim10",
           "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim_ddim50",
        #    "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim_ddim10"
           ]

# Print the matched folders
for directory in folders:
    # Directory containing the files
    files = glob.glob(directory + "/*.npz")
    # Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=['timestep', 'Inception Score'])

    # Iterate over all files in the directory
    for filename in tqdm(files):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Load the data from the file
        with np.load(file_path) as data:
            array = data["arr_0"]

        # Save each image in the array to a file
        for i in range(len(array)):
            cv2.imwrite(f"/home/admin/workspace/user/improved-diffusion/sampled_images/images/image_{i}.jpg", array[i][:,:,::-1])

        # call fidelity --gpu 0 --isc --input1 /home/admin/workspace/user/improved-diffusion/sampled_images/images
        result = subprocess.run(['fidelity', '--gpu', '0', '--isc', '--input1', '/home/admin/workspace/user/improved-diffusion/sampled_images/images'], capture_output=True, text=True)
    
        # Extract the Inception Score from the output
        output_lines = result.stdout.split('\n')
        
        mean, std = output_lines[0].split(" ")[-1], output_lines[-1].split(" ")[-1]
        
        # Extract timestep from filename
        timestep = filename.split('_')[-1].split('.')[0]

        # Append the timestep and Inception Score to the DataFrame
        df = df._append({'timestep': timestep, 'Inception Score': mean}, ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f"{directory}/is_results.csv", index=False)
    print(df)
