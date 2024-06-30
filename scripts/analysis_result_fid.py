from scripts import fid_score
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import glob

import glob
import os

base_directory = "/home/admin/workspace/user/improved-diffusion"
pattern = os.path.join(base_directory, "sampled*/")  # Pattern to match directories starting with 'sampled'

# Use glob.glob with the pattern and the os.DIR flag to match directories
folders = glob.glob(pattern)
folders = ["/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim_ddim10",
           "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim_ddim50",
           "/home/admin/workspace/user/improved-diffusion/sampled_images_uniform_noise_ddim_ddim10",
           "/home/admin/workspace/user/improved-diffusion/sampled_images_uniform_noise_ddim_ddim50"]
# Print the matched folders
for directory in folders:
    # Directory containing the files
    files = glob.glob(directory + "/*.npz")
    # Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=['timestep', 'FID'])

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

        # Calculate the FID score
        fid_value = fid_score.calculate_fid_given_paths(
            ["/home/admin/workspace/user/improved-diffusion/scripts/cifar_10_path_statistic.npz",
            "/home/admin/workspace/user/improved-diffusion/sampled_images/images"],
            128, "cuda:3", 2048
        )

        # Extract timestep from filename
        timestep = filename.split('_')[-1].split('.')[0]
        print(fid_value)
        # Append the timestep and FID to the DataFrame
        df = df._append({'timestep': timestep, 'FID': fid_value}, ignore_index=True)

    # Print the DataFrame
    df.to_csv(f"{directory}/fid_results.csv")
    print(df)