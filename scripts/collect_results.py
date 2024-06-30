import pandas as pd
import os
import sys
import numpy as np


def collect_results(results_dir, output_file):
    results = pd.DataFrame({"timestep": [], "fid": [], "is": []})
    for dir in results_dir:
        df_fid =  pd.read_csv(f"{dir}/fid_results.csv")
        # fid_score
        fid_score = min(df_fid["FID"].values)
        fid_score = np.round(fid_score, 2)
        
        # is_score
        df_is = pd.read_csv(f"{dir}/is_results.csv")
        is_score = max(df_is["Inception Score"].values)
        is_score = np.round(is_score, 2)
        
        timestep = dir.split('_')[-1]
        results = results._append({"timestep": timestep, "fid": fid_score, "is": is_score}, ignore_index=True)
        
    results.to_csv(output_file, index=False)
    return results
        
        
    
if __name__ == '__main__':
    results_dir = ["/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim10",
                   "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim50",
                   "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim100",
                   "/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim250",]
    collect_results(results_dir, "/home/admin/workspace/user/improved-diffusion/results/fid_is_results.csv")
    
