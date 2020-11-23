# %% [markdown]
# # Telco Churn Analysis 
# https://github.com/fastforwardlabs/cml_churn_demo_mlops
# 

# %%
# General purpose pkgs:
import os
import sys
import time
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import dask as dask
from dask.distributed import Client, LocalCluster, progress


# Import other utils:
root_path = os.getcwd()
sys.path.append(root_path)
import parsing_utils as psu

###########################
## Change these settings ##
###########################
output_fmt = 'parquet' # Choose ['csv', 'csv.gz', or 'parquet']
output_dir = r'c:/data/telco_dat/'+output_fmt

tot_iters = 1 # Total number of output files
num_reps = 1000 # Number of copies to replicate
distrib = 'normal' # Set distribution type for randomization. Select 'normal' or None.

use_dask = True
# cpu_cnt = multiprocessing.cpu_count() - 1
# cpu_cnt = 5 # Reduced cores based on available mem req by worker
# threads_per_worker = 1

####################################################################################
# Read in original data and infer column dtypes to apply to output file.
# Blank rows in "TotalCharges" manually removed so that csv schema parser can convert to decimal properly.
filename = os.path.join(root_path, 'data','WA_Fn-UseC_-Telco-Customer-Churn-mod.csv')

col_dtypes_orig = psu.infer_schema(filename)

# Load data:
data = pd.read_csv(filename)

# print(col_dtypes_orig)

# %%
# Data cleaning:
data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)
data['customerID'] = np.arange(len(data))
data.reset_index(drop=True, inplace=True)

# %%

def gen_perturbed_df(df, num_reps, iter_num, output_dir, output_fmt, distrib=None, return_df=False):
    """
    Generate perturbed dataframe based on input data. Useful in situations where there are large numbers of categorical data.
    Some groups may have too few members to be statistically significant.

    Note that normal distribution scale based on one standard deviation. 
    Noise is introduced by long tail and may lead to non-physical results.
    """

    # Construct padded dataframe:
    df_pad = pd.concat([data]*num_reps)
    df_pad.reset_index(drop=True, inplace=True)
    df_pad['customerID'] = np.arange(len(df_pad))

    ## Perturb numeric fields:
    # Vary tenure by 5%, round to whole number:
    df_pad['tenure'] = np.round( df_pad['tenure']*
                            (1 + psu.gen_random_vec(len(df_pad), 0.05, random_state=iter_num, distribution=distrib)) )
    df_pad['tenure'] = np.maximum(1, df_pad['tenure'].values) # Can't have lower than 1 month tenure

    # Vary MonthlyCharges by 10%:
    df_pad['MonthlyCharges'] = df_pad['MonthlyCharges']*\
                            (1 + psu.gen_random_vec(len(df_pad), 0.1, random_state=iter_num+1, distribution=distrib))
    df_pad['MonthlyCharges'] = np.maximum(0, df_pad['MonthlyCharges'].values)

    # Scale TotalCharges within 10% when tenure above 5 months.
    val_mask_gt = df_pad['tenure'] >= 5
    df_pad.loc[val_mask_gt,'TotalCharges'] = df_pad.loc[val_mask_gt,'TotalCharges']*\
                            (1 + psu.gen_random_vec(np.sum(val_mask_gt), 0.1, random_state=iter_num+2, distribution=distrib))

    # Update TotalCharges for less than 5 mo to be exact computation based on tenure*MonthlyCharges:
    val_mask_lt = df_pad['tenure'] < 5
    df_pad.loc[val_mask_lt,'TotalCharges'] = df_pad.loc[val_mask_lt,'tenure']*df_pad.loc[val_mask_lt,'MonthlyCharges']

    # TotalCharge can't be less than 0:
    df_pad['TotalCharges'] = np.maximum(0, df_pad['TotalCharges'].values)

    df_pad = psu.apply_schema(df_pad, col_dtypes_orig, output_fmt)

    output_fn = os.path.join(output_dir, 'perturb_dat_p'+str(iter_num).zfill(4))
    
    if output_fmt == None:
        # No output file. Just used to test timings without outputs. 
        print('Iteration number: ', iter_num)
    elif output_fmt == 'parquet':
        # Use pyarrow writer directly (good for decimal format):
        pq.write_table(df_pad, output_fn + '.parquet', flavor='spark')
    elif output_fmt == 'csv':
        df_pad.to_csv(output_fn + '.csv', index=False)
    elif output_fmt == 'csv.gz':
        df_pad.to_csv(output_fn + '.csv.gz', index=False, compression='gzip')
    else:
        raise ValueError('Selected output file format not recognized. Currently implemented for: parquet, csv, or csv.gz')
    
    if return_df == True:
        out = df_pad
    else:
        out = iter_num
    return(out)

tic = time.time()

# Run iterations:
if use_dask == True:
    if __name__ == "__main__":  
        with Client() as client:  
            tasks = [dask.delayed(gen_perturbed_df)(data, num_reps, ii, output_dir, output_fmt=output_fmt, distrib=distrib, return_df=False) for ii in range(tot_iters)]
            dask.compute(tasks, scheduler=client)
else:
    # Sequential:
    [gen_perturbed_df(data, num_reps, ii, output_dir, output_fmt=output_fmt, distrib=distrib, return_df=False) for ii in range(tot_iters)]

toc = time.time()

print('Total run time [s]:', np.round(toc - tic))

# Overall approach:
# Augment rows by replicating data. Perturb numeric feature columns. Pass feature cols back into trained model 
# to obtain inferred target labels. Apply original schema for decimal truncation and conversion for output.
# %%
