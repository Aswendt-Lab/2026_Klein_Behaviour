# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:43:09 2025

@author: arefk
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy import stats  # for statistical testing

###############################
# User setting: choose distance metric: "euclidean" or "vertical"
###############################
distance_metric = "euclidean"   # Change to "vertical" to use vertical distances

###############################
# Setup directories and paths
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures", "pythonFigs")
os.makedirs(figures_dir, exist_ok=True)

# File paths for the PRR data and the clustering output
prr_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
cluster_csv_file = os.path.join(output_dir, 'low_high_deficit_groups_kmeans_based_on_last_timepoint.csv')

###############################
# Load data
###############################
# Load PRR data
df_prr = pd.read_csv(prr_csv_file)

# Load clustering data
df_cluster = pd.read_csv(cluster_csv_file)
# Filter for tp == 0 if applicable
if 'tp' in df_cluster.columns:
    df_cluster_tp0 = df_cluster[df_cluster['tp'] == 0].copy()
else:
    df_cluster_tp0 = df_cluster.copy()

# Keep only one row per record_id with the fixed_type assignment
df_cluster_unique = df_cluster_tp0[['record_id', 'fixed_type']].drop_duplicates()

# Merge the PRR data with clustering assignment on record_id
df_merged = pd.merge(df_prr, df_cluster_unique, on='record_id', how='left')
df_merged.to_csv(os.path.join(output_dir, "all_assessment_with_merged_low_high_deficit_groups_kmeans_based_on_last_timepoint_cluster.csv"))