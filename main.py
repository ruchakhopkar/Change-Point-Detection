# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:39:52 2023

@author: 942691
"""
#RUCHAS NOTES
#no critical delta checks for last cluster
#no check for shunting
import pandas as pd
from functions import *
import time
import os
from conversion_012 import *
import Config as cfg
from tqdm import tqdm
from functions_TTF import *

#file structure
##################################################################################################
# for eg
# onset
#   |
#   scripts
    #     |->main.py
    #     |->Config.py
    #     |->functions.py
    #     |->dataset
    # 00_Raw
    #     |-> TT5TP212804-A-MDVLRPOH_Summary.csv
    #     |-> TT5TP212804-A-P_MDVL_BURNISH_RESULTS_TEMP.csv.jmp-rFMC.csv
    #     |-> TT5TP212804-B-MDVLRPOH_Summary.csv
    #     |-> TT5TP212804-B-P_MDVL_BURNISH_RESULTS_TEMP.csv.jmp-rFMC.csv
    #     |-> TT5TP212804-MDVLRPOH_Summary.csv
    #     |-> TT5TP212804-P_MDVL_BURNISH_RESULTS_TEMP.csv.jmp-rFMC.csv


##################################################################################################

#convert the raw data into files we can use
getPreProcessedData()


#list all SBRs to process
l = sorted(os.listdir(cfg.absolute_pth + cfg.output_preprocessing))

#make output folder
if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_decisions))):
        os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_decisions))

for x in tqdm(l):
    a = time.time()
    # read the input files
    burnish_results = pd.read_csv(cfg.absolute_pth + cfg.output_preprocessing + x + '//' + x +'-P_MDVL_BURNISH_RESULTS_TEMP_combined_rev012.csv')
    summary = pd.read_csv(cfg.absolute_pth + cfg.output_preprocessing + x + '//' + x +'-MDVLRPOH_Summary.csv')
    try:
        cfg.CURRENT_AP = cfg.AP_VERSIONS[burnish_results['TEST_SCRIPT_NAME'].unique()[0][6:8]]
    except:
        cfg.CURRENT_AP = 'AP3.6'
    
    #save the results
    if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_decisions, x +'/'))):
        os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_decisions, x))
    #run the analysis
    trend = trendAnalysis(summary, burnish_results, cfg.output_decisions)
    final_results, cluster_results = trend.main()
    
    #save all the results
    final_results.to_csv(cfg.absolute_pth + cfg.output_decisions+'/' + x + '/'+ x +'-P_MDVL_BURNISH_RESULTS_TEMP_oxidtrend_CPD_V3_2.csv', index = False)
    cluster_results.to_csv(cfg.absolute_pth + cfg.output_decisions+'/' + x + '/'+ x +'-P_MDVL_BURNISH_RESULTS_TEMP_CPD_V3_2.csv', index = False)
    
    
    ttfanalysis = TTFAnalysis(final_results, cluster_results, cfg.output_TTF)
    cluster_ttf_results = ttfanalysis.Processing()
    cluster_ttf_results.to_csv(cfg.absolute_pth + cfg.output_TTF+'/' + x + '/'+ x +'-P_MDVL_BURNISH_RESULTS_TEMP_CPD_V3_2.csv', index = False)
    b = time.time()

    print('Total execution time for SBR ' + x + ' is ', (b-a)/60)
    
