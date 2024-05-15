#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:57:08 2023

@author: ruchak
"""

import pandas as pd
import numpy as np
import os
import Config as cfg
from tqdm import tqdm
entries = ['MRE', 'BER', 'PW50', 'VGAS', 'SNRE', 'ASYM']
       
    
def processingSummaryFiles(dtMDVLRPOH01, pth2):
    '''
    This function will process all the summary files and combine them to save 1 file. 

    Parameters
    ----------
    dtMDVLRPOH01 : DataFrame of the first file/combined file.
    pth2 : string indicating the path to open the second dataframe.

    Returns
    -------
    dtMDVLRPOH02 : Combined DataFrame.

    '''
    #combine MDVLRPOH files
    
    dtMDVLRPOH02 = pd.read_csv(pth2)
    dtMDVLRPOH01 = dtMDVLRPOH01.rename(columns = {'FLAG_OXIDATION_TREND': 'FLAG_OXIDATION_TREND_01'})
    dtMDVLRPOH02 = dtMDVLRPOH02.rename(columns = {'FLAG_OXIDATION_TREND': 'FLAG_OXIDATION_TREND_02'})
    
    #add the flag oxidation 
    dtMDVLRPOH02 = dtMDVLRPOH02.merge(dtMDVLRPOH01[['DRIVE_SERIAL_NUM', 'FLAG_OXIDATION_TREND_01']], on = 'DRIVE_SERIAL_NUM')
    dtMDVLRPOH02['FLAG_OXIDATION_TREND'] = dtMDVLRPOH02['FLAG_OXIDATION_TREND_01'] | dtMDVLRPOH02['FLAG_OXIDATION_TREND_02']
    dtMDVLRPOH02 = dtMDVLRPOH02.drop(columns = ['FLAG_OXIDATION_TREND_01', 'FLAG_OXIDATION_TREND_02'])
    return dtMDVLRPOH02

def delete7OutOf8(dt01):
    '''
    This function will only keep 1 out of 8 entries for any dataframe

    Parameters
    ----------
    dt01 : Data Log Table dataframe to reduce data.

    Returns
    -------
    dt01 : Reduced DataFrame

    '''
    
    #check if the dataframe has already been reduced
    dt01['MRE_check'] = dt01['MRE_RESISTANCE'] - dt01['MRE_RESISTANCE'].shift(1)
    dt01['EW_SNR_TOT_check'] = dt01['EW_SNR_TOT'] - dt01['EW_SNR_TOT'].shift(1)
    dt01['SN_HD'] = (dt01['SERIAL_NUM'] + '_' + dt01['HEAD'].astype(str))
    br = pd.DataFrame()
        

    for head in dt01['SN_HD'].unique():
        dt01_sn_hd = dt01[dt01['SN_HD'] == head]
        isame, idiff = 0,0
        if len(dt01_sn_hd)>100:
            
            for j in range(1,8):
                if (dt01_sn_hd['MRE_check'].iloc[j] == 0):
                    isame += 1
                if (dt01_sn_hd['EW_SNR_TOT_check'].iloc[j] != 0):
                    idiff += 1
           
    
    
        #if the file has not been reduced previously, reduce it
        if (isame == 7) & (idiff > 0):
            
            sequence = list(dt01_sn_hd.index.values)[::8]
            subset  = dt01[dt01.index.isin(sequence)]
        
        else:
            subset = dt01_sn_hd.copy()
        br = pd.concat([br, subset], ignore_index = True)
       
    dt01 = br.copy()
    dt01 = dt01.drop(columns = ['SN_HD'])
    return dt01

def processingLogFiles(dt01, pth2):
    '''
    This function will process and combine the log files.

    Parameters
    ----------
    dt01 : DataFrame of the first file to combine
    pth2 : Path to the second dataframe to combine

    Returns
    -------
    dtc : Combined dataframe

    '''
    #get the SBR log file names
    
    dt01 = dt01.sort_values(by = ['SERIAL_NUM', 'HD_LGC_PSN', 'RH_STRESS_TIME'], ignore_index = True)
    dt02 = pd.read_csv(pth2).sort_values(by = ['SERIAL_NUM', 'HD_LGC_PSN', 'RH_STRESS_TIME'], ignore_index = True)
    
    #delete 7/8 rows
    dt01 = delete7OutOf8(dt01)
    dt02 = delete7OutOf8(dt02)

    #get the two clearances from the SBRs
    dt01_clr = dt01.groupby(['SBR','SERIAL_NUM', 'HD_LGC_PSN', 'STRESS_CLEARANCE']).count().reset_index()[['SBR','SERIAL_NUM', 'HD_LGC_PSN', 'STRESS_CLEARANCE']]
    dt02_clr = dt02.groupby(['SBR','SERIAL_NUM', 'HD_LGC_PSN', 'STRESS_CLEARANCE']).count().reset_index()[['SBR','SERIAL_NUM', 'HD_LGC_PSN', 'STRESS_CLEARANCE']]
    # print(dt01_clr)
    #get maxtimes
    dt01['RH_MEASURE_TIME'] = dt01['RH_MEASURE_TIME'].fillna(0)
    dt01['READER_DWELL_TIME'] = dt01['READER_DWELL_TIME'].fillna(0)
    dt01maxtime = pd.DataFrame(dt01.groupby(['SERIAL_NUM', 'HD_LGC_PSN'])[['RH_MEASURE_TIME', 'READER_DWELL_TIME']].max().reset_index(level=[0,1])).rename(columns = {'RH_MEASURE_TIME':'RH_MEASURE_TIME_offset',\
                                                                                                                                                                  'READER_DWELL_TIME':'READER_DWELL_TIME_offset'})
    dt01maxtime = dt01maxtime.fillna(0)
    dt02 = dt02.merge(dt01maxtime, on = ['SERIAL_NUM', 'HD_LGC_PSN'], how = 'left')

    #correct the time columns
    dtc = pd.concat([dt01, dt02], ignore_index = True)
    dtc['SBR_old'] = dtc['SBR']
    dtc['SBR'] = dtc['SBR'].iloc[0]
    dtc = dtc.rename(columns = {'RH_MEASURE_TIME': 'RH_MEASURE_TIME_old', 'READER_DWELL_TIME':'READER_DWELL_TIME_old'})
    dtc['RH_MEASURE_TIME_old'] = dtc['RH_MEASURE_TIME_old'].fillna(0)
    dtc['READER_DWELL_TIME_old'] = dtc['READER_DWELL_TIME_old'].fillna(0)
    dtc['RH_MEASURE_TIME_offset'] = dtc['RH_MEASURE_TIME_offset'].fillna(0)
    dtc['READER_DWELL_TIME_offset'] = dtc['READER_DWELL_TIME_offset'].fillna(0)
    dtc['RH_MEASURE_TIME'] = dtc['RH_MEASURE_TIME_old'] + dtc['RH_MEASURE_TIME_offset']
    dtc['READER_DWELL_TIME'] = dtc['READER_DWELL_TIME_old'] + dtc['READER_DWELL_TIME_offset']
    dtc['RH_STRESS_TIME'] = dtc['RH_MEASURE_TIME'] + dtc['READER_DWELL_TIME']
    dt01mintime = dtc.groupby(['SERIAL_NUM', 'HD_LGC_PSN'])['RH_STRESS_TIME'].min().reset_index()
    dt01mintime = dt01mintime.rename(columns = {'RH_STRESS_TIME': 'Min(RH_STRESS_TIME)'})

    #sort the table
    dtc = dtc.sort_values(by = ['SERIAL_NUM', 'HD_LGC_PSN', 'RH_STRESS_TIME'])

    #Add SN_HD column for plots
    dtc['SN_HD'] = dtc['SERIAL_NUM'] + '_' + dtc['HD_LGC_PSN'].astype(int).astype(str)
    try:
        cfg.CURRENT_AP = cfg.AP_VERSIONS[dtc['TEST_SCRIPT_NAME'].unique()[0][6:8]]
    except:
        cfg.CURRENT_AP = 'AP3.6'
    #Correct column DeltaMRE% so the AP36 spaghetti plot JSL file will still work
    dtc = dtc.merge(dt01mintime, on = ['SERIAL_NUM', 'HD_LGC_PSN'], how = 'left')
    dtc_atmintime = dtc[dtc['RH_STRESS_TIME'] == dtc['Min(RH_STRESS_TIME)']]
    dtc_atmintime = dtc_atmintime.add_suffix('_t0')
    mre_resistance_atmintime = dtc_atmintime[['SERIAL_NUM_t0', 'HD_LGC_PSN_t0', 'MRE_RESISTANCE_t0']]
    dtc = dtc.merge(mre_resistance_atmintime, left_on = ['SERIAL_NUM', 'HD_LGC_PSN'], right_on = ['SERIAL_NUM_t0', 'HD_LGC_PSN_t0'], how = 'left').drop(columns = ['SERIAL_NUM_t0', 'HD_LGC_PSN_t0'])
    dtc['DeltaMRE%'] = 100 * (dtc['MRE_RESISTANCE'] - dtc['MRE_RESISTANCE_t0'])/dtc['MRE_RESISTANCE_t0']
    dtc['DeltaMRE%'] = dtc['DeltaMRE%'].fillna(0)
    #correct column SBR to show just the original SBR number, not the modified
    
    

    #delete bad BER data
    col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['BER']
    dtc[col] = dtc[col].replace(0, np.nan)
    
    dtc['SBR_SN_HD'] = dtc['SBR_old'] + '_' + dtc['SERIAL_NUM'] + '_' + dtc['HEAD'].astype(int).astype(str)
    sbr_sn_hd = []
    for i in (dtc['SBR_SN_HD'].unique()):
        subset = dtc[dtc['SBR_SN_HD'] == i]
        if len(subset)>=3:
            sbr_sn_hd.append(i)
    dtc = dtc[dtc['SBR_SN_HD'].isin(sbr_sn_hd)]
    dtc_heads = dtc['SERIAL_NUM'] + '_' + dtc['HEAD'].astype(int).astype(str)
    dtc_heads = dtc_heads.unique()
    dt01_clr['SN_HD'] = dt01_clr['SERIAL_NUM'] + '_' + dt01_clr['HD_LGC_PSN'].astype(int).astype(str)
    dt02_clr['SN_HD'] = dt02_clr['SERIAL_NUM'] + '_' + dt02_clr['HD_LGC_PSN'].astype(int).astype(str)
    
    #Summary table to record deltas each head
    dt_delsum = pd.DataFrame()
    

    for i in range(len(dtc_heads)):
        
        ihd = dtc_heads[i]
        
        imat = dtc[dtc['SN_HD'] == ihd]
        
        mat_clr01 = dt01_clr[dt01_clr['SN_HD'] == ihd]
        mat_clr02 = dt02_clr[dt02_clr['SN_HD'] == ihd]
        
        if (len(mat_clr01)>0) & (len(mat_clr02)>0):
            iclr1 = mat_clr01['STRESS_CLEARANCE'].iloc[0]
            isbr1 = mat_clr01['SBR'].iloc[0]
            iclr1_mat = dtc[(dtc['SN_HD'] == ihd) & (dtc['SBR_old'] == isbr1)]
                 
        
            iclr2 = mat_clr02['STRESS_CLEARANCE'].iloc[0]
            isbr2 = mat_clr02['SBR'].iloc[0]
            iclr2_mat = dtc[(dtc['SN_HD'] == ihd) & (dtc['SBR_old'] == isbr2)]
            
            if (len(iclr1_mat)>0) & (len(iclr2_mat)>0):
            
                entry = {}
                entry['SBR1'] = isbr1
                entry['SBR2'] = isbr2
                entry['SN'] = imat['SERIAL_NUM'].iloc[0]
                entry['HD'] = imat['HD_LGC_PSN'].iloc[0]
                entry['clr1 Ang'] = iclr1
                entry['clr2 Ang'] = iclr2
                ichng = iclr1_mat.index[-1]
                thresh = 20 
                   
                temp_pre_avg = (dtc['TEMPERATURE'].iloc[ichng-thresh: ichng]).mean()
                temp_post_avg = (dtc['TEMPERATURE'].iloc[ichng: ichng+thresh]).mean()
                dtemp = temp_post_avg - temp_pre_avg
                snre_pre_avg = (dtc['EW_SNR_ELEC'].iloc[ichng-thresh: ichng]).mean()
                snre_post_avg = (dtc['EW_SNR_ELEC'].iloc[ichng: ichng+thresh]).mean()
                dsnr = snre_post_avg - snre_pre_avg
                pres_pre_avg = (dtc['PRESSURE'].iloc[ichng-thresh: ichng]).mean()
                pres_post_avg = (dtc['PRESSURE'].iloc[ichng: ichng+thresh]).mean()
                dpres = pres_post_avg - pres_pre_avg
                col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['BER']
                ber_pre_avg = (dtc[col].iloc[ichng-thresh: ichng]).mean()
                ber_post_avg = (dtc[col].iloc[ichng: ichng+thresh]).mean()
                dber = ber_post_avg - ber_pre_avg
                col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['PW50']
                p50_pre_avg = (dtc[col].iloc[ichng-thresh: ichng]).mean()
                p50_post_avg = (dtc[col].iloc[ichng: ichng+thresh]).mean()
                dp50 = p50_post_avg - p50_pre_avg
                entry['TEMP end of SBR1']= temp_pre_avg
                entry['TEMP start of SBR2'] = temp_post_avg
                entry['ESNRE end of SBR1'] = snre_pre_avg
                entry['ESNRE start of SBR2'] = snre_post_avg
                entry['PRESS end of SBR1'] = pres_pre_avg
                entry['PRESS start of SBR2'] = pres_post_avg
                entry['BER end of SBR1'] = ber_pre_avg
                entry['BER start of SBR2'] = ber_post_avg
                entry['P50 end of SBR1'] = p50_pre_avg
                entry['P50 start of SBR2'] = p50_post_avg
                entry['dBER'] = dber
                entry['dESNRE'] = dsnr
                entry['dEW_PW50'] = dp50
                entry['dTEMP'] = dtemp
                entry['dPRESS'] = dpres
                for j in iclr2_mat.index.values:
                    col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['BER']
                    dtc.loc[j, col] = dtc.loc[j, col] - dber
                    dtc.loc[j, 'EW_SNR_ELEC'] = dtc.loc[j, 'EW_SNR_ELEC'] - dsnr
                    col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['PW50']
                    dtc.loc[j, col] = dtc.loc[j, col] - dp50
                    
                
                entry = pd.DataFrame(entry, index = [0])
                dt_delsum = pd.concat([dt_delsum, entry], ignore_index = True)
                bercol = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['BER']
                p50col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['PW50']
                
                dtc = dtc.dropna(subset = [bercol, p50col, 'EW_SNR_ELEC'])
            
            
        
    diff = set(dtc.columns).difference(set(dt01.columns))
    dtc = dtc.drop(columns = diff)
    
    
    
   
    return dtc
    
def checkSSALT(pth, sbr, all_sbrs):
    '''
    Processing for SSALT files

    Parameters
    ----------
    pth : The path where log files are present
    sbr : Name of the current SBR
    all_sbrs : All files related to the current SBR

    Returns
    -------
    bool: True if SBR and processing is done. False, otherwise

    '''
    summary_file = sorted([i for i in all_sbrs if 'SSALT-MDVLRPOH' in i])
    log_file = sorted([i for i in all_sbrs if 'SSALT-P_MDVL_BURNISH_RESULTS_TEMP' in i])
    
    if (len(summary_file) == 1) & (len(log_file) == 1):
            init = pd.read_csv(pth + summary_file[0])
            init_dtc = pd.read_csv(pth + log_file[0])
            init_dtc = init_dtc.rename(columns = {'DRIVE_SERIAL_NUM': 'SERIAL_NUM', 'HD_PHYS_PSN': 'HEAD', 'DRIVE_SBR_NUM':'SBR', \
                                                  'RO_BER': 'RO_AVG_MEAN_CW_SOVA1', 'RH_STRESS_TIME_HOURS': 'RH_STRESS_TIME_hrs'})
            init_dtc['SBR'] = init_dtc['SBR'].str.split('-').str[0]
            if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_preprocessing))):
                os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_preprocessing))
                
            if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_preprocessing, sbr + '/'))):
                os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_preprocessing, sbr))
            init.to_csv(cfg.absolute_pth + cfg.output_preprocessing + sbr +'/'+ sbr +'-MDVLRPOH_Summary.csv', index = False)
            init_dtc.to_csv(cfg.absolute_pth + cfg.output_preprocessing + sbr +'/' + sbr + '-P_MDVL_BURNISH_RESULTS_TEMP_combined_rev012.csv', index = False)
            return True
    else:
        return False
    
def getPreProcessedData():
    '''
    This functions acts like a wrapper and will call all the functions and save the results

    Returns
    -------
    None.

    '''
    pth = cfg.absolute_pth + cfg.raw_data
    
    #get all sbrs
    all_files = sorted(os.listdir(pth))
    sbrs = sorted(list(set([x.split('-')[0] for x in all_files])))
    for sbr in tqdm(sbrs):
        all_sbrs = [i for i in all_files if sbr in i]
        
        check = checkSSALT(pth, sbr, all_sbrs)
        if check:
            continue
        get_summary = sorted([i for i in all_sbrs if ('MDVLRPOH' in i) and ('SSALT' not in i)])
        get_summary.remove(sbr + '-MDVLRPOH_Summary.csv')
        get_log = [i for i in all_sbrs if (i not in get_summary) and ('SSALT' not in i) and (i!= (sbr + '-MDVLRPOH_Summary.csv'))]
        get_log.remove(sbr + '-P_MDVL_BURNISH_RESULTS_TEMP.csv.jmp-rFMC.csv')
        
        # processing the summary files
        init = pd.read_csv(pth + sbr + '-MDVLRPOH_Summary.csv')
        
        for i in range(len(get_summary)):
            
            init = processingSummaryFiles(init, pth + get_summary[i])
        
        init_dtc = pd.read_csv(pth + sbr + '-P_MDVL_BURNISH_RESULTS_TEMP.csv.jmp-rFMC.csv')
        
        for i in range(len(get_log)):
            
            init_dtc = processingLogFiles(init_dtc, \
                                pth + get_log[i])
        if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_preprocessing))):
            os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_preprocessing))
            
        if not(os.path.exists(os.path.join(cfg.absolute_pth, cfg.output_preprocessing, sbr + '/'))):
            os.mkdir(os.path.join(cfg.absolute_pth, cfg.output_preprocessing, sbr))
        
        init.to_csv(cfg.absolute_pth + cfg.output_preprocessing + sbr +'/'+ sbr +'-MDVLRPOH_Summary.csv', index = False)
        init_dtc.to_csv(cfg.absolute_pth + cfg.output_preprocessing + sbr +'/' + sbr + '-P_MDVL_BURNISH_RESULTS_TEMP_combined_rev012.csv', index = False)
         
    
       
        
