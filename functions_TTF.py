#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:09:01 2023

@author: ruchak
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import Config as cfg
import os


class TTFAnalysis:
    def __init__(self, burnish_results, summary, output_name):
        '''
        This is a constructor for trend analysis

        Parameters
        ----------
        summary : Summary is a DataFrame which includes the MDVLRPOH file
        burnish_results : Burnish Results is the combined rev_012 file 
        output_name : output_name is a string that tells the location to store all the results

        Returns
        -------
        None.

        '''
        self.summary = summary
        self.burnish_results = burnish_results
        self.output_name = output_name
        self.cluster_results = pd.DataFrame()
        self.final_results = pd.DataFrame()
        self.raw_signals = ['DeltaMRE%', 'dBER', 'dPW50', 'dVGAS', 'dSNRE', 'dASYM']
        self.entries = ['MRE', 'BER', 'PW50', 'VGAS', 'SNRE', 'ASYM']
        self.smoothed = ['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM']
        self.linear_fit = ['MREfit', 'BERfit', 'PW50fit', 'VGASfit', 'SNREfit', 'ASYMfit']
        self.std_dev = {'MRE': [], 'BER':[], 'PW50':[], 'VGAS':[], 'SNRE':[], 'ASYM':[]}
        
    def common(self, df_sbr_sn_hd, summary):
        '''
        This will run the changepoint algorithm and give part analysis and overall trend analysis
    
        Parameters
        ----------
        df_sbr_sn_hd : Filtered DataFrame of the particular SBR-SN-HD
        entry : dictionary with current SBR-SN results
    
    
        Returns
        -------
        None.
    
        '''
        #save plots of the piecewise linear fits to PNG file
        #fit the LR and plot the curves
        df_sbr_sn_hd[['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', 'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%']] = \
            df_sbr_sn_hd[['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', 'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%']]\
                .interpolate(method = 'piecewise_polynomial', axis = 0)
        plt.figure(figsize = (15, 28), dpi = 80)
    
        y_true_colors = ['b', mcolors.CSS4_COLORS['darkviolet'], 'm', mcolors.CSS4_COLORS['mediumvioletred'], mcolors.CSS4_COLORS['yellowgreen'],mcolors.CSS4_COLORS['palevioletred']]
        smooth_colors = ['r', mcolors.CSS4_COLORS['orange'], mcolors.CSS4_COLORS['olive'], mcolors.CSS4_COLORS['palegreen'], mcolors.CSS4_COLORS['orchid'], 'c']
        y_colors = ['g', 'c', mcolors.CSS4_COLORS['darkkhaki'], mcolors.CSS4_COLORS['cadetblue'], mcolors.CSS4_COLORS['navy'], mcolors.CSS4_COLORS['crimson']]
        
        
        units = ['', '(dcd)', '(nm)', '(dB)', '(dB)', '(%)']
        subplots = np.arange(711, 718)
        
        ax1 = plt.subplot(subplots[0])
        
        
        plt.title('SERIAL_NUM '+df_sbr_sn_hd['SERIAL_NUM'].iloc[0])
        
        ax = ax1
        plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd['STRESS_CLEARANCE'].values, color = mcolors.CSS4_COLORS['darkgreen'], label = 'STRESS_CLEARANCE')
        plt.xlim(0, self.burnish_results['RH_STRESS_TIME_hrs'].max())
        ax.set_xlabel('RH_STRESS_TIME_hrs')
        
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")   
        plt.ylabel('STRESS CLEARANCE')
        
        
        
        for i in range(len(self.raw_signals)):
            
            #save overall slope
            
            
            ax = plt.subplot(subplots[i+1], sharex = ax1)
        
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.raw_signals[i]].values, y_true_colors[i], label = self.raw_signals[i])
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.smoothed[i]].values, smooth_colors[i], label = self.smoothed[i])
            
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.linear_fit[i]].values, y_colors[i], label = self.linear_fit[i])   
            
            if summary['CPD_' + self.entries[i] + ' TTF_CENSOR'].iloc[0] == 0:
                plt.axvline(summary['CPD_' + self.entries[i] + '_TTF'].iloc[0], color = 'black', ls = '--', label = 'TTF')
                plt.axhline(cfg.cutoff[self.linear_fit[i]], ls = '--', color = mcolors.CSS4_COLORS['darkslategray'])
                plt.plot([], [], ' ', label = 'TTF '+str(np.around(summary['CPD_' + self.entries[i]+'_TTF'].iloc[0], decimals = 2)))
                

            
                
                    
                    
            ax.set_xlabel('RH_STRESS_TIME_hrs')
    
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")   
            plt.ylabel(self.raw_signals[i] + units[i])
            
           
        if not(os.path.exists(os.path.join(cfg.absolute_pth, self.output_name, df_sbr_sn_hd['SBR'].iloc[0]+'/'))):
            os.makedirs(os.path.join(cfg.absolute_pth, self.output_name, df_sbr_sn_hd['SBR'].iloc[0]+'/'))
        
        plt.savefig(cfg.absolute_pth + self.output_name+'/'+df_sbr_sn_hd['SBR'].iloc[0]+'/'+ df_sbr_sn_hd['SBR'].iloc[0]+'_'+\
                    df_sbr_sn_hd['SERIAL_NUM'].iloc[0]+'_'+ df_sbr_sn_hd['HEAD'].iloc[0].astype(int).astype(str)+'_CPD_V3_2.png',\
                        bbox_inches = 'tight')
        
        plt.close() 
        #check for plateau and improving
        
    
    def Processing(self):
        '''
        This is a wrapper to get the plots and mark TTFs

        Returns
        -------
        cluster_results : TYPE
            DESCRIPTION.

        '''
        
        
        fmc_decoder = pd.read_csv(cfg.absolute_pth + 'scripts/FMC_decoder.csv')[['Results', 'FMC']]
        self.summary['SBR_SN_HD'] = self.summary['SBR'] +'_' + self.summary['SN'] + '_' + self.summary['HD'].astype(str)
        for sbr_sn_hd in self.burnish_results['SBR_SN_HD'].unique().tolist():
                
                
                #filter the DF to get rows with specific SBR, SN, HD
                df_sbr_sn_hd = self.burnish_results[self.burnish_results['SBR_SN_HD'] == sbr_sn_hd]
                df_sbr_sn_hd = df_sbr_sn_hd.reset_index()
                
                summary_sbr_sn_hd = self.summary[self.summary['SBR_SN_HD'] == sbr_sn_hd]
                
                self.common(df_sbr_sn_hd, summary_sbr_sn_hd)

        self.summary['CPD_InducTm to TTF results'] = self.summary['CPD_InducTm to TTF results'].replace('I', 'S')
        
        
        cluster_results = self.summary[['SBR', 'SN', 'HD', \
        'CPD_MRE_TTF', 'CPD_BER_TTF', 'CPD_PW50_TTF', 'CPD_VGAS_TTF', 'CPD_SNRE_TTF',\
        'CPD_shunt', 'CPD_SHUNT_time', \
        'CPD_MRE TTF_CENSOR', 'CPD_BER TTF_CENSOR', 'CPD_PW50 TTF_CENSOR', 'CPD_VGAS TTF_CENSOR', 'CPD_SNRE TTF_CENSOR',\
        'CPD_MRE slope', 'CPD_BER slope', 'CPD_PW50 slope', 'CPD_VGAS slope', 'CPD_SNRE slope',\
        'CPD_RH_STRESS_TIME_hrs_max', 'CPD_InducTm to TTF results']]
        cluster_results['CPD_InducTm to TTF results'] = cluster_results['CPD_InducTm to TTF results'].str[:-2]
        cluster_results = cluster_results.merge(fmc_decoder, left_on = 'CPD_InducTm to TTF results', \
                                                right_on = 'Results', how = 'left').drop(columns = ['Results'])
        return cluster_results
        
        
                
            