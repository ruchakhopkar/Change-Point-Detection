#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:52:10 2023

@author: ruchak
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter, find_peaks
from tqdm import tqdm
import math
import ruptures as rpt
import Config as cfg
from scipy import interpolate
pd.options.mode.chained_assignment = None 
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)
def fill_nan(A, peaks):
    '''
    interpolate to fill nan values
    '''
    df = pd.DataFrame()
    df['x'] = A
    df.loc[peaks, 'x'] = np.nan
    df['x'] = df['x'].interpolate(method = 'polynomial', order = 2)
    df = df.ffill().bfill()
    B = df['x'].values
    return B
class trendAnalysis:
    def __init__(self, summary, burnish_results, output_name):
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
        
    def deleteBadBER(self):
        '''
        This function will deletr all bad BER --> BER cannot be zero

        Returns
        -------
        None.

        '''
        
        # sort the table 
        self.burnish_results = self.burnish_results.sort_values(by = ['SERIAL_NUM', 'HEAD'])
        
        # delete bad BER
        col = cfg.RAW_SIGNALS[cfg.CURRENT_AP]['BER']
        self.burnish_results = self.burnish_results[self.burnish_results[col]!=0]
        
        # get max RH stress time of the entire SBR
        if 'RH_STRESS_TIME_hrs' not in self.burnish_results.columns:
            self.burnish_results['RH_STRESS_TIME_hrs'] = self.burnish_results['RH_STRESS_TIME']/3600
        
    
    def checkDF(self):
        '''
        This function will check the DF to see if the file has been previously reduced to 1/8. 
        If not, it will reduce it.

        Returns
        -------
        None.

        '''
        
        #check if the dataframe has already been reduced
        self.burnish_results['MRE_check'] = self.burnish_results['MRE_RESISTANCE'] - self.burnish_results['MRE_RESISTANCE'].shift(1)
        self.burnish_results['EW_SNR_TOT_check'] = self.burnish_results['EW_SNR_ELEC'] - self.burnish_results['EW_SNR_ELEC'].shift(1)
        self.burnish_results['SN_HD'] = (self.burnish_results['SERIAL_NUM'] + '_' + self.burnish_results['HEAD'].astype(str))
        br = pd.DataFrame()
        for head in self.burnish_results['SN_HD'].unique():
            dt01_sn_hd = self.burnish_results[self.burnish_results['SN_HD'] == head]
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
                subset = self.burnish_results[self.burnish_results.index.isin(sequence)].copy()
                
            else:
                subset = dt01_sn_hd.copy()
            br = pd.concat([br, subset], ignore_index = True)
        
        self.burnish_results = br.copy()
            
        #save all the results in the dataframe
        self.burnish_results = self.burnish_results.drop(columns = ['SN_HD'])
       
    def cleanSignal(self):
        '''
        This function is used to clean the BER and SNRE signals by recognizing the jump and the sudden changes in the signal

        Returns
        -------
        None.

        '''
        #get deltas
        deltas = {'BER':[], 'SNRE':[]}
        
        
        for sbr_sn_hd in self.burnish_results['SBR_SN_HD'].unique().tolist():
            #filter the dataframe to only get the current SBR_SN_HD
            df_sbr_sn_hd = self.burnish_results[self.burnish_results['SBR_SN_HD'] == sbr_sn_hd]
            
            for i in range(len(self.entries)):
                if self.entries[i] in ['BER', 'SNRE']:
                    
                        #get the correct column name based on AP version
                        col = cfg.RAW_SIGNALS[cfg.CURRENT_AP][self.entries[i]]
                        x = df_sbr_sn_hd[col].values
                        
                        try:
                            #get gradients
                            grads = np.gradient(x)
                            cross = np.zeros_like(grads)
                            #check if the gradients are out of +/- 2 std dev
                            for j in range(len(grads)):
                                if (grads[j] > (np.mean(grads) + 3*np.std(grads))) | \
                                      (grads[j] < (np.mean(grads) - 3*np.std(grads))):
                                      cross[j] = grads[j]
                            #cumulatively sum all the crossing and subtract them from the original signal
                            cross = np.cumsum(cross)
                            x = x-cross
                            
                            peaks, props = find_peaks(x, height = np.mean(x) + 2*np.std(x), plateau_size = (0, 100))
                            neg_x = -x
                            peaks2, props2 = find_peaks(neg_x, height = np.mean(neg_x) + 2*np.std(neg_x), plateau_size = (0, 100))
                            peaks = [*peaks, *peaks2]
                            
                            x = fill_nan(x, peaks)
                            
                            
                        except:
                            
                            pass
                        
                        deltas[self.entries[i]] = [*deltas[self.entries[i]], *x]
           
        #store deltas
        for i in range(len(self.raw_signals)):
            if self.entries[i] in ['BER', 'SNRE']:
                self.burnish_results[cfg.RAW_SIGNALS[cfg.CURRENT_AP][self.entries[i]]] = deltas[self.entries[i]] 
            

        

    def getDeltas(self):
        '''
        This function gets differences of the current signal with the initial value and update the burnish_results dataframe.

        Returns
        -------
        None.

        '''
        
        
        #get SBR-SN-HD
        self.burnish_results['SN_HD'] = self.burnish_results['SERIAL_NUM'] + '_' + self.burnish_results['HEAD'].astype(str)
        self.burnish_results['SBR_SN_HD'] = self.burnish_results['SBR'] + '_' + self.burnish_results['SERIAL_NUM'] + '_' + self.burnish_results['HEAD'].astype(str)
        self.cleanSignal()
        #get deltas
        deltas = {'MRE': [], 'BER':[], 'PW50':[], 'VGAS':[], 'SNRE':[], 'ASYM':[]}
        
        for sbr_sn_hd in self.burnish_results['SBR_SN_HD'].unique().tolist():
            
            #filter the dataframe to only get the current SBR_SN_HD
            df_sbr_sn_hd = self.burnish_results[self.burnish_results['SBR_SN_HD'] == sbr_sn_hd]
            
            for i in range(len(self.entries)):
                
                #get the correct column name based on AP version
                col = cfg.RAW_SIGNALS[cfg.CURRENT_AP][self.entries[i]]
                
                #get % change if MRE, else get delta change
                if self.entries[i] == 'MRE':
                    
                      deltas[self.entries[i]] = [*deltas[self.entries[i]], \
                                         *(100*(df_sbr_sn_hd[col] - df_sbr_sn_hd[col].iloc[:3].mean())/(df_sbr_sn_hd[col].iloc[:3].mean()))]
                else:
                    
                    deltas[self.entries[i]] = [*deltas[self.entries[i]], *df_sbr_sn_hd[col] - df_sbr_sn_hd[col].iloc[:3].mean()]
            
        
        #store deltas
        for i in range(len(self.raw_signals)):
            #if self.raw_signals[i]!='DeltaMRE%':
            self.burnish_results[self.raw_signals[i]] = deltas[self.entries[i]] 
        
        
                
        
    def savgol(self, df_sbr_sn_hd, y, svg_amt = 0.2):
        '''
        This function is used to perform the Savitzky-Golay filtering

        Parameters
        ----------
        df_sbr_sn_hd : DataFrame
            This is the filtered SBR_SN_HD dataframe from total burnish results
        
        y : String
            The name of the column in the df_sbr_sn_hd which needs to be smoothed
        svg_amt : Float, optional
            DESCRIPTION. The default is 0.2. The percent of total length used to perform smoothing

        Returns
        -------
        w : list
            Smoothed column

        '''
        
        poly = 2
        flag = 0
        svg = np.ceil(svg_amt * len(df_sbr_sn_hd))
        
        #The SVG value has to be odd
        if svg%2 == 0:
            svg += 1
        
        #The SVG value has to be greater than the polyorder
        if svg<=2:
            svg = 3
        
        #The SVG value has to be less than or equal to the length of signal to be processed
        if svg>len(df_sbr_sn_hd):
            svg = len(df_sbr_sn_hd)
            if svg%2 == 0:
                svg -=1
            if svg<=2:
                flag = 1
        
        
        #if still an error occurs, use rolling mean
        if flag:
            return df_sbr_sn_hd[y].rolling(30).mean()
        w = savgol_filter(df_sbr_sn_hd[y].values, int(svg), poly)
        
        
        return w
    def smoothing(self):
        '''
        This function will use Savitzky-Golay filtering method to get a smoothed curve

        Returns
        -------
        None.

        '''

        #output signals to be saved
        smoothed = {'SG_MRE':[], 'SG_BER': [], 'SG_PW50':[], 'SG_VGAS':[], 'SG_SNRE': [], 'SG_ASYM':[]}
        
        
        #for every SBR_SN_HD
        for sn_hd in self.burnish_results['SN_HD'].unique().tolist():
            df_sbr_sn_hd = self.burnish_results[self.burnish_results['SN_HD'] == sn_hd]
            
            #smoothen each of the above signals and append results
            for i in range(len(self.raw_signals)):
                
                smoothed[list(smoothed)[i]] = [*smoothed[list(smoothed)[i]], *self.savgol(df_sbr_sn_hd, self.raw_signals[i])]
        
        #store the final smoothed results
        for i in smoothed:
            self.burnish_results[i] = smoothed[i]
        for sn_hd in self.burnish_results['SN_HD'].unique().tolist():
            df_sbr_sn_hd = self.burnish_results[self.burnish_results['SN_HD'] == sn_hd]
            self.getSTD(df_sbr_sn_hd)
        self.stddev = {}
        for i in range(len(self.entries)):
            self.stddev[self.entries[i]] = np.std(self.std_dev[self.entries[i]])
        del self.std_dev
    
    def skipSN(self, df_sbr_sn_hd, entry):
        '''
        skipSN will check if the total number of points are less <N_SKIP. If so, it will not process the dataframe further

        Parameters
        ----------
        df_sbr_sn_hd : dataframe that gives the filtered burnish_results for this particular SBR_SN_HD
        entry : entry is a dictionary with data regarding the SBR SN HD 

        Returns
        -------
        bool
        True if <N_SKIP points
        False otherwise

        '''
        
        if len(df_sbr_sn_hd)<cfg.N_SKIP:
            
            entry['CPD_comment'] = 'Less than N_SKIP '+str(cfg.N_SKIP) + ' data points'
            entry['CPD_InducTm_CENSOR'] = 1
            
            #save the file without continuing the analysis
            self.final_results = pd.concat([self.final_results, df_sbr_sn_hd], ignore_index=True)
            entry = pd.DataFrame(entry, index = [0])
            self.cluster_results = pd.concat([self.cluster_results, entry], ignore_index = True)
            return df_sbr_sn_hd, True
        return df_sbr_sn_hd, False
    
    def checkMissing(self, df_sbr_sn_hd, entry):
        '''
        Check if >N_MISSING rows. If so, skip analyzing.
        If not, reduce the dataframe to only have non-na values, and use that further for analyzing

        Parameters
        ----------
        df_sbr_sn_hd : Filtered dataframe with SBR-SN-HD
        entry : dictionary with all row entries for this particular SBR-SN-HD

        Returns
        -------
        None, True -> when more missing entries
        reduced_dataframe, False -> when less missing entries

        '''
        # check for missing rows > N_MISSING for MRE, BER, PW50, VGAS
        
        df_sbr_sn_hd_reduced = df_sbr_sn_hd.dropna(subset = ['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', \
                                            'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%', 'RH_STRESS_TIME_hrs']).reset_index(drop = True)
        
        
        if (len(df_sbr_sn_hd) - len(df_sbr_sn_hd_reduced)) > cfg.N_MISSING:
            
            entry['CPD_comment'] = 'Missing > N_MISSING Rows, do not analyze'
            entry['CPD_InducTm_CENSOR'] = 1
            
            #save the file without continuing the analysis
            self.final_results = pd.concat([self.final_results, df_sbr_sn_hd], ignore_index = True)
            entry = pd.DataFrame(entry, index = [0])
            self.cluster_results = pd.concat([self.cluster_results, entry], ignore_index = True)
            
            return None, True
        
        #forward fill the rows with the missing data
        df_sbr_sn_hd[['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', 'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%']] = \
            df_sbr_sn_hd[['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', 'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%']].fillna(method = 'bfill', axis = 0).fillna(method = 'ffill', axis = 0)
        
        df_sbr_sn_hd_reduced = df_sbr_sn_hd.dropna(subset = ['SG_MRE', 'SG_BER', 'SG_PW50', 'SG_VGAS', 'SG_SNRE', 'SG_ASYM', \
                                            'dBER', 'dVGAS', 'dPW50', 'dSNRE', 'dASYM','DeltaMRE%', 'RH_STRESS_TIME_hrs']).reset_index(drop = True)
        
        
        if (len(df_sbr_sn_hd) - len(df_sbr_sn_hd_reduced)) > 0:
            
            entry['CPD_comment'] = 'Missing > N_MISSING Rows, do not analyze'
            entry['CPD_InducTm_CENSOR'] = 1
            
            #save the file without continuing the analysis
            self.final_results = pd.concat([self.final_results, df_sbr_sn_hd], ignore_index = True)
            entry = pd.DataFrame(entry, index = [0])
            self.cluster_results = pd.concat([self.cluster_results, entry], ignore_index = True)
            
            return None, True
        
        return df_sbr_sn_hd, False
    
    def shuntCalculation(self, df_sbr_sn_hd):
        '''
        Does the shunt calculation for the SBR-SN-HD

        Parameters
        ----------
        df_sbr_sn_hd : Filtered DataFrame of the particular SBR-SN-HD

        Returns
        -------
        df_sbr_sn_hd : Updated dataframe with a new column called shunt_MRE
        shunt_mat : Filtered dataframe with only those row whose shunt_MRE >0

        '''
        # Point to Point shunt calculations
        df_sbr_sn_hd['shunt_MRE'] = np.where(((df_sbr_sn_hd['DeltaMRE%'].diff(1) > (cfg.SHUNT_DMRE * 0.01))) & (df_sbr_sn_hd['DeltaMRE%'].diff(2) > (cfg.SHUNT_DMRE * 0.01)) & 
        (df_sbr_sn_hd['DeltaMRE%'].diff(3) > (cfg.SHUNT_DMRE * 0.01)) & (df_sbr_sn_hd['DeltaMRE%'].shift(-1).diff(1) > (cfg.SHUNT_DMRE * 0.01)) &
        (df_sbr_sn_hd['DeltaMRE%'].shift(-1).diff(2) > (cfg.SHUNT_DMRE * 0.01)) & (df_sbr_sn_hd['DeltaMRE%'].shift(-1).diff(3) > (cfg.SHUNT_DMRE * 0.01)) &
        (df_sbr_sn_hd['DeltaMRE%'].shift(-2).diff(1) > (cfg.SHUNT_DMRE * 0.01)) & (df_sbr_sn_hd['DeltaMRE%'].shift(-2).diff(2) > (cfg.SHUNT_DMRE * 0.01)) &
        (df_sbr_sn_hd['DeltaMRE%'].shift(-2).diff(3) > (cfg.SHUNT_DMRE * 0.01)) & ((df_sbr_sn_hd['DeltaMRE%'].diff(-1) > -(cfg.SHUNT_DMRE * 0.01)) & 
        (df_sbr_sn_hd['DeltaMRE%'].diff(-2) > -(cfg.SHUNT_DMRE * 0.01)) & (df_sbr_sn_hd['DeltaMRE%'].diff(-3) > -(cfg.SHUNT_DMRE * 0.01))) & (
        (df_sbr_sn_hd['VGAS_DB'].diff(-1) < cfg.SHUNT_DVGA) | (df_sbr_sn_hd['VGAS_DB'].diff(-2) < cfg.SHUNT_DVGA) |
        (df_sbr_sn_hd['DeltaMRE%'].diff(-3) < cfg.SHUNT_DVGA)), 1, 0)
        shunt_mat = df_sbr_sn_hd[df_sbr_sn_hd['shunt_MRE']>0]   
        return df_sbr_sn_hd, shunt_mat      
    
    def commentsPart(self, entry, slp_name, cmt_name):
        '''
        Get comments for this particular part

        Parameters
        ----------
        entry : dictionary with current SBR-SN results
        part_number : the part from the plots to analyze

        Returns
        -------
        entry : modified dictionary with part comments

        '''
        
        #get comment values based on slopes
        ideg = 0
        ipv = 0
        iflat = 0
        imp = 0
        #Rev 051: BER, P50, VGAS deltas in final (or only) cluster must exceed critical_deltas
        rev051 = ['BER', 'PW50', 'VGAS']
        #label single slope as flat, improving or degrading based on slope and magnitude of change
        for i in self.entries:
            try:
                
                if i not in ['SNRE', 'ASYM']:
                    #for MRE, BER, PW50 and VGAS
                    if np.absolute(entry['CPD_' + i + slp_name]) <= cfg.overall_dict[i]:
                        entry['CPD_' + i+ cmt_name] = 'flat'
                        if i in rev051:
                            iflat += 1
                        
                    elif (entry['CPD_' + i + slp_name] > cfg.overall_dict[i]):
                        entry['CPD_' + i + cmt_name] = 'degrading'
                        if i in rev051:
                            ideg += 1
                        if i in ['PW50', 'VGAS']:
                            ipv += 1
                    else:
                            entry['CPD_' + i + cmt_name] = 'improving'
                            if i in rev051:
                                imp += 1
                    
                        
                elif i not in ['SNRE']:
                    #for ASYM
                    if np.absolute(entry['CPD_' + i + slp_name]) <= cfg.overall_dict[i]:
                        entry['CPD_' + i+ cmt_name] = 'flat'
                    elif entry['CPD_' + i + slp_name] > cfg.overall_dict[i]:
                        entry['CPD_' + i + cmt_name] = 'degrading'
                    else:
                        entry['CPD_' + i + cmt_name] = 'degrading'
                else:
                    #for SNRE
                    if np.absolute(entry['CPD_' + i + slp_name]) <= cfg.overall_dict['SNRE']:
                        entry['CPD_' + i + cmt_name] = 'flat'
                    elif entry['CPD_' + i + slp_name] > cfg.overall_dict['SNRE']:
                        entry['CPD_' + i + cmt_name] = 'improving'
                    
                    else:
                        entry['CPD_' + i + cmt_name] = 'degrading'
                    
                
            except:
                entry['CPD_' + i + cmt_name] = '' 
        
        #rules for this part
        if (ideg == 3) | ((entry['CPD_MRE' + cmt_name] == 'degrading') & (entry['CPD_BER' + cmt_name] == 'degrading') & (ipv >= 1)):
            entry['CPD_' + cmt_name + ' comment'] = 'degrading'

        elif iflat == 3:
            entry['CPD_' + cmt_name + ' comment'] = 'flat'
        
        elif imp == 3:
            entry['CPD_' + cmt_name + ' comment'] = 'improving'
        else:
            entry['CPD_' + cmt_name + ' comment'] = ''
        return entry
    
    def getSTD(self, df_sbr_sn_hd, entry = None):
        '''
        This function will find the standard deviation of the noise in the original signal

        Parameters
        ----------
        df_sbr_sn_hd : DataFrame
                The filtered dataframe of the current SBR-SN-HD
        entry : Dictionary
                The current updated metrics for the SBR-SN-HD

        Returns
        -------
        entry : Dictionary
                The dictionary updated with new metrics.

        '''
        if entry is None:
           for i in range(len(self.entries)):
               self.std_dev[self.entries[i]].append(np.std(df_sbr_sn_hd[self.raw_signals[i]].values - \
                                                             df_sbr_sn_hd[self.smoothed[i]].values))
        else:
        
            for i in range(len(self.entries)):
                entry['CPD_' + self.entries[i] + ' St. Dev.'] = np.std(df_sbr_sn_hd[self.raw_signals[i]].values - \
                                                              df_sbr_sn_hd[self.smoothed[i]].values)
            return entry
        
    def getLR(self, df_sbr_sn_hd, entry):
        '''
        If the number of elements in the head are less than 100, fit a linear regression line

        Parameters
        ----------
        df_sbr_sn_hd : The filtered dataframe of the current SBR-SN-HD
        entry : Dictionary
                The current updated metrics for the SBR-SN-HD

        Returns
        -------
        df_sbr_sn_hd : Dataframe
            The updated dataframe of the current SBR-SN-HD
        entry : Dictionary
            The dictionary updated with new metrics.

        '''
        #get TTF values, the actual ones and the ones used for analysis
        self.cuts, self.cuts_analysis = self.getTTF(df_sbr_sn_hd, entry)
        for j in range(len(self.smoothed)):
            
            # fit a linear regression with y as the smoothed signal
            subset = df_sbr_sn_hd.copy()
            subset = subset.dropna(subset = self.smoothed[j])
            
            lr = LinearRegression().fit(subset['RH_STRESS_TIME_hrs'].values.reshape(-1,1), subset[self.smoothed[j]].values.reshape(-1,1))
            
            #save the intercept and slopes
            entry['CPD_' + self.entries[j] + ' c1'] = lr.intercept_[0]
            
            
            entry['CPD_' + self.entries[j] + ' slope'] = lr.coef_[0,0]
            
            #get predictions and save the
            subset[self.entries[j] + 'fit'] = lr.predict(subset['RH_STRESS_TIME_hrs'].values.reshape(-1,1))

            subset = subset[['RH_STRESS_TIME_hrs', self.entries[j] + 'fit']]
            df_sbr_sn_hd = df_sbr_sn_hd.merge(subset, on = 'RH_STRESS_TIME_hrs', how = 'left')
        
        # get comments only for the InducTm to TTF part of the signal
        entry = self.commentsPart(entry, ' slope', ' InducTm to TTF')  
        #get the InducTm and the ChangePoint for all the signals
        for i in range(len(self.entries)):
            if entry['CPD_' + self.entries[i] + ' InducTm to TTF'] == 'degrading':
                entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()
                entry['CPD_' + self.entries[i] + '_ChangePoint'] = np.nan
            else:
                entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                entry['CPD_' + self.entries[i] + '_ChangePoint'] = np.nan
        entry['CPD_comment'] = entry['CPD_ InducTm to TTF comment']
        
        #change the censor based on the comment
        if entry['CPD_ InducTm to TTF comment'] == 'degrading':
            
            entry['CPD_InducTm_CENSOR'] = 0
        else:
            
            entry['CPD_InducTm_CENSOR'] = 1
        return df_sbr_sn_hd, entry
        
        
    def getChangepoints(self, df_sbr_sn_hd, entry):
        '''
        Get the changepoints in every curve

        Parameters
        ----------
        df_sbr_sn_hd : Filtered DataFrame of the particular SBR-SN-HD
        entry : dictionary with current SBR-SN results

        Returns
        -------
        None.

        '''
        
        #get TTF values, the actual ones and the ones used for analysis
        self.cuts, self.cuts_analysis = self.getTTF(df_sbr_sn_hd, entry)
        
        #get changepoints
        inductms = []
        for j in range(len(self.smoothed)): 
            
            #check if TTF is crossing: if so, induction time should only occur before the TTF time, 
            # if not use 90% of the total data, the last 10% has a lot of noise
            if self.cuts_analysis[self.linear_fit[j]]:
                subset = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']<entry[self.entries[j] + '_TTF']].copy()
            else:
                subset = df_sbr_sn_hd
            
            #handle the condition when the signal starts by crossing the TTF
            if len(subset) == 0:
                entry['CPD_' + self.entries[j] + '_InducTm'] = entry[self.entries[j] + '_TTF']
                entry['CPD_' + self.entries[j] + '_ChangePoint'] = np.nan
            else:
                subset = subset.reset_index(drop = True)
            
                # #get the changepoint
                try:
                    model = rpt.Dynp(model = 'clinear', jump = 3, min_size = 0.15*len(df_sbr_sn_hd)).fit(subset[self.smoothed[j]].values)
                    if self.smoothed[j] == 'SG_ASYM':
                        ts = model.predict(n_bkps = 1)[0]
                    else:
                        try:
                            ts1 = model.predict(n_bkps = 1)[0]
                            x = ['RH_STRESS_TIME_hrs']
                            subset = df_sbr_sn_hd.copy()
                            subset['newx0'] = (subset['RH_STRESS_TIME_hrs'].values - ts1)* np.where(subset['RH_STRESS_TIME_hrs'].values < ts1, 0, 1)
                            x.append('newx0')
                            subset = subset.reset_index(drop = True)
                            
                            # fit a linear regression with y as the smoothed signal
                            
                            lr = LinearRegression().fit(subset[x].values, subset[self.smoothed[j]].values.reshape(-1,1))
                            
                            
                        
                            slopes=(lr.coef_[0, 0] + lr.coef_[0,1])
                            
                            
                                
                                
                            #if the slopes are not degrading enough, use the 1 breakpoint method
                            if self.smoothed[j] == 'SG_SNRE':
                                
                                if (not(slopes<-cfg.overall_dict['SNRE'])) & ((subset['RH_STRESS_TIME_hrs'].max())<1400):
                                    
                                    #get 2 breakpoints, and choose the breakpoint which has a degrading slope
                                    
                                    ts = model.predict(n_bkps = 2)[:2]
                                    
                                    slopes = []
                                    for k in range(len(ts)):
                                        x = ['RH_STRESS_TIME_hrs']
                                        subset = df_sbr_sn_hd.copy()
                                        subset['newx'+str(k)] = (subset['RH_STRESS_TIME_hrs'].values - ts[k])* np.where(subset['RH_STRESS_TIME_hrs'].values < ts[k], 0, 1)
                                        x.append('newx' + str(k))
                                        subset = subset.reset_index(drop = True)
                                        
                                        # fit a linear regression with y as the smoothed signal
                                        
                                        lr = LinearRegression().fit(subset[x].values, subset[self.smoothed[j]].values.reshape(-1,1))
                                        
                                        
                                    
                                        slopes.append(lr.coef_[0, 0] + lr.coef_[0,1])
                                    idx = np.argmin(slopes)
                                    if (slopes[idx]<-cfg.overall_dict['SNRE']):
                                       ts = ts[idx] 
                                    else:
                                        ts = ts1
                                else:
                                    ts = ts1
                                
                                
                            
                            else:
                                
                                if (not(slopes>cfg.overall_dict[self.entries[j]])) & ((subset['RH_STRESS_TIME_hrs'].max())<1400):
                                    
                                    #get 2 breakpoints, and choose the breakpoint which has a degrading slope
                                    
                                    ts = model.predict(n_bkps = 2)[:2]
                                    
                                    slopes = []
                                    for k in range(len(ts)):
                                        x = ['RH_STRESS_TIME_hrs']
                                        subset = df_sbr_sn_hd.copy()
                                        subset['newx'+str(k)] = (subset['RH_STRESS_TIME_hrs'].values - ts[k])* np.where(subset['RH_STRESS_TIME_hrs'].values < ts[k], 0, 1)
                                        x.append('newx' + str(k))
                                        subset = subset.reset_index(drop = True)
                                        
                                        # fit a linear regression with y as the smoothed signal
                                        
                                        lr = LinearRegression().fit(subset[x].values, subset[self.smoothed[j]].values.reshape(-1,1))
                                        
                                        
                                    
                                        slopes.append(lr.coef_[0, 0] + lr.coef_[0,1])
                                    idx = np.argmax(slopes)
                                    if (slopes[idx]>cfg.overall_dict[self.entries[j]]):
                                       ts = ts[idx]
                                       
                                    else:
                                        ts = ts1
                                    
                                else:
                                    ts = ts1
                        except:
                            
                            ts = model.predict(n_bkps = 1)[0]
                            
                
                # # if no changepoint detected, put the inductime at the beginning
                except:
                    ts = 1
                    
                    entry['CPD_' + self.entries[j] + '_ChangePoint'] = np.nan
                
            #add all induction times
            entry['CPD_' + self.entries[j] + '_InducTm'] = subset['RH_STRESS_TIME_hrs'].iloc[ts-1]
            if 'CPD_' + self.entries[j] + '_ChangePoint' not in entry.keys():
                entry['CPD_' + self.entries[j] + '_ChangePoint'] = subset['RH_STRESS_TIME_hrs'].iloc[ts-1]
            inductms.append(subset['RH_STRESS_TIME_hrs'].iloc[ts-1])
                
                
        inductms = sorted(inductms)
        
        #the smallest induction time is set aside as the overall induction time.
        entry['CPD_InducTm'] = inductms[0]
        
        return entry, df_sbr_sn_hd

    def getPieceWiseResults(self, df_sbr_sn_hd, entry):
        '''
        This function will compute PWLF and evaluate the overall trends of the curves

        Parameters
        ----------
        df_sbr_sn_hd : Filtered DataFrame of the particular SBR-SN-HD
        x : list of changepoints
        entry : dictionary with current SBR-SN results


        Returns
        -------
        entry : updated dictionary with current SBR-SN results
        df_sbr_sn_hd : Updated DataFrame of the particular SBR-SN-HD 
        

        '''
        
        # Perform piecewise linear fit
        
        for i in range(len(self.smoothed)):
            x = ['RH_STRESS_TIME_hrs', 'newx0']
            subset = df_sbr_sn_hd.copy()
           
            # for the X-axis, use the RH_STRESS_TIME_hrs, 
            # x0-> 0 when less than induction time, RH_STRESS_TIME_hrs between induction time and TTF
            # x1 -> 0 when less than TTF time, RH_STRESS_TIME_hrs between TTF and end of time
            
            subset['newx0'] = (subset['RH_STRESS_TIME_hrs'].values - entry['CPD_' + self.entries[i] + '_InducTm'])* np.where(subset['RH_STRESS_TIME_hrs'].values < entry['CPD_' + self.entries[i] + '_InducTm'], 0, 1)
            if (self.cuts_analysis[self.linear_fit[i]]) & (entry[self.entries[i] + '_TTF']!=df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()):
                    subset['newx1'] = (subset['RH_STRESS_TIME_hrs'].values - entry[self.entries[i] + '_TTF'])* np.where(subset['RH_STRESS_TIME_hrs'].values < entry[self.entries[i] + '_TTF'], 0, 1)
                    x.append('newx1')
                    
                
            subset = subset.reset_index(drop = True)
            
            # fit a linear regression with y as the smoothed signal
            
            lr = LinearRegression().fit(subset[x].values, subset[self.smoothed[i]].values.reshape(-1,1))
            
            #save the intercept and slopes
            entry['CPD_' + self.entries[i] + ' c1'] = lr.intercept_[0]
            
            
            entry['CPD_' + self.entries[i] + ' slope pre induct'] = lr.coef_[0,0]
            entry['CPD_' + self.entries[i] + ' slope'] = lr.coef_[0,1] + entry['CPD_' + self.entries[i] + ' slope pre induct']
            if len(x)>2:
                entry['CPD_' + self.entries[i] + ' slope end'] = lr.coef_[0, 2] + entry['CPD_' + self.entries[i] + ' slope']
                        
            #get predictions and save the
            subset[self.entries[i] + 'fit'] = lr.predict(subset[x].values)
            
            #check if the signal can be degrading or not by chekcing difference between first and last element
            
            
            
            subset = subset[['RH_STRESS_TIME_hrs', self.entries[i] + 'fit']]
            df_sbr_sn_hd = df_sbr_sn_hd.merge(subset, on = 'RH_STRESS_TIME_hrs', how = 'left')
        
        return entry, df_sbr_sn_hd
    
    def getNegativeSlope(self, df_sbr_sn_hd, entry):
        '''
        This function checks for the intial clearance drops 

        Parameters
        ----------
        df_sbr_sn_hd : DataFrame
            Filtered dataframe of the current SBR-SN-HD
        entry : dictionary
            dictionary with current SBR-SN results

        Returns
        -------
        entry : dictionary
            updated dictionary with current SBR-SN results

        '''
        
       
        #filter a subset to only have STRESS_TIME less than 50 hrs
        subset = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']<50]
        
        subset = subset.reset_index(drop = True)
        #input signals to be processed
        signals_process = ['dBER', 'dPW50', 'dSNRE']
        #output signals to be saved
        smoothed = {'SG_BER': [], 'SG_PW50':[], 'SG_SNRE': []}
        keys_neg = ['initial_neg_slope_BER', 'initial_neg_slope_PW50', 'initial_neg_slope_SNRE']
        neg_values = {}
        #smoothen each of the above signals and append results
        for i in range(len(signals_process)):
            try:
                #make the smoothing a little less stiff
                smoothed[list(smoothed)[i]] = [*smoothed[list(smoothed)[i]], *subset[signals_process[i]].rolling(3).mean()]
                #get the gradients
                neg_slopes = np.gradient(smoothed[list(smoothed)[i]])
                
                if signals_process != 'dSNRE':
                    neg_slopes = np.argwhere(-neg_slopes > 0).flatten()
                max_count = 0
                ending_val = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                try:
                    #get a double gradient to get trend in the signals
                    double_grad = np.gradient(neg_slopes)
                except ValueError:
                    return entry
                cnt = 0
                
                #check if initial negative slope valid. If so, save it
                for j in range(len(double_grad)):
                    if double_grad[j]!= 1:
                        cnt = 0
                        break
                    else:
        
                        cnt += 1
                        
                        if cnt > max_count:
                            max_count = cnt
                            ending_val = subset['RH_STRESS_TIME_hrs'].iloc[j]
                
                if (max_count > 4):
                    neg_values[keys_neg[i]] = ending_val
            except:
                pass
        if len(neg_values)>=2:
            entry.update(neg_values)
        return entry
            
            
        
       
        
    def checkDegrading(self, entry, df_sbr_sn_hd):
        '''
        This function checks if the signal is degrading before and after InducTm. 
        If so make the InducTm at the beginning

        Parameters
        ----------
        entry : dictionary with current SBR-SN results
        df_sbr_sn_hd : Filtered DataFrame of the current SBR-SN-HD

        Returns
        -------
        entry : Updated dictionary of the current SBR-SN

        '''
        
        for i in range(len(self.entries)):
            entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] = 'NO'
            #check if the drive is an infant fail
            if (entry['CPD_' + self.entries[i] + ' pre-InducTm'] == 'degrading') & (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] == 'degrading'):
                    #save the previous induction time
                    
                    entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[0]
                    entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] = 'YES'
            
            #check to see if the signal is first degrading and then improving/flat. If so put induction time at end
            elif (entry['CPD_' + self.entries[i] + ' pre-InducTm'] == 'degrading') & (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] in ['flat', 'improving', '']):
                    
                    
                    if entry['CPD_' + self.entries[i] + ' TTF to end'] in ['degrading']:
                        entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] = 'YES'
                        entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[0]
                        entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'degrading'
                        
                    elif entry['CPD_' + self.entries[i] + ' TTF to end'] not in entry.keys():
                        pre_induc = len(df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']<entry['CPD_' + self.entries[i] + '_InducTm']])
                        induc_ttf = len(df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']>entry['CPD_' + self.entries[i] + '_InducTm']])
                        if pre_induc > (0.75*induc_ttf):
                            try:
                                if self.entries[i] not in ['ASYM', 'SNRE']:
                                    if ((df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs'] ==entry['CPD_' + self.entries[i] + '_ChangePoint']][self.smoothed[i]].iloc[0])-\
                                        df_sbr_sn_hd[self.smoothed[i]].iloc[0])>cfg.degrad_thresholds[self.entries[i]]:
                                        entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = entry['CPD_' + self.entries[i] + ' pre-InducTm']
                                        entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] = 'YES'
                                        entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[0]
                                    else:
                                        entry['CPD_' + self.entries[i] + '_InducTm'] =df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[-1]
                                elif self.entries[i] == 'SNRE':
                                    if ((df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs'] ==entry['CPD_' + self.entries[i] + '_ChangePoint']][self.smoothed[i]].iloc[0])-\
                                        df_sbr_sn_hd[self.smoothed[i]].iloc[0])<cfg.degrad_thresholds[self.entries[i]]:
                                        entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = entry['CPD_' + self.entries[i] + ' pre-InducTm']
                                        entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] = 'YES'
                                        entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[0]
                                    else:
                                        entry['CPD_' + self.entries[i] + '_InducTm'] =df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[-1]
                            except:
                                entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[-1]
                                pass
                            
                            
                    
                    else:
                        
                        entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[-1]
                        
            
            #check if the signal is never degrading. If so, the induction time should be at the end
            elif (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] in['flat', 'improving', '']):
                if entry['CPD_' + self.entries[i] + ' TTF to end'] in ['degrading']:
                   entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_' + self.entries[i] + '_TTF']
                   entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'degrading'
                   if entry['CPD_InducTm']>entry['CPD_' + self.entries[i] + '_InducTm']:
                      entry['CPD_InducTm'] = entry['CPD_' + self.entries[i] + '_InducTm']
                else:
                    
                    entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values[-1]
            
        
        
                
                
        return entry
        
    def commonRules(self, entry, df_sbr_sn_hd):
        '''
        This function tries to find the maximum number of comments applied and uses that as the overall comment.
        Needs to be updated

        Parameters
        ----------
        entry : dictionary with current SBR-SN results

        Returns
        -------
        entry : updated dictionary with current SBR-SN results
        
        '''       
        
        entry = self.checkDegrading(entry, df_sbr_sn_hd)
       
        #check to see if 4/5 signals are degrading
        if ((entry['CPD_VGAS InducTm to TTF'] == 'degrading')|(entry['CPD_PW50 InducTm to TTF'] == 'degrading')) & \
            ((entry['CPD_BER InducTm to TTF'] == 'degrading')|(entry['CPD_SNRE InducTm to TTF'] == 'degrading')):
                entry['CPD_comment'] = 'degrading'
                
            
        #check to see if 4/5 signals are flat
        elif np.sum((entry['CPD_VGAS InducTm to TTF'] == 'flat')+(entry['CPD_PW50 InducTm to TTF'] == 'flat')+(entry['CPD_BER InducTm to TTF'] == 'flat')+\
            (entry['CPD_SNRE InducTm to TTF'] == 'flat') + (entry['CPD_MRE InducTm to TTF'] == 'flat')) >=4:
            entry['CPD_comment'] = 'plateau'
        #check to see if 4/5 signals are improving
        elif np.sum((entry['CPD_VGAS InducTm to TTF'] == 'improving')+(entry['CPD_PW50 InducTm to TTF'] == 'improving')+(entry['CPD_BER InducTm to TTF'] == 'improving')+\
            (entry['CPD_SNRE InducTm to TTF'] == 'improving') + (entry['CPD_MRE InducTm to TTF'] == 'improving')) >=4:
            entry['CPD_comment'] = 'improving'
        
        else:
            entry['CPD_comment'] = ''
        #count the total number of degrading signals   
        degrad = np.sum((entry['CPD_VGAS InducTm to TTF'] == 'degrading')+(entry['CPD_PW50 InducTm to TTF'] == 'degrading')+\
                    (entry['CPD_BER InducTm to TTF'] == 'degrading')+(entry['CPD_SNRE InducTm to TTF'] == 'degrading') +\
                        (entry['CPD_MRE InducTm to TTF'] == 'degrading'))  
        entry['CPD_degrad'] = degrad
        
        if (degrad == 4) & (np.sum((entry['CPD_VGAS InducTm to TTF'] == 'improving')+(entry['CPD_PW50 InducTm to TTF'] == 'improving')+(entry['CPD_BER InducTm to TTF'] == 'improving')+\
            (entry['CPD_SNRE InducTm to TTF'] == 'improving') + (entry['CPD_MRE InducTm to TTF'] == 'improving')) == 1):
                entry['ABNORMAL'] = 1
        
        
        if (((self.cuts['MREfit'] == True) + (self.cuts['BERfit'] == True) + (self.cuts['PW50fit'] == True) + (self.cuts['VGASfit'] == True) + (self.cuts['SNREfit'] == True))\
            >= 4) & ((entry['CPD_comment']=='plateau') | (entry['CPD_comment']=='')):
                entry['CPD_comment'] = 'degrading'
                
        
        if (entry['CPD_comment'] == 'plateau') | (entry['CPD_comment'] == 'improving') | (entry['CPD_comment'] == ''):
            entry['CPD_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
        
        
        if (entry['CPD_comment'] == 'degrading'):
            entry['CPD_InducTm_CENSOR'] = 0
        else:
            entry['CPD_InducTm_CENSOR'] = 1
            
        #check to see if shunt
        if (df_sbr_sn_hd['DeltaMRE%']>3).any():
            if (entry['CPD_MRE InducTm to TTF'] == 'degrading') & (entry['CPD_VGAS InducTm to TTF'] == 'improving'):
                
                    entry['CPD_shunt'] = 'SHUNT'
                    entry['CPD_SHUNT_time'] = entry['CPD_MRE_InducTm']
            elif (entry['CPD_MRE pre-InducTm'] == 'degrading') & (entry['CPD_VGAS pre-InducTm'] == 'improving'):
                
                    entry['CPD_shunt'] = 'SHUNT'
                    entry['CPD_SHUNT_time'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()
            elif (entry['CPD_MRE_InducTm'] == df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()) & (entry['CPD_MRE InducTm to TTF'] == 'degrading') &\
                (entry['CPD_VGAS pre-InducTm'] == 'improving'):
                
                    entry['CPD_shunt'] = 'SHUNT'
                    entry['CPD_SHUNT_time'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()
            elif (entry['CPD_MRE TTF to end'] == 'degrading') & (entry['CPD_VGAS TTF to end'] == 'improving'):
                
                    entry['CPD_shunt'] = 'SHUNT after oxidation'
                    entry['CPD_SHUNT_time'] = entry['CPD_MRE_TTF']
            else:
                shunt_df = pd.DataFrame()
                shunt_df['RH_STRESS_TIME_hrs'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].copy()
                shunt_df['SG_MRE'] = df_sbr_sn_hd['SG_MRE'].copy()
                shunt_df['SG_VGAS'] = df_sbr_sn_hd['SG_VGAS'].copy()
                shunt_df['MRE_grad'] = np.gradient(df_sbr_sn_hd['SG_MRE'])
                shunt_df['VGAS_grad'] = np.gradient(df_sbr_sn_hd['SG_VGAS'])
                
                shunt_df = shunt_df[(shunt_df['MRE_grad']>0) & (shunt_df['VGAS_grad']<0)]
                shunt_df = shunt_df.reset_index(drop = True)
                if len(shunt_df)>0:
                    try:
                        ##get longest consecutive list
                        consecutive = np.gradient(shunt_df.index)
                        ind = 0
                        ideal_ind = 0
                        ideal_cnt = 0
                        cnt = 0
                        flag = 0
                        for i in range(len(consecutive)):
                            if consecutive[i] == 1:
                                cnt += 1
                                if flag == 0:
                                    flag = 1
                                    ind = i
                            else:
                                if cnt> ideal_cnt:
                                    ideal_cnt = cnt
                                    ideal_ind = ind
                                cnt = 0
                                flag = 0
                    
                        if (ideal_cnt>0) & ((shunt_df['SG_MRE'].iloc[ideal_ind + ideal_cnt - 1]-shunt_df['SG_MRE'].iloc[ideal_ind])>=1):
                            
                            if shunt_df['RH_STRESS_TIME_hrs'].iloc[ideal_ind]< entry['MRE_TTF']:
                                entry['CPD_shunt'] = 'SHUNT after oxidation'
                            else:
                                entry['CPD_shunt'] = 'SHUNT'
                            
                            entry['CPD_SHUNT_time'] = shunt_df['RH_STRESS_TIME_hrs'].iloc[ideal_ind]
                    except:
                        pass
                    
                     
                                
                            
                    
                
            
        
        
        
        
        return entry
    
    def getTTF(self, df_sbr_sn_hd, entry):
        '''
        This function is responsible to get the TTF time for the given SBR SN drive. 

        Parameters
        ----------
        df_sbr_sn_hd : dataframe
            Filtered dataframe of the curretn SBR-SN-HD
        entry : dictionary
            Dictionary with current SBR-SN results

        Returns
        -------
        cuts : TTF cuts dictionary for current SBR-SN
        cuts_analysis : TTF + 20hrs cuts dictionary for current SBR-SN used for analysis

        '''
        #default dictionary for TTF calculation
        cuts = {'MREfit': False, 'BERfit': False, 'VGASfit': False, 'PW50fit': False, 'SNREfit':False, \
                  'ASYMfit':False}
        cuts_analysis = {'MREfit': False, 'BERfit': False, 'VGASfit': False, 'PW50fit': False, 'SNREfit':False, \
                  'ASYMfit':False}
        ttf = []
        
        
        for i in range(len(self.smoothed)):
            delta = 0
            #get the first time when the signal crosses TTF thresholding
            
            # if entry['CPD_' + self.entries[i] + ' St. Dev.'] > 3*self.stddev[self.entries[i]]:
                
                
            #     entry['CPD_' + self.entries[i] + '_TTF_val'] =  delta = entry['CPD_' + self.entries[i] + ' St. Dev.']*cfg.cutoff[self.linear_fit[i]]/(3*self.stddev[self.entries[i]])
            
            #     if self.entries[i] == 'SNRE':
            #         cut = df_sbr_sn_hd[df_sbr_sn_hd[self.smoothed[i]] <= delta]
            #     elif self.entries[i] == 'ASYM':
            #         cut = df_sbr_sn_hd[np.abs(df_sbr_sn_hd[self.smoothed[i]]) >= delta]
            #     else:
            #         cut = df_sbr_sn_hd[df_sbr_sn_hd[self.smoothed[i]] >= delta]
                    
            # else:
            if self.entries[i] == 'SNRE':
                cut = df_sbr_sn_hd[df_sbr_sn_hd[self.smoothed[i]] <= (cfg.cutoff[self.linear_fit[i]])]
            elif self.entries[i] == 'ASYM':
                cut = df_sbr_sn_hd[np.abs(df_sbr_sn_hd[self.smoothed[i]]) >= (cfg.cutoff[self.linear_fit[i]])]
            else:
                cut = df_sbr_sn_hd[df_sbr_sn_hd[self.smoothed[i]] >= (cfg.cutoff[self.linear_fit[i]])] 
                
            #also get +20 hours after TTF for analysis
            cut_actual = cut['RH_STRESS_TIME_hrs'].min()
            cut = cut[cut['RH_STRESS_TIME_hrs'] > (cut['RH_STRESS_TIME_hrs'].min() + 20)]['RH_STRESS_TIME_hrs'].min()
        
            if not(math.isnan(cut)):
                cuts_analysis[self.linear_fit[i]] = True
            if not(math.isnan(cut_actual)):
                cuts[self.linear_fit[i]] = True
                entry[self.entries[i] + '_TTF'] = cut
                entry['CPD_'+self.entries[i] + '_TTF'] = cut_actual
                entry['CPD_' + self.entries[i] + ' TTF_CENSOR'] = 0
    
            else:
                entry['CPD_'+self.entries[i] + '_TTF'] = entry[self.entries[i] + '_TTF'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                entry['CPD_' + self.entries[i] + ' TTF_CENSOR'] = 1
            if self.entries[i]!= 'ASYM':
                ttf.append(entry['CPD_' + self.entries[i] + '_TTF'])
        entry['CPD_TTF'] = sorted(ttf)[0]
        
        return cuts, cuts_analysis
    
    def checkCensor(self, df_sbr_sn_hd, entry):
        '''
        This function will get rid of all the 3 different slopes and instead replace it with only 1 slope for
        pass cases. 

        Parameters
        ----------
        df_sbr_sn_hd : DataFrame
                    The filtered dataframe of the current SBR-SN-HD
        entry : Dictionary
                Dictionary with current SBR-SN analysis results

        Returns
        -------
        entry : Dictionary
                Updated dictionary with current SBR-SN analysis results

        '''
        if entry['CPD_InducTm_CENSOR']:
            for i in range(len(self.smoothed)):
                x = ['RH_STRESS_TIME_hrs']
                subset = df_sbr_sn_hd.copy()
    
                # fit a linear regression with y as the smoothed signal
    
                lr = LinearRegression().fit(subset[x].values.reshape(-1,1), subset[self.smoothed[i]].values.reshape(-1,1))
    
                #save the intercept and slopes
                entry['CPD_' + self.entries[i] + ' slope'] = lr.coef_[0]
                try:
                    del entry['CPD_' + self.entries[i] + ' slope pre induct']
                except:
                    pass
                try:
                    del entry['CPD_' + self.entries[i] + ' slope end']
                except:
                    pass
        inductms, ttf, cpd = [], [], []
        for i in range(len(self.entries)):
            if self.entries[i]!='ASYM':
                inductms.append(entry['CPD_' + self.entries[i] + '_InducTm'])
                ttf.append(entry['CPD_' + self.entries[i] + '_TTF'])
                try:
                    cpd.append(entry['CPD_' + self.entries[i] + '_ChangePoint'])
                except:
                    pass
            
        entry['CPD_Min. InducTm'] = self.entries[np.argmin(inductms)]
        entry['CPD_Min. TTF'] = self.entries[np.argmin(ttf)]
        entry['CPD_ChangePoint'] = self.entries[np.argmin(cpd)]
        return entry
    
    def checkFI(self, df_sbr_sn_hd, entry):
        '''
        If the signal is flat or improving, get the updated 1 slope for the entire signal

        Parameters
        ----------
        df_sbr_sn_hd : DataFrame
                    The filtered dataframe of the current SBR-SN-HD
        entry : Dictionary
                Dictionary with all updated metrics for the current SBR-SN-HD

        Returns
        -------
        df_sbr_sn_hd : DataFrame
                    The updated dataframe of the current SBR-SN-HD with the change linearfit data
        entry : Dictionary
                The updated dictionary of the current SBR-SN-HD with the changed slope metrics.

        '''
        #check if flat/improving
        flatness = []
        if entry['CPD_comment'] in ['flat', 'improving', '']:
            for i in range(len(self.smoothed)):
                if self.smoothed[i]!='SG_ASYM':
                    if (((self.smoothed[i] == 'SG_SNRE') & (entry['CPD_' + self.entries[i] + ' slope'] >= 0)) | \
                        ((self.smoothed[i] != 'SG_SNRE') & (entry['CPD_' + self.entries[i] + ' slope'] <= 0))) & \
                        (entry['CPD_' + self.entries[i] + ' pre-InducTm'] in ['flat', 'improving', '']) & \
                        (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] in ['flat', 'improving', '']) & \
                         (entry['CPD_' + self.entries[i] + ' TTF to end'] in ['flat', 'improving', '']):
                            entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()
                            flatness.append(self.entries[i])
                df_sbr_sn_hd = df_sbr_sn_hd.drop(columns = self.entries[i] + 'fit')
            entry, df_sbr_sn_hd = self.getPieceWiseResults(df_sbr_sn_hd, entry)
            for i in range(len(flatness)):
                entry['CPD_' + flatness[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
            if len(flatness)>2:
                entry['CPD_comment'] = ''
                entry['CPD_InducTm_CENSOR'] = 1
        
        
        return df_sbr_sn_hd, entry
                
                    
    def common(self, df_sbr_sn_hd, entry):
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
        
        entry = self.getSTD(df_sbr_sn_hd, entry)
        if len(df_sbr_sn_hd) <=100:
            df_sbr_sn_hd, entry = self.getLR(df_sbr_sn_hd, entry)
        else:
            
            #get changepoints
            entry, df_sbr_sn_hd = self.getChangepoints(df_sbr_sn_hd, entry)
            
            #get piecewise linear fit
            entry, df_sbr_sn_hd = self.getPieceWiseResults(df_sbr_sn_hd, entry)
            
            #check flat/improving
            
            #get comments for every part
            slps = [' slope pre induct', ' slope', ' slope end']
            cmts = [' pre-InducTm', ' InducTm to TTF', ' TTF to end']
            for part in range(3):
                try:
                    entry = self.commentsPart(entry, slps[part], cmts[part])
                except:
                    pass
            
            
            #get the negative slope values
            entry = self.getNegativeSlope(df_sbr_sn_hd, entry)
        
        
            #apply rules 
            entry = self.commonRules(entry, df_sbr_sn_hd)
           
            df_sbr_sn_hd, entry = self.checkFI(df_sbr_sn_hd, entry)
        
            
        entry = self.getEndpoints(df_sbr_sn_hd, entry)
        
        
        entry = self.PostProcessing(entry)
        
        
        
        #save plots of the piecewise linear fits to PNG file
        #fit the LR and plot the curves
        plt.figure(figsize = (15, 28), dpi = 80)

        y_true_colors = ['b', mcolors.CSS4_COLORS['darkviolet'], 'm', mcolors.CSS4_COLORS['mediumvioletred'], mcolors.CSS4_COLORS['yellowgreen'],mcolors.CSS4_COLORS['palevioletred']]
        smooth_colors = ['r', mcolors.CSS4_COLORS['orange'], mcolors.CSS4_COLORS['olive'], mcolors.CSS4_COLORS['palegreen'], mcolors.CSS4_COLORS['orchid'], 'c']
        y_colors = ['g', 'c', mcolors.CSS4_COLORS['darkkhaki'], mcolors.CSS4_COLORS['cadetblue'], mcolors.CSS4_COLORS['navy'], mcolors.CSS4_COLORS['crimson']]
        
        
        units = ['', '(dcd)', '(nm)', '(dB)', '(dB)', '(%)']
        subplots = np.arange(711, 718)
        entry['CPD_RH_STRESS_TIME_hrs_max'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
        ax1 = plt.subplot(subplots[0])
        
        if entry['CPD_shunt'] != '':
            plt.title('SERIAL_NUM '+df_sbr_sn_hd['SERIAL_NUM'].iloc[0] + ' comment '+entry['CPD_shunt'])
        else:
            plt.title('SERIAL_NUM '+df_sbr_sn_hd['SERIAL_NUM'].iloc[0] + ' comment '+entry['CPD_InducTm to TTF results'])
        ax = ax1
        plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd['STRESS_CLEARANCE'].values, color = mcolors.CSS4_COLORS['darkgreen'], label = 'STRESS_CLEARANCE')
        plt.xlim(0, self.burnish_results['RH_STRESS_TIME_hrs'].max())
        ax.set_xlabel('RH_STRESS_TIME_hrs')
        
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")   
        plt.ylabel('STRESS CLEARANCE')
        
        stress_clearances = list(set(df_sbr_sn_hd['STRESS_CLEARANCE'].values))
        for i in range(len(stress_clearances)):
            entry['STRESS_CLEARANCE' + str(i+1)] = stress_clearances[i]
        for i in range(len(self.raw_signals)):
            
            #save overall slope
            
            
            ax = plt.subplot(subplots[i+1], sharex = ax1)
        
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.raw_signals[i]].values, y_true_colors[i], label = self.raw_signals[i])
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.smoothed[i]].values, smooth_colors[i], label = self.smoothed[i])
            
            plt.plot(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values, df_sbr_sn_hd[self.linear_fit[i]].values, y_colors[i], label = self.linear_fit[i])   
            plt.plot([], [], ' ', label = 'InducTm '+str(np.around(entry['CPD_' + self.entries[i] + '_InducTm'], decimals = 2)))
            if self.cuts[self.linear_fit[i]]:
                plt.plot([], [], ' ', label = 'TTF '+str(np.around(entry['CPD_' + self.entries[i]+'_TTF'], decimals = 2)))
                plt.axvline(entry['CPD_' + self.entries[i] + '_TTF'], color = 'black', ls = '--', label = 'TTF')
            
            
            
            
            
            plt.axvline(entry['CPD_' + self.entries[i] + '_InducTm'], color = 'r', ls = '--', label = 'InducTm')
            
            if not(np.isnan(entry['CPD_' + self.entries[i] + '_ChangePoint'])):
                plt.axvline(entry['CPD_' + self.entries[i] + '_ChangePoint'], color = 'g', ls = '--', label = 'ChangePoint')
            
            
            
            if self.cuts[self.linear_fit[i]]:
                if 'CPD_' + self.entries[i] + '_TTF_val' in entry:
                    plt.axhline(entry['CPD_' + self.entries[i] + '_TTF_val'], ls = '--', color = mcolors.CSS4_COLORS['darkslategray'])
                    if self.linear_fit[i] == 'ASYMfit':
                        plt.axhline(-entry['CPD_' + self.entries[i] + '_TTF_val'], ls = '--', color = mcolors.CSS4_COLORS['darkslategray'])
                else:
                    plt.axhline(cfg.cutoff[self.linear_fit[i]], ls = '--', color = mcolors.CSS4_COLORS['darkslategray'])
                if self.linear_fit[i] == 'ASYMfit':
                    plt.axhline(-cfg.cutoff[self.linear_fit[i]], ls = '--', color = mcolors.CSS4_COLORS['darkslategray'])
            
            
            ax.set_xlabel('RH_STRESS_TIME_hrs')

            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")   
            plt.ylabel(self.raw_signals[i] + units[i])
            
           
        if not(os.path.exists(os.path.join(cfg.absolute_pth, self.output_name, df_sbr_sn_hd['SBR'].iloc[0]+'/'))):
            os.makedirs(os.path.join(cfg.absolute_pth, self.output_name, df_sbr_sn_hd['SBR'].iloc[0]+'/'))
        
        plt.savefig(cfg.absolute_pth + self.output_name+'/'+df_sbr_sn_hd['SBR'].iloc[0]+'/'+ df_sbr_sn_hd['SBR'].iloc[0]+'_'+\
                    df_sbr_sn_hd['SERIAL_NUM'].iloc[0]+'_'+ df_sbr_sn_hd['HEAD'].iloc[0].astype(int).astype(str)+'_CPD_V3_2.png',\
                        bbox_inches = 'tight')
        # plt.show()
        plt.close() 
        #check for plateau and improving
        entry = self.checkCensor(df_sbr_sn_hd, entry)
        
        
        self.final_results = pd.concat([self.final_results, df_sbr_sn_hd], ignore_index=True)
        
        entry = pd.DataFrame(entry, index = [0])
        
        self.cluster_results = pd.concat([self.cluster_results, entry], ignore_index = True)

    
    def getEndpoints(self, df_sbr_sn_hd, entry):
        '''
        This function gets the endpoints for the raw, smooth and linear fit data

        Parameters
        ----------
        df_sbr_sn_hd : Filtered DataFrame of the particular SBR-SN-HD
        entry : dictionary with current SBR-SN results

        Returns
        -------
        None.

        '''
        add = {'BER': 0 , 'PW50':0, 'VGAS': 0, 'SNRE': 0, 'MRE': 0}
        for i in range(len(self.entries)):
            entry['CPD_' + self.entries[i] + ' degrading censor'] = entry['CPD_' + self.entries[i] + ' TTF censor'] = entry['CPD_' + self.entries[i] + ' InducTm to TTF']
        for i in range(len(self.raw_signals)):
            temp = df_sbr_sn_hd.copy()
            #get raw signals start and end points
            entry['CPD_' + self.raw_signals[i] + ' startpoint'] = df_sbr_sn_hd[self.raw_signals[i]].iloc[0]
            entry['CPD_' + self.raw_signals[i] + ' endpoint'] = df_sbr_sn_hd[self.raw_signals[i]].iloc[-1]
            flag = 0
            if (self.cuts_analysis[self.linear_fit[i]]) & (entry[self.entries[i] + '_TTF']!=df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()):
                # get the start and end points after TTF for smoothed and linear fit
                entry['CPD_' + self.smoothed[i] + ' post-TTF endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                              entry[self.entries[i] + '_TTF']][self.smoothed[i]].max()
                entry['CPD_' + self.smoothed[i] + ' post-TTF startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                              entry[self.entries[i] + '_TTF']][self.smoothed[i]].min()
                entry['CPD_' + self.linear_fit[i] + ' post-TTF endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                              entry[self.entries[i] + '_TTF']][self.linear_fit[i]].max()
                entry['CPD_' + self.linear_fit[i] + ' post-TTF startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                              entry[self.entries[i] + '_TTF']][self.linear_fit[i]].min()
                
                flag = 1
            
            
            if flag == 1:
                    df_sbr_sn_hd = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']< entry[self.entries[i]+'_TTF']]
            
            #get overall slope of the signal upto the TTF
            entry['CPD_' + self.entries[i] + ' overall slope'] = LinearRegression().fit(df_sbr_sn_hd['RH_STRESS_TIME_hrs'].values.reshape(-1,1), \
                                                                   df_sbr_sn_hd[self.linear_fit[i]].values.reshape(-1,1)).coef_[0]

            if entry['CPD_' + self.entries[i] + '_InducTm']!=df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min():
                #get the pre-induction time linear and smoothed signal start and end points
                try:
                    entry['CPD_' + self.smoothed[i] + ' pre-InducTm endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']< \
                                                                      entry['CPD_' + self.entries[i] + '_InducTm']][self.smoothed[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.smoothed[i] + ' pre-InducTm startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']< \
                                                                      entry['CPD_' + self.entries[i] + '_InducTm']][self.smoothed[i]].values[0]
                except:
                    pass
                try:
                    entry['CPD_' + self.smoothed[i] + ' post-InducTm endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                                      entry['CPD_' + self.entries[i] + '_InducTm']][self.smoothed[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.smoothed[i] + ' post-InducTm startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                                      entry['CPD_' + self.entries[i] + '_InducTm']][self.smoothed[i]].values[0]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' pre-InducTm endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']< \
                                                                  entry['CPD_' + self.entries[i] + '_InducTm']][self.linear_fit[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' pre-InducTm startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']< \
                                                                  entry['CPD_' + self.entries[i] + '_InducTm']][self.linear_fit[i]].values[0]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' post-InducTm endpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                                  entry['CPD_' + self.entries[i] + '_InducTm']][self.linear_fit[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' post-InducTm startpoint'] = df_sbr_sn_hd[df_sbr_sn_hd['RH_STRESS_TIME_hrs']> \
                                                                  entry['CPD_' + self.entries[i] + '_InducTm']][self.linear_fit[i]].values[0]
                except:
                    pass
                

                
            else:
                
                try:
                    entry['CPD_' + self.smoothed[i] + ' post-InducTm endpoint'] = df_sbr_sn_hd[self.smoothed[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.smoothed[i] + ' post-InducTm startpoint'] = df_sbr_sn_hd[self.smoothed[i]].values[0]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' post-InducTm endpoint'] = df_sbr_sn_hd[self.linear_fit[i]].values[-1]
                except:
                    pass
                try:
                    entry['CPD_' + self.linear_fit[i] + ' post-InducTm startpoint'] = df_sbr_sn_hd[self.linear_fit[i]].values[0]
                except:
                    pass
               
            try:
                if (flag == 0)  & \
                    (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] == 'degrading') & (self.entries[i]!='ASYM') & \
                        (((entry['CPD_comment'] == 'degrading') & (entry['CPD_' + self.entries[i] + '_InducTm']!=df_sbr_sn_hd['RH_STRESS_TIME_hrs'].min()))| \
                            (entry['CPD_comment'] == 'dMRE but not all parameters oxidizing')) & (entry['CPD_' + self.entries[i] + ' Used Pre-InducTm'] != 'YES') & \
                            (entry['CPD_' + self.entries[i] + ' TTF to end'] != 'degrading'):
                            
                    if (self.entries[i]!='SNRE') & \
                        ((entry['CPD_' + self.linear_fit[i] + ' post-InducTm endpoint']-entry['CPD_' + self.linear_fit[i] + ' post-InducTm startpoint']) < cfg.degrad_thresholds[self.entries[i]]):
                            
                            entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'flat'
                            entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                            add[self.entries[i]] = 1
                            
                    elif (self.entries[i] == 'SNRE') & \
                        ((entry['CPD_' + self.linear_fit[i] + ' post-InducTm endpoint']-entry['CPD_' + self.linear_fit[i] + ' post-InducTm startpoint']) > cfg.degrad_thresholds[self.entries[i]]):
                            
                            entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'flat'
                            entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                            add[self.entries[i]] = 1
            except:
                
                entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'flat'
                entry['CPD_' + self.entries[i] + '_InducTm'] = df_sbr_sn_hd['RH_STRESS_TIME_hrs'].max()
                add[self.entries[i]] = 1

            df_sbr_sn_hd = temp
            
            
            if self.entries[i]!= 'ASYM':
                if (self.cuts[self.linear_fit[i]]== True) & (entry['CPD_' + self.entries[i] + ' InducTm to TTF'] != 'degrading'):
                    entry['CPD_' + self.entries[i] + ' InducTm to TTF'] = 'degrading'
                    entry['CPD_' + self.entries[i] + ' TTF censor'] = 'degrading'
                    entry['CPD_' + self.entries[i] + '_InducTm'] = entry['CPD_' + self.entries[i] + '_TTF']

        if ((entry['CPD_BER InducTm to TTF'] == 'degrading') | (entry['CPD_SNRE InducTm to TTF'] == 'degrading')) & \
            ((entry['CPD_PW50 InducTm to TTF'] == 'degrading') | (entry['CPD_VGAS InducTm to TTF'] == 'degrading')):
            entry['CPD_comment'] = 'degrading'
            entry['CPD_InducTm_CENSOR'] = 0
        else:    
            entry['CPD_comment'] = 'plateau'
            entry['CPD_InducTm_CENSOR'] = 1
        inductms = []
        for i in range(len(self.entries)):
            if not(np.isnan(entry['CPD_' + self.entries[i] + '_InducTm'])):
                inductms.append(entry['CPD_' + self.entries[i] + '_InducTm'])
        if entry['CPD_InducTm_CENSOR'] == 0:    
            entry['CPD_InducTm'] = min(inductms)
        return entry

    def Processing(self):
        '''
        Analyze all the heads present in the SBR

        Returns
        -------
        None.

        '''
        
        cnt = 0
        
        for sbr_sn_hd in self.burnish_results['SBR_SN_HD'].unique().tolist():
            
                cnt += 1
                
                entry = {}
                #filter the DF to get rows with specific SBR, SN, HD
                df_sbr_sn_hd = self.burnish_results[self.burnish_results['SBR_SN_HD'] == sbr_sn_hd]
                df_sbr_sn_hd = df_sbr_sn_hd.reset_index()
                
                #fill known values
                entry['SBR'] = df_sbr_sn_hd['SBR'].iloc[0]
                entry['SN'] = df_sbr_sn_hd['SERIAL_NUM'].iloc[0]
                entry['HD'] = df_sbr_sn_hd['HEAD'].iloc[0]
                entry['HD_NUM'] = df_sbr_sn_hd['HD_NUM'].iloc[0]
                entry['CPD_VERSION'] = cfg.VERSION
                entry['CPD_MRE_TTF'] = entry['CPD_BER_TTF'] = entry['CPD_VGAS_TTF'] = entry['CPD_PW50_TTF'] = entry['CPD_ASYM_TTF'] = entry['CPD_SNRE_TTF'] = entry['CPD_comment'] = ''
                entry['CPD_InducTm_CENSOR'] = 1
                entry['CPD_shunt'] = ''
                
                #skip heads if <N_SKIP data points 
                df_sbr_sn_hd_skip, skip = self.skipSN(df_sbr_sn_hd, entry)
                if skip:
                    continue
                df_sbr_sn_hd = df_sbr_sn_hd_skip.copy()
                #check if >N_MISSING data points
                df_sbr_sn_hd_miss, missing = self.checkMissing(df_sbr_sn_hd, entry)
                
                if missing:
                    continue
                df_sbr_sn_hd = df_sbr_sn_hd_miss.copy()
                entry['CPD_MRE TTF to end'] = entry['CPD_BER TTF to end'] = entry['CPD_PW50 TTF to end'] = entry['CPD_VGAS TTF to end'] = entry['CPD_SNRE TTF to end'] = entry['CPD_ASYM TTF to end'] = ''
                entry['CPD_MRE Used Pre-InducTm'] = entry['CPD_BER Used Pre-InducTm'] = entry['CPD_PW50 Used Pre-InducTm'] = entry['CPD_VGAS Used Pre-InducTm'] = \
                    entry['CPD_SNRE Used Pre-InducTm'] = entry['CPD_ASYM Used Pre-InducTm'] = 'NO'
                entry['CPD_SHUNT_time'] = ''
                #Point to point dMRE for big shunts (rev039)
                #df_sbr_sn_hd, shunt_mat = self.shuntCalculation(df_sbr_sn_hd)
                
                self.common(df_sbr_sn_hd, entry)
                
                
                
                
    def PreProcessing(self):
        '''
        Calls all the required functions to preprocess the heads

        Returns
        -------
        None.

        '''
        
        self.deleteBadBER()
        self.checkDF()
        self.getDeltas()
        self.smoothing()
    
    def OverallRow(self, cmts):
        '''
        This function gets a row values and converts it into one value in the form of DDDDDD
        Parameters
        ----------
        cmts : List of columns to integrate

        Returns
        -------
        final_cmt : returns the integrated column value for this particular row

        '''
        d = {'degrading': 'D', 'plateau': 'S', 'improving': 'I', 'None':'N','flat': 'S', \
             'Less than N_SKIP '+str(cfg.N_SKIP) + ' data points': 'Less than N_SKIP '+str(cfg.N_SKIP) + ' data points',\
             'Missing > N_MISSING Rows, do not analyze' : 'Missing > N_MISSING Rows, do not analyze',\
             'SHUNT, do not analyze': 'SHUNT', \
             'degrading possible shunt': 'degrading possible shunt', \
             'dMRE' : 'dMRE', 'SHUNT' : 'SHUNT', 
             'dMRE but not all parameters oxidizing' : 'dMRE but not all parameters oxidizing',
             'YES':'Y', 'NO':'N'
             }
        final_cmt = ''
        for i in range(len(cmts)):
            final_cmt += d[cmts[i]]
            if i != len(cmts)-1:
               final_cmt += '_' 
            
        return final_cmt
    
    def getComments(self, entry, cols, col_name):
        '''
        This function acts like a wrapper of Postprocessing functions

        Parameters
        ----------
        entry : Dictionary with all the cluster results
        cols : These are the list of columns to be integrated
        col_name : This is a string of the name of output column

        Returns
        -------
        None.

        '''
        entry = pd.DataFrame(entry, index = [0])
        
        comments = entry[cols]
        if col_name == 'CPD_Used Pre-InducTm':
            
            comments = comments.replace({'':'NO'})
            comments = comments.fillna('NO')
        else:
            comments = comments.replace({'':'None'})
            comments = comments.fillna('None')
        
        
        entry[col_name] = comments.apply(lambda row: self.OverallRow(row[cols]), axis = 1)
            
        
        entry = entry.drop(columns = cols)
        entry = entry.to_dict(orient = 'records')
        
        return entry[0]


        
    def PostProcessing(self, entry):
        '''
        this function does all the post-processing required.

        Returns
        -------
        None.

        '''
        
        cmt_names = ['pre-InducTm', 'InducTm to TTF', 'TTF to end']
        
        for i in range(len(cmt_names)):
            
            try:
                entry = self.getComments(entry, ['CPD_MRE '+cmt_names[i], 'CPD_BER '+cmt_names[i],'CPD_PW50 '+cmt_names[i],'CPD_VGAS '+cmt_names[i],\
                                  'CPD_SNRE '+cmt_names[i], 'CPD_ASYM '+cmt_names[i]], 'CPD_' + cmt_names[i] + ' results')
            except:
                pass
        try:
            entry = self.getComments(entry, ['CPD_MRE Used Pre-InducTm', 'CPD_BER Used Pre-InducTm', 'CPD_PW50 Used Pre-InducTm', \
                                             'CPD_VGAS Used Pre-InducTm', 'CPD_SNRE Used Pre-InducTm', 'CPD_ASYM Used Pre-InducTm'], \
                                     'CPD_Used Pre-InducTm')
        except:
            pass
        

        entry = self.getComments(entry, ['CPD_MRE TTF censor', 'CPD_BER TTF censor', 'CPD_PW50 TTF censor', 'CPD_VGAS TTF censor', \
                                          'CPD_SNRE TTF censor', 'CPD_ASYM TTF censor'], 'CPD_TTF censor')

        entry = self.getComments(entry, ['CPD_MRE degrading censor', 'CPD_BER degrading censor', 'CPD_PW50 degrading censor', 'CPD_VGAS degrading censor', \
                                              'CPD_SNRE degrading censor', 'CPD_ASYM degrading censor'], 'CPD_degrading censor')

        
        
        if entry['CPD_comment']!= 'degrading':
            entry['CPD_InducTm_CENSOR'] = 1
        del entry['MRE_TTF']
        del entry['BER_TTF']
        del entry['PW50_TTF']
        del entry['VGAS_TTF']
        del entry['SNRE_TTF']
        del entry['ASYM_TTF']
        return entry
        
        
    def main(self):
        '''
        This function acts as a wrapper

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        self.PreProcessing()
        self.Processing()
        
        return self.final_results, self.cluster_results
        
