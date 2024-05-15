#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:40:33 2023

@author: ruchak
"""



#Paths
absolute_pth = '/home/ruchak/JSL_Python/'
raw_data = '00_Raw/'
output_preprocessing = 'cleaned_dataset/'
output_decisions = 'advanced_engineering_and_diagnostics'
output_TTF = 'results_and_failmodes'

VERSION = 'v20230918'
N_SKIP = 5 #if < these number of points, skip analyzing
N_MISSING = 100 #if > these number of points are NA, skip analyzing

#SHUNT CALCULATION
SHUNT_DMRE = 3
SHUNT_DVGA = -0.2

#Piecewise Regression
SNR_OVERALL = 0.0013
PW50_OVERALL = 0.0007
MRE_OVERALL = 0.007
BER_OVERALL = 0.00019
ASYM_OVERALL = 0.0063
VGAS_OVERALL = 0.002
overall_dict = {'MRE':MRE_OVERALL, 'BER':BER_OVERALL, 'PW50':PW50_OVERALL, 'VGAS':VGAS_OVERALL, 'SNRE':SNR_OVERALL, 'ASYM':ASYM_OVERALL}


N_BKPS = 2
PEN = 20 #30
cutoff = {'MREfit': 5, 'BERfit': 0.25, 'VGASfit': 2, 'PW50fit': 0.5, 'SNREfit':-2, 'ASYMfit':12.5}
#saving files
REV = '062'
MOV_AVG_N = 40
RAW_SIGNALS = {'AP3.8':{'MRE': 'MRE_RESISTANCE', 'BER': 'BER_MEAN', 'PW50': 'PW50_DIBIT', 'VGAS':'VGAS_DB', 'SNRE':'EW_SNR_ELEC', \
               'ASYM': 'EW_ASYM'}, \
               'AP3.6': {'MRE': 'MRE_RESISTANCE', 'BER': 'RO_AVG_MEAN_CW_SOVA1', 'PW50': 'EW_PW50', 'VGAS':'VGAS_DB', 'SNRE':'EW_SNR_ELEC', \
                              'ASYM': 'EW_ASYM'}}

AP_VERSIONS = {'D9': 'AP3.8', 'D8': 'AP3.6'}

degrad_thresholds = {'MRE': 1.2, 'BER': 0.015, 'PW50':0.05, 'VGAS': 0.6, 'SNRE': -0.025}


