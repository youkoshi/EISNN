# import Interpolation
import os
import re

from loguru import logger
import numpy as np


from collections import defaultdict


def gatherCSV(rootPath, outsuffix = 'Tracking'):
    '''==================================================
        Collect all EIS.csv files in the rootPath
        Parameter: 
            rootPath: current search path
            outsuffix: Saving path of EIS.csv files
        Returen:
            EISDict: a 2D-dict of EIS data
            Storage Frame: EISDict[_sessionIndex][_channelIndex] = "_filepath"
        ==================================================
    '''
    _filename       = None
    _filepath       = None
    _trackpath      = None
    _csvpath        = None
    _sessionIndex   = None
    _channelIndex   = None
    _processed      = None

    EISDict = defaultdict(dict)

    ## Iterate session
    session_pattern = re.compile(r"(.+?)_(\d{8})_01")
    bank_pattern    = re.compile(r"([1-4])")
    file_pattern    = re.compile(r"EIS_ch(\d{3})\.csv")

    ## RootDir
    for i in os.listdir(rootPath):
        match_session = session_pattern.match(i)
        ## SessionDir
        if match_session:
            logger.info(f"Session Begin: {i}")
            _sessionIndex = match_session[2]
            for j in os.listdir(f"{rootPath}/{i}"):
                match_bank = bank_pattern.match(j)
                ## BankDir
                if match_bank:
                    logger.info(f"Bank Begin: {j}")
                    _trackpath = f"{rootPath}/{i}/{j}/{outsuffix}"
                    if not os.path.exists(_trackpath):
                        continue

                    for k in os.listdir(f"{rootPath}/{i}/{j}/{outsuffix}"):
                        match_file = file_pattern.match(k)
                        ## File
                        if match_file:
                            _filename = k
                            _filepath = f"{rootPath}/{i}/{j}/{outsuffix}/{k}"
                            _channelIndex = (int(match_bank[1])-1)*32+int(match_file[1])
                            
                            EISDict[_sessionIndex][_channelIndex] = f"{rootPath}/{i}/{j}/{outsuffix}/{k}"
                            
    return EISDict

# Data Readout
def readChannel(chID, fileDict):
    '''==================================================
        Read EIS.csv file by Channel
        Parameter: 
            chID: channel index
            fileDict: EISDict[_sessionIndex][_channelIndex] = "_filepath"
        Returen:
            freq: frequency
            Zreal: real part of impedance
            Zimag: imaginary part of impedance
        ==================================================
    '''
    chData = []
    for ssID in fileDict.keys():
        _data   = np.loadtxt(fileDict[ssID][chID], delimiter=',')
        _freq   = _data[:,0]
        _Zreal  = _data[:,3]
        _Zimag  = _data[:,4]
        chData.append(np.stack((_freq, _Zreal, _Zimag),axis=0))

    return np.stack(chData, axis=0)




def EIS_recal_ver02(data, _phz_0 = None):
    f_poi = data[0,:]
    # Z_poi = data[1,:] * np.exp(1j*np.deg2rad(data[2,:]))
    Z_poi = data[1,:] + 1j*data[2,:]
    Y_poi = 1/Z_poi

    Rg0 = 1.611e13
    Cp0 = 1.4e-9
    
    _Rg0_rescale = 1/Rg0*np.power(f_poi,1.583)
    _Cp0_rescale = Cp0*np.power(f_poi,0.911)
    Y_org = Y_poi - _Rg0_rescale + 1j*_Cp0_rescale
    # Y_org = Y_poi - _Rg0_rescale 
    # Y_org = Y_poi + 1j*_Cp0_rescale
    # Y_org = Y_poi
    Z_org = 1/Y_org

    # Phz Calibration
    if _phz_0 is None:
        _phz_0 = np.loadtxt("./phz_Calib.txt")
    
    Z_ampC = np.abs(Z_org)
    # Z_phzC = np.angle(Z_org) - _phz_0
    Z_phzC = np.angle(Z_org) - _phz_0

    Z_rec = Z_ampC * np.exp(1j*Z_phzC)

    # C = 5e-10
    Rs0 = 100
    Z_rec = Z_rec - Rs0



    Cp0 = 5e-10
    _Cp0_rescale = Cp0 * f_poi
    Z_rec = 1/(1/Z_rec - 1j * _Cp0_rescale)

    

    # Ls0 = 1.7e-4
    Ls0 = 5e-4
    _Ls0_rescale = Ls0 * f_poi
    Z_rec = Z_rec - 1j * _Ls0_rescale

    # C = 5e-10
    Rs0 = 566
    Z_rec = Z_rec - Rs0
    
    return np.stack([f_poi, np.real(Z_rec), np.imag(Z_rec)], axis=1).T
    


