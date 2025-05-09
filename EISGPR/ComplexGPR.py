
import re
import os

import numpy as np
from loguru import logger

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from collections import defaultdict
from datetime import datetime

import torch
import gpytorch
from sklearn.preprocessing import StandardScaler



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
    _channelNum     = 0

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
                    _channelNum = np.max([_channelNum, _channelIndex])
                            
    return EISDict, _channelNum

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
    for ssID in EISDict.keys():
        _data   = np.loadtxt(fileDict[ssID][chID], delimiter=',')
        _freq   = _data[:,0]
        _Zreal  = _data[:,1] * np.cos(np.deg2rad(_data[:,2])) 
        _Zimag  = _data[:,1] * np.sin(np.deg2rad(_data[:,2])) 
        chData.append(np.stack((_freq, _Zreal, _Zimag),axis=0))

    return np.stack(chData, axis=0)

def ComplexGPTrain(x_train, y_train, x_predict, device, training_iter = 50):
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
    model = ComplexGPModel(x_train, y_train, likelihood).to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_train))
        pred = likelihood(model(x_predict))
    # logger.info("Model Evaluation Finished.")

    return observed_pred, pred




class ComplexGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    rootPath = "D:/Baihm/EISNN/Archive/"
    savePath = "D:/Baihm/EISNN/Archive/TorchDataset/"


    if not os.path.exists(savePath):
        os.mkdir(savePath)
        os.mkdir(f"{savePath}/Figures/")

    archive_pattern = re.compile(r"(.+?)_归档")
    dirList = os.listdir(rootPath) 
    
    # for dirID in range(len(dirList)):
    for dirID in range(4,17):
        ## Log: 20250213-22:33 - 第一批先处理20组数据，差不多14个小时，明天早上再看
        match_archive = archive_pattern.match(dirList[dirID])
        workPath = f"{rootPath}/{dirList[dirID]}"
        if match_archive and os.path.isdir(workPath):
            logger.info(f"Archive Begin: {match_archive[1]} - {dirID}/{len(dirList)}")
            EISDict, channelNum = gatherCSV(workPath)


            # 根据EISDict的key确定日期范围，然后把日期范围映射到0~days
            # Speed Rate = n means 1 day = n points
            SPEED_RATE = 1
            x_day = [datetime.strptime(date, '%Y%m%d') for date in EISDict.keys()]
            x_train = np.array([(poi - x_day[0]).days for poi in x_day])
            x_predict = np.linspace(0,max(x_train),max(x_train)*SPEED_RATE+1)
            x_train_tensor = torch.tensor(x_train).to(device).float()
            x_predict_tensor = torch.tensor(x_predict).to(device).float()

            tensorDataList = []
            for chID in range(89,channelNum):
            # for chID in range(channelNum):
                logger.info(f"Channel Begin: {dirID}-ch{chID:03d}")
                chData = readChannel(chID, EISDict)
                freq_list = np.linspace(0,np.shape(chData)[2]-1,101,dtype=int)

                EIS_GP = []
                n_RI = np.shape(x_predict)[0]  

                for i in freq_list:
                    # Training Data Normalizzation
                    scaler_real = StandardScaler()
                    scaler_imag = StandardScaler()

                    y_train_real = scaler_real.fit_transform(chData[:,1,i].reshape(-1,1))
                    y_train_imag = scaler_imag.fit_transform(chData[:,2,i].reshape(-1,1))
                    
                    y_train_tensor = torch.tensor(np.hstack([y_train_real, y_train_imag]) , device=device).float()
                
                    # y_train_tensor = torch.tensor(chData[:,1:,i], device=device).float()
                    observed_pred, pred = ComplexGPTrain(x_train_tensor, y_train_tensor, x_predict_tensor, device, training_iter = 50)
                    mean_pred_norm = pred.mean.cpu().numpy()
                    var_pred_norm = pred.covariance_matrix.cpu().numpy()

                    mean_pred = np.hstack(
                        [scaler_real.inverse_transform(mean_pred_norm[:,0].reshape(-1, 1)),
                        scaler_imag.inverse_transform(mean_pred_norm[:,1].reshape(-1, 1))]
                    )

                    var_scale = mean_pred.transpose().reshape(-1,1) @ mean_pred.transpose().reshape(1,-1)
                    var_pred = var_scale * var_pred_norm


                    # Extract Mean & Var for RI Data
                    _sigmaR = np.array([var_pred[i][i] for i in range(n_RI)])
                    _sigmaI = np.array([var_pred[i+n_RI][i+n_RI] for i in range(n_RI)])
                    _covRI = np.array([var_pred[i][i+n_RI] for i in range(n_RI)])

                    _meanR = mean_pred[:,0]
                    _meanI = mean_pred[:,1]

                    # Calculate Mean & Var for AP Data
                    _amp_mean = np.abs(_meanR+1j*_meanI)
                    _phz_mean = np.angle(_meanR+1j*_meanI)

                    _amp_var = np.sqrt(((_meanR**2)*_sigmaR + (_meanI**2)*_sigmaI + 2*_meanR*_meanI*_covRI))/(_amp_mean)
                    _phz_var = np.sqrt(((_meanI**2)*_sigmaR + (_meanR**2)*_sigmaI - 2*_meanR*_meanI*_covRI))/(_amp_mean**2)

                    _phz_mean = np.rad2deg(_phz_mean)
                    _phz_var = np.rad2deg(_phz_var)

                    EIS_GP.append([[_amp_mean,_amp_var],[_phz_mean, _phz_var]])

                    # logger.info(f"Freq: {i} - Done")

                EIS_GP = np.stack(EIS_GP, axis=1)
                EIS_GP = np.transpose(EIS_GP, (0,2,1,3))
                tensorDataList.append(torch.tensor(EIS_GP, dtype=torch.float32))


                # Save Plot
                amp = EIS_GP[0]
                phz = EIS_GP[1]

                fig = plt.figure(figsize=(12, 6))
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                init_elev = 21  # 仰角
                init_azim = 55  # 方位角
                ax1.view_init(elev=init_elev, azim=init_azim)
                ax2.view_init(elev=init_elev, azim=init_azim)

                y = np.array(x_predict).flatten()
                x = np.log10(chData[0,0,freq_list]).flatten()
                X, Y = np.meshgrid(x, y, indexing='ij')

                ax1.plot_surface(X, Y, np.log10(amp[0]), cmap='viridis_r')
                ax2.plot_surface(X, Y, phz[0], cmap='viridis')

                ax1.set_zlim([2,8.5])
                ax2.set_zlim([-120,30])
                
                if not os.path.exists(f"{savePath}/Figures/{match_archive[1]}"):
                    os.mkdir(f"{savePath}/Figures/{match_archive[1]}/")
                fig.savefig(f"{savePath}/Figures/{match_archive[1]}/ch{chID:03d}.png")
                logger.info(f"Figures Saved: {savePath}/Figures/{match_archive[1]}/ch{chID:03d}.png")


            tensorDataList = torch.stack(tensorDataList, dim=0)
            tensorFile = f"{savePath}{match_archive[1]}.pt"
            torch.save(tensorDataList, tensorFile)
            logger.info(f"{savePath}{match_archive[1]}.pt Saved! \nData Shape: {tensorDataList.shape}")
            