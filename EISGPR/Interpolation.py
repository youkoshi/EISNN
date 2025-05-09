import numpy as np
from loguru import logger

from datetime import datetime

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

''' ========================================================
                            Data Loader
    ========================================================
'''
def piecewise_interp(x_day_all, chData, eis_seq, run_list = None, eis_cluster = None, SPEED_RATE = 1, LOG_FLAG = True):
    '''==================================================
        Seperate the EIS data into clusters for interpolation
        Parameter: 
            chData: EIS data #sample x #freq x2
            eis_seq: Index of valid data #valid_sample x 1
            run_list: Index of valid freq #valid_freq x 1, default None for all freq valid
            eis_cluster: cluster label for each data in eis_seq #valid_sample x 1, default None for all data in one cluster
            SPEED_RATE: Speed of interpolation, default 1 for 1x interpolation
            LOG_FLAG: Logarithm flag, default True for log scale
        Returen:
            x_train_full: x_train data #train x 1
            y_train_full: y_train data #train x #freq x2
            x_eval_full: x_eval data #eval x 1
            n_clusters: number of clusters #uniq_clusters x 1
            train_mask_list: mask for each cluster in train data
            eval_mask_list: mask for each cluster in eval data
            eis_cluster_eval: cluster label for each data in x_eval_full
        ==================================================
    '''
    # Init xy according to the datetime

    x_day = [x_day_all[i] for i in eis_seq]

    x_train_full = np.array([(poi - x_day[0]).days for poi in x_day])
    x_eval_full = np.linspace(0,max(x_train_full),max(x_train_full)*SPEED_RATE+1)

    y_train_full = np.stack([chData[eis_seq,1,:],chData[eis_seq,2,:]], axis=2)
    
    if run_list is not None:
        y_train_full = y_train_full.take(run_list, axis=1)


    if LOG_FLAG:
        y_train_log = np.log(y_train_full[:,:,0] + 1j*y_train_full[:,:,1])
        y_train_full = np.stack([y_train_log.real, y_train_log.imag], axis=2)


    # Segmentation of clusters
    if eis_cluster is None:
        eis_cluster = np.zeros_like(eis_seq)
    unique_clusters = np.unique(eis_cluster)
    n_clusters = len(unique_clusters)

    train_mask_list = []
    eval_mask_list = []
    eis_cluster_eval = np.zeros_like(x_eval_full)
    for i in range(n_clusters):
        # 取当前状态和下一个状态的数据
        train_mask = (eis_cluster == unique_clusters[i])
        if i == n_clusters - 1:
            x_eval_end = x_eval_full.max() + 1
        else:
            x_eval_end = x_train_full[(eis_cluster == unique_clusters[i+1])].min()
        # x_state = x_train[state_mask]
        # y_state = y_train[:,state_mask]

        eval_mask = (x_eval_full >= x_train_full[train_mask].min()) & (x_eval_full < x_eval_end)
        
        train_mask_list.append(train_mask)
        eval_mask_list.append(eval_mask)
        eis_cluster_eval[eval_mask] = unique_clusters[i]

    return x_train_full, y_train_full, x_eval_full, n_clusters, train_mask_list, eval_mask_list, eis_cluster_eval
    


def GPDataLoader(x_train, y_train, x_eval, NORM_X_FLAG = True, NORM_Y_FLAG = True):
    '''==================================================
        Regularize the data for GP
        Parameter: 
            x_train: x_train data #train x 1
            y_train: y_train data #train x #freq x2
            x_eval: x_eval data #eval x 1
            NORM_X_FLAG: Normalization flag for x_train, default True
            NORM_Y_FLAG: Normalization flag for y_train, default True
        Returen:
            x_train: regularized x_train data
            y_train: regularized y_train data
            x_eval: regularized x_eval data
            ScalerSet: ScalerSet[0]: x_train scaler, ScalerSet[1]: y_train real scaler, ScalerSet[2]: y_train imag scaler
        ==================================================
    '''
    Scaler_X        = StandardScaler()
    Scaler_Y_real   = StandardScaler()
    Scaler_Y_imag   = StandardScaler()

    if NORM_Y_FLAG:
        y_train[:,:,0] = Scaler_Y_real.fit_transform(y_train[:,:,0])
        y_train[:,:,1] = Scaler_Y_imag.fit_transform(y_train[:,:,1])
    if NORM_X_FLAG:
        x_train = Scaler_X.fit_transform(x_train.reshape(-1, 1)).flatten()
        x_eval = Scaler_X.transform(x_eval.reshape(-1, 1)).flatten()

    
    y_train = np.hstack((y_train[:,:,0], y_train[:,:,1]))

    logger.info(f"\nx: {np.shape(x_train)} \ny: {np.shape(y_train)} \nx_pred{np.shape(x_eval)}")

    return x_train, y_train, x_eval, [Scaler_X, Scaler_Y_real, Scaler_Y_imag]


def GPDataExporter(x_train, y_train, x_eval, y_eval_mean, y_eval_var, ScalerSet, NORM_X_FLAG, NORM_Y_FLAG):
    '''==================================================
        Deregularize the data for Saving
        Parameter: 
            x_train: regularized x_train data
            y_train: regularized y_train data
            x_eval: regularized x_eval data
            y_eval_mean: mean from GPR #eval x #freq x2
            y_eval_var: var from GPR #eval x #freq x2
            ScalerSet: ScalerSet[0]: x_train scaler, ScalerSet[1]: y_train real scaler, ScalerSet[2]: y_train imag scaler
            NORM_X_FLAG: Normalization flag for x_train, default True
            NORM_Y_FLAG: Normalization flag for y_train, default True
        Returen:
            x_train: deregularized x_train data
            y_train: dederegularized y_train data
            x_eval: dederegularized x_eval data
            y_eval: dederegularized y_eval_mean
            y_eval_err: dederegularized y_eval_var
        ==================================================
    '''
    n_freq = np.shape(y_train)[1]//2 
    y_train = np.stack((y_train[:,:n_freq], y_train[:,n_freq:]), axis=2)
    y_eval_mean = np.stack((y_eval_mean[:,:n_freq], y_eval_mean[:,n_freq:]), axis=2)
    y_eval_var = np.stack((y_eval_var[:,:n_freq], y_eval_var[:,n_freq:]), axis=2)
    if NORM_X_FLAG:
        x_train = ScalerSet[0].inverse_transform(x_train.reshape(-1, 1)).flatten()
        x_eval = ScalerSet[0].inverse_transform(x_eval.reshape(-1, 1)).flatten()
    
    if NORM_Y_FLAG:
        y_train_real = ScalerSet[1].inverse_transform(y_train[:,:,0])
        y_train_imag = ScalerSet[2].inverse_transform(y_train[:,:,1])
        
        y_eval_mean_real = ScalerSet[1].inverse_transform(y_eval_mean[:,:,0])
        y_eval_mean_imag = ScalerSet[2].inverse_transform(y_eval_mean[:,:,1])

        y_eval_var_real = y_eval_var[:,:,0] * ScalerSet[1].var_
        y_eval_var_imag = y_eval_var[:,:,1] * ScalerSet[2].var_
    else:
        y_train_real = y_train[:,:,0]
        y_train_imag = y_train[:,:,1]

        y_eval_mean_real = y_eval_mean[:,:,0]
        y_eval_mean_imag = y_eval_mean[:,:,1]

        y_eval_var_real = y_eval_var[:,:,0]
        y_eval_var_imag = y_eval_var[:,:,1]

    y_train = np.stack([y_train_real, y_train_imag], axis=2)
    y_eval = np.stack([y_eval_mean_real, y_eval_mean_imag], axis=2)
    y_eval_err = np.stack([y_eval_var_real, y_eval_var_imag], axis=2)
    
    logger.info(f"\nx: {np.shape(x_train)} \ny: {np.shape(y_train)} \nx_pred{np.shape(x_eval)} \ny_pred{np.shape(y_eval)} \ny_pred{np.shape(y_eval_err)}")

    return x_train, y_train, x_eval, y_eval, y_eval_err




''' ========================================================
                    Multitask GPR Model
    ========================================================
'''
class EISGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            # gpytorch.kernels.RBFKernel(),
            # gpytorch.kernels.RQKernel(),
            # gpytorch.kernels.LinearKernel(),
            # gpytorch.kernels.PolynomialKernel(power=3.0),
            # gpytorch.kernels.PiecewisePolynomialKernel(),
            # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3),
            # gpytorch.kernels.CosineKernel(),
            
            gpytorch.kernels.MaternKernel(nu=0.5), 
            
            num_tasks=num_tasks, 
            rank=2
        )
        # self.covar_module.data_covar_module.lengthscale = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def EISGPTrain(x_train, y_train, x_eval, cluster_id, device, training_iter = 200, lr = 0.05):
    num_tasks = y_train.shape[1]
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks, rank = 0).to(device)
    model = EISGPModel(x_train, y_train, likelihood, num_tasks).to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Iteration begin
    loss_inst       = []
    length_inst     = []
    noise_inst      = []
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        poi_noise   = model.likelihood.noise.detach().cpu().numpy()
        poi_length  = model.covar_module.data_covar_module.lengthscale.detach().cpu().numpy()
        # poi_length  = 0
        
        loss_inst.append(loss.item())
        noise_inst.append(poi_noise)
        length_inst.append(poi_length)
        if not (i+1)%100:
            logger.info(f"C{cluster_id} - Iter {i+1}/{training_iter}\tLoss: {loss.item()}")
            
    # logger.info("Model Training Finished.")

    # Make predictions
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
        pred = likelihood(model(x_eval))

    return [pred, np.array(loss_inst), np.array(length_inst), np.array(noise_inst)]



''' ========================================================
                    Piecewise GPR Interpolation
    ========================================================
'''
def PiecewiseGPR(x_day_full, chData, eis_seq, eis_cluster = None, SPEED_RATE = 1, training_iter = 200, lr = 0.05):
    '''==================================================
        Piecewise GPR for EIS data
        Parameter: 
            chData: EIS data #sample x #freq x2
            eis_seq: Index of valid data #valid_sample x 1
            eis_cluster: cluster label for each data in eis_seq #valid_sample x 1, default None for all data in one cluster
            SPEED_RATE: Speed of interpolation, default 1 for 1x interpolation
            training_iter: Number of training iterations, default 200
            lr: Learning rate, default 0.05
        Returen:
            y_eval_mean: mean from GPR #eval x #freq x2
            y_eval_var: var from GPR #eval x #freq x2
        ==================================================
    '''
    # Regularization Type
    LOG_FLAG=True
    NORM_X_FLAG=True
    NORM_Y_FLAG=True

    x_train_full, y_train_full, x_eval_full,  n_clusters, train_mask_list, eval_mask_list, eis_cluster_eval = \
        piecewise_interp(x_day_full, chData, eis_seq, None, 
                     eis_cluster = eis_cluster, 
                     SPEED_RATE = SPEED_RATE, 
                     LOG_FLAG=LOG_FLAG)


    y_eval_full = np.zeros((np.shape(x_eval_full)[0], np.shape(y_train_full)[1], 2))
    y_eval_err_full = np.zeros((np.shape(x_eval_full)[0], np.shape(y_train_full)[1], 2))



    for i in range(n_clusters):
    # for i in [1]:

        x_train = x_train_full[train_mask_list[i]]
        y_train = y_train_full[train_mask_list[i],:,:]
        x_eval = x_eval_full[eval_mask_list[i]]

        x_train, y_train, x_eval, ScalerSet = \
            GPDataLoader(x_train, y_train, x_eval, 
                NORM_X_FLAG=NORM_X_FLAG, NORM_Y_FLAG=NORM_Y_FLAG)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_tensor = torch.from_numpy(x_train).float().to(device)
        x_eval_tensor = torch.from_numpy(x_eval).float().to(device)
        y_train_tensor = torch.from_numpy(y_train).float().to(device)


        y_eval_tensor, _, _, _ = EISGPTrain(x_train_tensor, y_train_tensor, x_eval_tensor, 
                cluster_id = i, device = device, 
                training_iter=training_iter, lr=lr)
        y_eval_mean = y_eval_tensor.mean.cpu().numpy()
        y_eval_var = y_eval_tensor.variance.detach().cpu().numpy()


        x_train, y_train, x_eval, y_eval, y_eval_err = \
                GPDataExporter(x_train, y_train, x_eval, y_eval_mean, y_eval_var, ScalerSet,
                            NORM_X_FLAG=NORM_X_FLAG, NORM_Y_FLAG=NORM_Y_FLAG)
            
        y_eval_full[eval_mask_list[i],:,:] = y_eval
        y_eval_err_full[eval_mask_list[i],:,:] = y_eval_err


    return x_train_full, y_train_full, x_eval_full, y_eval_full, y_eval_err_full, eis_cluster_eval



''' ========================================================
                    Plot Summary of Preprocessing
    ========================================================
'''
def EISPreprocessPlot(fig, chData, x_train, y_train, x_eval, y_eval, y_eval_err, eis_seq, eis_cluster, eis_anomaly):
    '''==================================================
        Plot summary for EIS data preprocessing
        Parameter: 
            fig: figure object
            chData: EIS data of one channel
            x_train: x_train data
            y_train: y_train data
            x_eval: x_eval data
            y_eval: mean from GPR
            y_eval_err: var from GPR 
            eis_seq: Index of valid data
            eis_cluster: cluster label for each data in eis_seq
            eis_anomaly: Index of outlier data
        ==================================================
    '''
    axis = [0] * 12
    axis[0] = fig.add_subplot(3,4,1)   
    axis[1] = fig.add_subplot(3,4,2)            
    axis[2] = fig.add_subplot(3,4,3)         
    axis[3] = fig.add_subplot(3,4,4, projection='3d')      
    axis[4] = fig.add_subplot(3,4,5)      
    axis[5] = fig.add_subplot(3,4,6)         
    axis[6] = fig.add_subplot(3,4,7)         
    axis[7] = fig.add_subplot(3,4,8, projection='3d')         
    axis[8] = fig.add_subplot(3,4,9)         
    axis[9] = fig.add_subplot(3,4,10)    
    axis[10] = fig.add_subplot(3,4,11)    
    # axis[9] = fig.add_subplot(2,5,12)    


    axis[0].set_title("Origin Amp")
    axis[1].set_title("Interpolated mean")
    axis[2].set_title("Interpolated var")
    # axis[3].set_title("Overview")
    
    axis[4].set_title("Origin Phase")
    axis[5].set_title("Interpolated mean")
    axis[6].set_title("Interpolated var")
    # axis[7].set_title("Overview")


    axis[8].set_title("Cluster")
    axis[9].set_title("Outlier")
    axis[10].set_title("Trace")
    # axis[11].set_title("Nope")


    init_elev = 40  # 仰角
    init_azim = 45  # 方位角
    axis[3].view_init(elev=init_elev, azim=init_azim)
    axis[7].view_init(elev=init_elev, azim=init_azim)

    axis[0].set_ylim((5e2, 2e8))
    axis[4].set_ylim((-95, 0))

    # Origin Amp & Phase
    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(len(eis_seq)):
        _x = eis_seq[i]
        ch_eis = chData[_x,:,:]
        _color = cmap(_x/np.shape(chData)[0])
        axis[0].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{i:02d}")
        axis[4].semilogx(ch_eis[0,:], np.rad2deg(np.angle(ch_eis[1,:]+1j*ch_eis[2,:])), color = _color, linewidth=2, label=f"S{i:02d}")

    
    # Interpolation Mean
    y_EIS_train =   np.exp(y_train[:,:,0] + 1j * y_train[:,:,1])
    y_EIS_eval = np.exp(y_eval[:,:,0] + 1j * y_eval[:,:,1])

    _freq_poi = chData[0,0,:]
    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(np.shape(x_eval)[0]):
        axis[1].loglog(_freq_poi, (np.abs(y_EIS_eval[i,:])), color = cmap(i/np.shape(x_eval)[0]))
        axis[5].semilogx(_freq_poi, np.rad2deg(np.angle(y_EIS_eval[i,:])), color = cmap(i/np.shape(x_eval)[0]))
        
    for i in range(np.shape(x_train)[0]):
        axis[1].semilogx(_freq_poi, (np.abs(y_EIS_train[i,:])), 'black', alpha = 0.3)
        axis[5].semilogx(_freq_poi, np.rad2deg(np.angle(y_EIS_train[i,:])), 'black', alpha = 0.3)
    
    axis[1].sharex(axis[0])
    axis[1].sharey(axis[0])
    axis[5].sharex(axis[4])
    axis[5].sharey(axis[4])

    # Interpolation Var
    
    for i in range(np.shape(x_eval)[0]):
        axis[2].fill_between(_freq_poi, np.exp(y_eval[i,:,0] - 2*np.sqrt(y_eval_err[i,:,0])), np.exp(y_eval[i,:,0] + 2*np.sqrt(y_eval_err[i,:,0])), 
                alpha=0.2, color = cmap(i/np.shape(x_eval)[0]))
        axis[6].fill_between(_freq_poi, np.rad2deg(y_eval[i,:,1] - 2*np.sqrt(y_eval_err[i,:,1])), np.rad2deg(y_eval[i,:,1] + 2*np.sqrt(y_eval_err[i,:,1])), 
                alpha=0.2, color = cmap(i/np.shape(x_eval)[0]))
    axis[2].set_xscale('log')
    axis[2].set_yscale('log')
    axis[6].set_xscale('log')
    axis[2].sharex(axis[0])
    axis[2].sharey(axis[0])
    axis[6].sharex(axis[4])
    axis[6].sharey(axis[4])


    # Interpolate Plain
    _x = np.arange(np.shape(x_eval)[0])
    _y = np.log10(_freq_poi).flatten()
    X, Y = np.meshgrid(_x, _y, indexing='ij')
    axis[3].plot_surface(X, Y, np.log10(np.abs(y_EIS_eval[:,:])), cmap='viridis_r', alpha=0.8)
    axis[7].plot_surface(X, Y, -np.rad2deg(np.angle(y_EIS_eval[:,:])), cmap='viridis_r', alpha=0.8)



    # Cluster
    cmap = plt.colormaps.get_cmap('Set1')
    for i in range(len(eis_seq)):
        _x = eis_seq[i]
        ch_eis = chData[_x,:,:]
        _color = cmap(eis_cluster[i])
        axis[8].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"{chr(ord('A')+eis_cluster[i])}")
        
    _legend_handle = []
    for i in range(len(np.unique(eis_cluster))):
        _legend_handle.append(mpatches.Patch(color = cmap(i), label = f"{chr(ord('A')+i)}:{len(eis_cluster[eis_cluster==i])}"))
    axis[8].legend(handles=_legend_handle)
    axis[8].sharex(axis[0])
    axis[8].sharey(axis[0])


    # Outlier
    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(len(eis_anomaly)):
        _x = eis_anomaly[i]
        ch_eis = chData[_x,:,:]
        _color = cmap(_x/np.shape(chData)[0])
        axis[9].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{_x:02d}")
    axis[9].legend()
    axis[9].sharex(axis[0])
    axis[9].sharey(axis[0])

    # Interpolation Trace
    cmap = plt.colormaps.get_cmap('viridis_r')
    for i in range(np.shape(y_eval)[1]):
        axis[10].fill_between(x_eval, y_eval[:,i,0] - 2*np.sqrt(y_eval_err[:,i,0]), y_eval[:,i,0] + 2*np.sqrt(y_eval_err[:,i,0]), 
                        alpha=0.2, color = cmap(i/np.shape(y_eval)[1]))
        
        axis[10].plot(x_train, y_train[:,i,0], color = cmap(i/np.shape(y_eval)[1]), linestyle = ' ', marker = 'o')

        


