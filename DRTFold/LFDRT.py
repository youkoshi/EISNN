import numpy as np
from scipy.linalg import eig, svd, solve


import matplotlib.pyplot as plt






def Loewner_Framework(f, Z, REALFLAG = True):
    '''==================================================
        Construct Loewner Pencel
        Parameter: 
            f:          real array of frequency values
            Z:          complex array of impedance values (H = Z)
            REALFLAG:   boolean flag to indicate if the model should have real entries
        Returen:
            L:          Loewner matrix
            Ls:         Shifted Loewner matrix
            H_left:     Impedance values for group left
            H_right:    Impedance values for group right
        ==================================================
    '''
    _n = len(f)
    s = 2j * np.pi * f

    # Ensuring the input have an even number of elements 
    # for constructing the model having real entries
    if REALFLAG:
        if _n % 2 != 0:
            _n = _n - 1

    # Left & Right Data for Loewner Framework
    s_left  = s[:_n:2]
    H_left  = Z[:_n:2]
    s_right = s[1:_n:2] 
    H_right = Z[1:_n:2]

    # Construct complex conjugate values for ensuring model having real entries
    if REALFLAG:
        s_left  = np.stack([s_left, s_left.conj()], axis=1).flatten()
        H_left  = np.stack([H_left, H_left.conj()], axis=1).flatten()
        s_right = np.stack([s_right, s_right.conj()], axis=1).flatten()
        H_right = np.stack([H_right, H_right.conj()], axis=1).flatten()

    # Constructing the Loewner Matrix & Shifted Loewner Matrix
    # L   = (H_left[:,None] - H_right[None,:]) / (s_left[:,None] - s_right[None,:])
    # Ls  = (s_left[:,None] * H_left[:,None] - s_right[None,:] * H_right[None,:]) / (s_left[:,None] - s_right[None,:])
    L   = (H_left[None,:] - H_right[:,None]) / (s_left[None,:] - s_right[:,None])
    Ls  = (s_left[None,:] * H_left[None,:] - s_right[:,None] * H_right[:,None]) / (s_left[None,:] - s_right[:,None])

    # Transforming the conplex L & Ls to obtain matrices with real entries
    if REALFLAG:
        _J_diag = np.eye(_n//2)
        _J  = (1/np.sqrt(2)) * np.array([[1, 1j], [1, -1j]])
        _J  = np.kron(_J_diag, _J)

        L       = (_J.conj().T @ L @ _J).real
        Ls      = (_J.conj().T @ Ls @ _J).real
        H_left  = ((_J.T @ H_left).T).real
        H_right = (_J.conj().T @ H_right).real

        
    return L, Ls, H_left, H_right

def state_space_model(L, Ls, H_left, H_right):
    '''==================================================
        Construct state space model from Loewner Pencel
        Parameter: 
            L:          Loewner matrix
            Ls:         Shifted Loewner matrix
            H_left:     Impedance values for group left
            H_right:    Impedance values for group right
        Returen:
            Ek, Ak, Bk, Ck:
                Ek x' = Ak x + Bk u
                   y  = Ck x + Dk u (Dk = 0)
        ==================================================
    '''
    # rank of the Loewner Pencel
    _rank = np.linalg.matrix_rank(np.concatenate((L, Ls), axis=0))
    Y_L, svd_L, X_L = svd(L, full_matrices=False, lapack_driver='gesvd')
    X_L = X_L.T
    
    # Reduced state space model interpolating the data
    Yk = Y_L[:, :_rank]
    Xk = X_L[:, :_rank]

    Ek = -Yk.T@L@Xk
    Ak = -Yk.T@Ls@Xk
    Bk = Yk.T@H_right
    Ck = H_left.T@Xk

    return Ek, Ak, Bk, Ck

def DRT_Transform(Ek, Ak, Bk, Ck, REALFLAG = True, real_th = 1e3):
    '''==================================================
        Transform state space model to DRT model
        Parameter: 
            Ek, Ak, Bk, Ck:
                Ek x' = Ak x + Bk u
                   y  = Ck x + Dk u (Dk = 0)
        Returen:
            R_i:    R_i from RC pair in DRT
            C_i:    C_i from RC pair in DRT
            tau_i   tau_i from RC pair in DRT
        ==================================================
    '''
    # Solve Av= λEv & wT A= λ wT E & Res = CvwB/wEv, wEv =  δ
    _pol, _U = eig(Ak, Ek)     # 
    wB = solve(_U, solve(Ek,Bk))
    Cv = Ck @ _U
    _res = Cv * wB

    # Calculate R_i & tau_i
    R_i     = (-_res / _pol)
    C_i     = (1/_res)
    tau_i   = (-1/_pol) 
    # tau_i   = abs(-1/_pol) 

    if REALFLAG:
        real_ratio = np.where(np.abs(tau_i.imag) == 0, np.inf, np.abs(tau_i.real / (tau_i.imag+1e-20)))
        tau_i = np.abs(tau_i[real_ratio > real_th])
        R_i = np.abs(R_i[real_ratio > real_th])
        C_i = np.abs(C_i[real_ratio > real_th])


    return R_i, C_i, tau_i

def DRT_Reconstruction_SSM(Ek, Ak, Bk, Ck, f, Z):
    '''==================================================
        Reconstruct DRT from state space model
        Parameter: 
            R_i:    R_i from RC pair in DRT
            tau_i   tau_i from RC pair in DRT
            f:  real array of frequency values
            Z:  complex array of impedance values (H = Z)
        Returen:
            H:  reconstructed impedance values
        ==================================================
    '''
    s = 2j * np.pi * f
    H = np.array([Ck @ solve(si * Ek - Ak, Bk) for si in s])
    res_ReZ = np.abs(((Z.real - H.real) / np.abs(Z))) * 100
    res_ImZ = np.abs(((Z.imag - H.imag) / np.abs(Z))) * 100

    return H, res_ReZ, res_ImZ


def DRT_Reconstruction_DRT(R_i, tau_i, f, Z):
    '''==================================================
        Reconstruct DRT from state space model
        Parameter: 
            Ek, Ak, Bk, Ck:
                Ek x' = Ak x + Bk u
                   y  = Ck x + Dk u (Dk = 0)
            f:  real array of frequency values
            Z:  complex array of impedance values (H = Z)
        Returen:
            H:  reconstructed impedance values
        ==================================================
    '''
    s = 2j * np.pi * f  # Broadcasting tau_i to match f
    _RC = R_i[None, :] / (1+s[:,None] * tau_i[None,:])
    H = np.sum(_RC, axis=1)

    res_ReZ = np.abs(((Z.real - H.real) / np.abs(Z))) * 100
    res_ImZ = np.abs(((Z.imag - H.imag) / np.abs(Z))) * 100

    return H, res_ReZ, res_ImZ

def DRT_singularity_analysis(f, Z, REALFLAG = True):
    '''==================================================
        DRT Singularity Analysis
        Parameter: 
            f:          real array of frequency values
            Z:          complex array of impedance values (H = Z)
            REALFLAG:   boolean flag to indicate if the model should have real entries
        Returen:
            R_i:        R_i from RC pair in DRT
            tau_i:      tau_i from RC pair in DRT
        ==================================================
    '''
    L, Ls, H_left, H_right = Loewner_Framework(f, Z, REALFLAG)
    Y, svd_L, X = svd(np.concatenate([L, Ls]), full_matrices=False)

    return svd_L

def DRT_Analysis_Single(f, Z, REALFLAG = True):
    '''==================================================
        DRT Analysis
        Parameter: 
            f:          real array of frequency values
            Z:          complex array of impedance values (H = Z)
            REALFLAG:   boolean flag to indicate if the model should have real entries
        Returen:
            R_i:        R_i from RC pair in DRT
            tau_i:      tau_i from RC pair in DRT
            H:          reconstructed impedance values
            res_ReZ:    relative error of real part of impedance values
            res_ImZ:    relative error of imaginary part of impedance values
        ==================================================
    '''
    L, Ls, H_left, H_right = Loewner_Framework(f, Z, REALFLAG)
    Ek, Ak, Bk, Ck = state_space_model(L, Ls, H_left, H_right)
    R_i, C_i, tau_i = DRT_Transform(Ek, Ak, Bk, Ck)
    # H, res_ReZ, res_ImZ = DRT_Reconstruction_SSM(Ek, Ak, Bk, Ck, f, Z)
    H, res_ReZ, res_ImZ = DRT_Reconstruction_DRT(R_i, tau_i, f, Z)

    return R_i, C_i, tau_i, H, res_ReZ, res_ImZ
    
def DRT_Analysis_Batch(chData, REALFLAG = True):
    '''==================================================
        DRT Analysis for Batch Data
        Parameter: 
            chData:     list of tuples (f, Z) for each channel
            REALFLAG:   boolean flag to indicate if the model should have real entries
        Returen:
            results:    list of tuples (R_i, C_i, tau_i, H, res_ReZ, res_ImZ) for each channel
        ==================================================
    '''
    DRTdata = []
    f = chData[0,0,:]
    for i in range(chData.shape[0]):
        _Z = chData[i,1,:] + 1j*chData[i,2,:]
        R_i, C_i, tau_i, _, _, _ = DRT_Analysis_Single(f, _Z, REALFLAG)
        DRTdata.append((np.array([R_i, C_i, tau_i])))
    return DRTdata


def DRT_Single_Plot(f, Z_plot, H_plot, R_plot, C_plot, tau_plot, res_ReZ_plot, res_ImZ_plot, svd_L_plot):
    '''==================================================
        Plot DRT Analysis Results
        Parameter: 
            f:              real array of frequency values
            Z_plot:         complex array of impedance values (H = Z)
            H_plot:         reconstructed impedance values
            R_plot:         R_i from RC pair in DRT
            tau_plot:       tau_i from RC pair in DRT
            res_ReZ_plot:   relative error of real part of impedance values
            res_ImZ_plot:   relative error of imaginary part of impedance values
            svd_L_plot:     singular values from Loewner Pencel
        ==================================================
    '''

    fig, axis = plt.subplots(2,3,figsize=(12,8), constrained_layout=True)
    cmp = plt.colormaps.get_cmap('rainbow_r')
    axis[0,0].plot(Z_plot.real, -Z_plot.imag, label = "EIS")
    axis[0,0].plot(H_plot.real, -H_plot.imag, label = "DRT")
    axis[0,0].set_aspect('equal')


    axis[0,1].loglog(f, np.abs(Z_plot), label = "EIS")
    axis[0,1].loglog(f, np.abs(H_plot), label = "DRT")
    axis[0,2].semilogx(f, -np.angle(Z_plot), label = "EIS")
    axis[0,2].semilogx(f, -np.angle(H_plot), label = "DRT")

    axis[1,0].stem(tau_plot, R_plot, linefmt='-.', markerfmt='o', basefmt='-')
    axis[1,0].set_xscale('log')
    axis[1,0].set_yscale('log')
    axis[1,0].set_xlabel(r'$\tau_i\ [s]$')
    axis[1,0].set_ylabel(r'$R_i\ [\Omega]$')  

    # 右轴：C_plot
    ax2 = axis[1,0].twinx()  # 创建共享x轴的右侧y轴
    ax2.stem(tau_plot, C_plot, basefmt='-', linefmt='C1-.', markerfmt='C1x')
    ax2.set_ylabel(r'$C_i\ [F]$')
    ax2.set_yscale('log')

    axis[1,1].plot(f,res_ReZ_plot, ':+', label = "Re(Z)")
    axis[1,1].plot(f,res_ImZ_plot, '--o', label = "Im(Z)")
    axis[1,1].set_xscale('log')

    axis[1,2].plot(svd_L_plot/svd_L_plot[0], ':*')
    axis[1,2].set_yscale('log')

    return fig


def DRT_Plot_Batch(fig, DRTdata, chData):
    _w = 0.1
    _h = 0.1
    axis = [0] * 5
    axis[0] = fig.add_axes([0.0625,0.5,0.25,0.4])
    axis[1] = fig.add_axes([0.375,0.5,0.25,0.4])
    axis[2] = fig.add_axes([0.6875,0.5,0.25,0.4])
    
    axis[3] = fig.add_axes([0.0625,0.05,0.4,0.4])
    axis[4] = fig.add_axes([0.5375,0.05,0.4,0.4])
    # axis[3] = fig.add_axes([0.05,0.05,0.2,0.4])
    # axis[4] = fig.add_axes([0.5,0.05,0.2,0.4])


    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(chData.shape[0]):
        ch_eis = chData[i,:,:]
        ch_drt = DRTdata[i]
        _color = cmap(i/chData.shape[0])

        axis[0].plot(ch_eis[1,:], -ch_eis[2,:], color = _color, linewidth=2)
        axis[1].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2)
        axis[2].semilogx(ch_eis[0,:], np.rad2deg(np.angle(ch_eis[1,:]+1j*ch_eis[2,:])), color = _color, linewidth=2)

        _ml, _sl, _bl = axis[3].stem(ch_drt[2,:], ch_drt[0,:], linefmt='-.', markerfmt='o', basefmt='-')
        plt.setp(_ml, color=_color, linewidth=2)
        plt.setp(_sl, color=_color, linewidth=2)
        plt.setp(_bl, color=_color, alpha=0.5)

        
        _ml, _sl, _bl = axis[4].stem(ch_drt[2,:], ch_drt[1,:], linefmt='-.', markerfmt='o', basefmt='-')
        plt.setp(_ml, color=_color, linewidth=2)
        plt.setp(_sl, color=_color, linewidth=2)
        plt.setp(_bl, color=_color, alpha=0.5)

    axis[0].set_aspect('equal', adjustable='datalim')
    axis[3].set_xscale('log')
    axis[3].set_yscale('log')
    axis[4].set_xscale('log')
    axis[4].set_yscale('log')





def DRT_Plot_Batch_3D(fig, DRTdata, chData):
    axis = [0] * 5
    axis[0] = fig.add_axes([0.0625,0.5,0.25,0.4])
    axis[1] = fig.add_axes([0.375,0.5,0.25,0.4])
    axis[2] = fig.add_axes([0.6875,0.5,0.25,0.4])
    
    axis[3] = fig.add_axes([0.0625,0.05,0.4,0.4], projection='3d')
    axis[4] = fig.add_axes([0.5375,0.05,0.4,0.4], projection='3d')

    z_offset = 0.2
    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(chData.shape[0]):
        ch_eis = chData[i,:,:]
        ch_drt = DRTdata[i]
        _color = cmap(i/chData.shape[0])

        axis[0].plot(ch_eis[1,:], -ch_eis[2,:], color = _color, linewidth=2)
        axis[1].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2)
        axis[2].semilogx(ch_eis[0,:], np.rad2deg(np.angle(ch_eis[1,:]+1j*ch_eis[2,:])), color = _color, linewidth=2)

        z = np.ones_like(ch_drt[2, :]) * i * z_offset
        x = np.log10(ch_drt[2, :])  
        y1 = np.log10(ch_drt[0, :])  # R_i
        y2 = np.log10(ch_drt[1, :])  # C_i

        # for xi, yi, zi in zip(x, y1, z):
        #     axis[3].plot([xi, xi], [zi, zi], [0, yi], color=_color, linewidth=1.5)
        # for xi, yi, zi in zip(x, y2, z):
        #     axis[4].plot([xi, xi], [zi, zi], [0, yi], color=_color, linewidth=1.5)

        axis[3].scatter(x, z, y1, color=_color, s=20)
        axis[4].scatter(x, z, y2, color=_color, s=20)

        

    axis[0].set_aspect('equal', adjustable='datalim')
