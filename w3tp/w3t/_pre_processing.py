import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def rectsqueeze(t, t1, t2, k):
    """
    Mimics MATLAB's rectsqueeze function to ensure smooth start and end.
    """
    return 1 / (1 + np.exp(-2 * k * (t - t1))) - 1 / (1 + np.exp(-2 * k * (t - t2)))


def mccholesky(omegaaxisinput, SS, Nsim=1, domegasim=0.0001, FileName='No'):
    """
    Simulates realizations (time series) using the cross-spectral density matrix SS.
    
    Parameters:
    omegaaxisinput (array): Frequency axis input.
    SS (array): Cross-spectral density matrix (Ndof, Ndof, Nomegaaxisinput).
    Nsim (int): Number of simulations. Default is 1.
    domegasim (float): Frequency step for simulations. Default is 0.0001.
    FileName (str): Name of the output file. If 'No', returns results as a list.
    """
    GG = np.zeros_like(SS, dtype=complex)
    for n in range(len(omegaaxisinput)):
        if np.max(np.abs(SS[:, :, n])) < 1.0e-10:
            continue
        GG[:, :, n] = np.linalg.cholesky(SS[:, :, n])
    
    omegaaxissim = np.arange(domegasim, max(omegaaxisinput) + domegasim, domegasim)
    NFFT = 2 ** int(np.ceil(np.log2(2 * len(omegaaxissim))))
    t = np.linspace(0, 2 * np.pi / domegasim, NFFT)
    XX = [t] if FileName == 'No' else []
    
    for z in range(Nsim):
        phi = 2 * np.pi * np.random.rand(SS.shape[0], len(omegaaxissim))
        x = np.zeros((SS.shape[0], NFFT))
        
        for m in range(GG.shape[0]):
            for n in range(m + 1):
                c = np.interp(omegaaxissim, omegaaxisinput, GG[m, n, :]) * np.exp(1j * phi[n, :])
                x[m, :] += np.real(np.fft.ifft(c, NFFT)) * NFFT * np.sqrt(2 * domegasim)
        
        if FileName == 'No':
            XX.append(x)
        else:
            np.save(f"{FileName}_Nr_{z+1}.npy", {'t': t, 'x': x})
    
    print("Done")
    return XX if FileName == 'No' else None


def matrix2acs(t, TS, FileName,plot=True):
    """
    Converts motion data into ACS-compatible format and saves it to a .txt file.
    
    Parameters:
    t (array): Time axis for input motion, resampled to 1000 Hz.
    TS (array): Nx3 matrix containing displacements and rotations in [mm, mm, rad].
    FileName (str): Name of the output .txt file.
    """
    # Resampling and smoothing
    t1, t2 = 5, max(t) - 5
    Ri = rectsqueeze(t, t1, t2, 1)
    TS *= Ri[:, np.newaxis]
    
    t1, t2 = 1, max(t) - 1
    Ri = rectsqueeze(t, t1, t2, 10)
    TS *= Ri[:, np.newaxis]
    
    # Resampling to 1000 Hz
    ti = np.arange(0, max(t), 1 / 1000)
    TS_RS = np.array([interp.interp1d(t, TS[:, i], kind='cubic')(ti) for i in range(3)]).T
    
    # Convert to degrees
    TS_RS_tmp = TS_RS.copy()
    TS_RS_tmp[:, 2] *= 360 / (2 * np.pi)
    TS_RS_tmp_dot = np.vstack(([0, 0, 0], np.diff(TS_RS_tmp, axis=0) * 1000))
    TS_RS_tmp_2dot = np.vstack(([0, 0, 0], np.diff(TS_RS_tmp_dot, axis=0))) * 1000
    
    # Check constraints
    if np.max(TS_RS_tmp[:, :2]) > 95 or np.max(TS_RS_tmp[:, 2]) > 89:
        print("Too large displacements or rotations")
        return
    
    if np.max(TS_RS_tmp_dot[:, :2]) > 20 * (3.6 * 2 * np.pi) or np.max(TS_RS_tmp_dot[:, 2]) > 4 * (3.6 * 2 * np.pi):
        print("Too large velocity or angular velocity")
        return
    
    if np.max(TS_RS_tmp_2dot[:, :2]) > 20 * (3.6 * 2 * np.pi) ** 2 or np.max(TS_RS_tmp_2dot[:, 2]) > 6 * (3.6 * 2 * np.pi) ** 2:
        print("Too large acceleration or angular acceleration")
        return
    
    # Convert data to rotation of servomotors in degrees
    TS_RS_ACS = np.zeros((6, len(ti)))
    TS_RS_ACS[0:2, :] = TS_RS[:, 0] * 360 / 10  # Horizontal motion
    TS_RS_ACS[2:4, :] = TS_RS[:, 1] * 360 / 10  # Vertical motion
    TS_RS_ACS[4:6, :] = -TS_RS[:, 2] * 360 / (2 * np.pi)  # Torsional motion
    
    # Save to file
    np.savetxt(FileName, TS_RS_ACS, delimiter=' ', fmt='%.6f')

    if plot:

        # Make 3x3 figures with disp, velo, accel
        fig, axs = plt.subplots(3, 3, figsize=(12, 8),sharex=True)
        axs[0, 0].plot(ti, TS_RS[:,0])
        axs[0,0].set_ylabel("$mm$")
        axs[1, 0].plot(ti, TS_RS[:,1])
        axs[1,0].set_ylabel("$mm/s$")
        axs[2, 0].plot(ti, TS_RS[:,2])
        axs[2,0].set_ylabel("$deg/s^2$")



        axs[0, 1].plot(ti, TS_RS_tmp_dot[:, 0])
        axs[0,1].set_ylabel("$mm$")
        axs[1, 1].plot(ti, TS_RS_tmp_dot[:, 1])
        axs[1,1].set_ylabel("$mm/s$")
        axs[2, 1].plot(ti, TS_RS_tmp_dot[:, 2])
        axs[2,1].set_ylabel("$deg/s^2$")


        axs[0, 2].plot(ti, TS_RS_tmp_2dot[:, 0])
        axs[0,2].set_ylabel("$mm/s^2$")
        axs[1, 2].plot(ti, TS_RS_tmp_2dot[:, 1])
        axs[1,2].set_ylabel("$mm/s^2$")
        axs[2, 2].plot(ti, TS_RS_tmp_2dot[:, 2])
        axs[2,2].set_ylabel("$deg/s^2$")

        for k1 in range(3):
            for k2 in range(3):
                axs[k1,k2].grid(True)
                if k1 == 2:
                    axs[k1,k2].set_xlabel("Time [s]")
        
        plt.tight_layout()




