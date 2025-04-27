# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:17:09 2022

@author: oiseth
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:09:00 2021

@author: oiseth
"""
import numpy as np
from scipy import signal as spsp
from scipy import linalg as spla
from scipy import special as spspes
from matplotlib import pyplot as plt
from copy import deepcopy
from ._exp import Experiment
import pandas as pd
import os
import csv



__all__ = ["AerodynamicDerivatives2x2","AerodynamicDerivative2x2",]

    
   
class AerodynamicDerivative2x2:
    """ 
    A class used to represent a aerodynamic derivative
    
    Arguments
    ---------
    reduced_velocities  : float
        reduced velocities
    ad_load_cell_a      : float
        contribution to aerodynamic derivative from load cell a
    ad_load_cell_b      : float
        contribution to aerodynamic derivative from load cell b
    mean_wind_speeds    : float
        mean wind velocities
    frequencies         : float
        frequencies of the motions applied to obtain ads
    label               : str
        aerodynamic derivative label
    ---------
    
    Methods:
    --------
    value : property
        Returns the total aerodynamic derivative (sum of contributions from both load cells).
        
    poly_fit(damping=True, order=2)
        Fits a polynomial of specified order to the aerodynamic derivative vs reduced velocity data.
        
    plot(mode="total", conv="normal", ax=[], V=1.0, damping=True, order=2)
        Plots the aerodynamic derivative and optionally overlays a polynomial fit.

    plot2(mode="poly only", conv="normal", ax=[], damping=True, order=2, valid=True, ADlabel='i')
        Alternative plotting method focused on polynomial fit and support for invalid results.
    --------
    
    """
    def __init__(self,label="x",reduced_velocities=[],ad_load_cell_a=[],ad_load_cell_b=[],mean_wind_speeds=[], frequencies=[]):
        """  
            
        Arguments
        ---------
        reduced_velocities  : float
            reduced velocities
        ad_load_cell_a      : float
            contribution to aerodynamic derivative from load cell 1
        ad_load_cell_b      : float
            contribution to aerodynamic derivative from load cell 2
        mean_wind_speeds    : float
            mean wind velocities
        frequencies         : float
            frequencies of the motions applied to obtain ads
        label               : str
            aerodynamic derivative label
        ---------
        
        """
        self.reduced_velocities = reduced_velocities
        self.ad_load_cell_a = ad_load_cell_a
        self.ad_load_cell_b = ad_load_cell_b
        self.mean_wind_speeds = mean_wind_speeds
        self.frequencies = frequencies
        self.label = label
    
    @property    
    def value(self):
        return self.ad_load_cell_a + self.ad_load_cell_b
    
    def poly_fit(self, damping=True, order=2):
        """
        Fits a polynomial to the aerodynamic derivative data.

        Parameters:
        -----------
        damping : bool, optional
            Indicates whether the aerodynamic derivative corresponds to a damping term (True) 
            or a stiffness term (False). Currently not used for changing the fit behavior,
            but kept for future use.

        order : int, optional
            Order of the polynomial to fit.

        Returns:
        --------
        poly_coeff : np.ndarray or None
            Polynomial coefficients of the fitted curve.

        k_range : np.ndarray or None
            Range of reduced frequency (1/V_r) used in the fitting, with min and max values.
        """
        # Extract aerodynamic derivative values and reduced velocities
        ads = self.value
        vreds = self.reduced_velocities

        # Proceed only if there is data to fit
        if ads.size > 0:
            # Initialize polynomial coefficient array and k_range (reduced frequency range)
            poly_coeff = np.zeros(np.max(order) + 1)
            k_range = np.zeros(2)

            # Define the range of reduced frequency (1 / V_r)
            k_range[0] = 1 / np.max(vreds)
            k_range[1] = 1 / np.min(vreds)

            # Fit a polynomial of specified order to the data
            # The 'damping' parameter is currently not altering logic
            poly_coeff = np.polyfit(vreds, ads, order)

            return poly_coeff, k_range
        else:
            # Return None if no data is available
            return None, None
        
        
    def plot(self, mode="total", conv="normal", ax=[], V=1.0, damping=True, order=2):
        """
        Plots the aerodynamic derivative as a function of reduced velocity,
        with optional polynomial fitting overlay.

        Parameters:
        -----------
        mode : str, optional
            Determines the plot mode. Options:
            - "total": plots raw aerodynamic derivative data.
            - "velocity": same as "total" but includes wind speed label.
            - "total+poly": plots both raw data and a polynomial fit.

        conv : str, optional
            Convention flag for how data is interpreted (currently unused).

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If not provided, a new figure and axes are created.

        V : float, optional
            Wind speed value used for labeling in "velocity" mode.

        damping : bool, optional
            Indicates whether the AD is associated with damping (True) or stiffness (False).

        order : int, optional
            Order of the polynomial fit if enabled.
        """

        # Fit a polynomial to the data
        poly_coeff, k_range = self.poly_fit(damping=damping, order=order)
        
        # If no data is available, use placeholders
        if poly_coeff is None:
            V = 0
            y = 0
        else:
            # Evaluate polynomial over a fine velocity range
            p = np.poly1d(poly_coeff)
            V = np.linspace(1 / k_range[1], 1 / k_range[0], 200)
            y = p(V)

        # If no axis is provided, create a new figure and axis
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Plot based on selected convention and mode
        if conv == "normal":
            if mode == "total":
                # Plot total measured aerodynamic derivative
                ax.plot(self.reduced_velocities, self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
                ax.set_ylabel("$" + self.label + "$")
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid(True)

            elif mode == "velocity":
                # Plot with velocity annotation
                ax.plot(self.reduced_velocities, self.ad_load_cell_a + self.ad_load_cell_b, "o", label=f"V = {V:.1f} m/s")
                ax.set_ylabel("$" + self.label + "$")
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)

            elif mode == "total+poly":
                # Plot both total values and polynomial fit
                ax.plot(self.reduced_velocities, self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
                ax.plot(V, y, label=f"Poly fit (order {order})")
                ax.set_ylabel("$" + self.label + "$")
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)
            


    def plot2(self, mode='poly only', conv="normal", ax=[], damping=True, order=2, valid=True, ADlabel='i'):
        """
        Plots the aerodynamic derivative as a function of reduced velocity,
        with optional polynomial fitting overlay.

        Parameters:
        -----------
        mode : str, optional
            Selects the plotting mode: "total" for raw data only, or "total+poly" to include polynomial fit.

        conv : str, optional
            Convention flag for how data is interpreted (currently unused).

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If not provided, a new figure and axes are created.

        damping : bool, optional
            Indicates whether the AD is associated with damping (True) or stiffness (False).

        order : int, optional
            Order of the polynomial fit if enabled.

        valid : bool, optional
            If False, plots a dummy point (e.g., used to indicate an invalid or placeholder result).
            If True, plots the fitted polynomial.

        ADlabel : str, optional
            Label for the aerodynamic derivative to display on the y-axis.

        """

        # Fit a polynomial to the data
        poly_coeff, k_range = self.poly_fit(damping=damping, order=order)

        # If no data is available, use placeholders
        if poly_coeff is None:
            V = 0
            y = 0
        else:
            # Evaluate polynomial over a fine velocity range
            p = np.poly1d(poly_coeff)
            V = np.linspace(1 / k_range[1], 1 / k_range[0], 200)
            y = p(V)

        # If no axis is provided, create a new figure and axis
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Plot total measured aerodynamic derivative
        if valid == False:
            ax.plot(0, 0, label='Single')
            ax.set_ylabel("$" + ADlabel + "$")
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.legend()
            ax.grid(True)

        # Plot both total values and polynomial fit
        else:
            ax.plot(V, y, label='Single')
            ax.set_ylabel("$" + ADlabel + "$")
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.legend()
            ax.grid(True)

    def get_points(self):
        """
        Returns the aerodynamic derivative data points as a tuple.

        Returns:
        --------
        tuple:
            - reduced_velocities: Reduced velocities.
            - ad_load_cell_a: Contribution from load cell A.
            - ad_load_cell_b: Contribution from load cell B.
        """
        AD = self.ad_load_cell_a + self.ad_load_cell_b

        return self.reduced_velocities, AD


class AerodynamicDerivatives2x2:
    """
    A class used to represent all aerodynamic derivatives for a 2 dof motion
    
    parameters:
    ----------
    h1...h4 : obj
        aerodynamic derivatives related to the vertical self-excited force
    a1...a4 : obj
        aerodynamic derivative related to the pitching moment
    ---------   
    
    methods:
    -------
    .fromWTT()
        obtains aerodynamic derivatives from a sequence of single harmonic wind tunnel tests
    .append()
        appends an instance of the class AerodynamicDerivtives to self    
    .plot()
        plots all aerodynamic derivatives    
    """

    def __init__(self, h1=None, h2=None, 
                 h3=None, h4=None, a1=None, a2=None, a3=None, a4=None, meanV=None):
        """
        Initializes the aerodynamic derivatives for a 2-degree-of-freedom system.

        parameters:
        ----------
        h1...h4 : obj
         aerodynamic derivatives related to the vertical self-excited force
        a1...a4 : obj
         aerodynamic derivative related to the pitching moment
        meanV : float or None
            Mean wind speed used for normalization or fitting purposes.
        ---------
        """
        
        self.h1 = h1 or AerodynamicDerivative2x2(label="H_1^*")
        self.h2 = h2 or AerodynamicDerivative2x2(label="H_2^*")
        self.h3 = h3 or AerodynamicDerivative2x2(label="H_3^*")
        self.h4 = h4 or AerodynamicDerivative2x2(label="H_4^*")

        self.a1 = a1 or AerodynamicDerivative2x2(label="A_1^*")
        self.a2 = a2 or AerodynamicDerivative2x2(label="A_2^*")
        self.a3 = a3 or AerodynamicDerivative2x2(label="A_3^*")
        self.a4 = a4 or AerodynamicDerivative2x2(label="A_4^*")

        self.meanV = meanV
    
        
    @classmethod
    def fromWTT(cls, experiment_in_still_air, experiment_in_wind, section_width, section_length, filter_order=6, cutoff_frequency=7):
        """
        Creates an instance of AerodynamicDerivatives4x4 from wind tunnel experiments.

        Parameters:
        ----------
        experiment_in_still_air : Experiment
            Experiment object containing test data in still air.
        experiment_in_wind : Experiment
            Experiment object containing test data in wind.
        section_width : float
            Width of the bridge deck section model.
        section_length : float
            Length of the section model.
        upstream_in_rig : bool, optional
            Indicates whether the upstream section is placed in the rig (default is True).
        filter_order : int, optional
            Order of the Butterworth filter used for signal processing (default is 6).
        cutoff_frequency : float, optional
            Cutoff frequency of the Butterworth filter (default is 7 Hz).

        Returns:
        --------
        tuple:
        - An instance of AerodynamicDerivatives2x2 containing identified coefficients.
        - An Experiment object for model predictions.
        - An Experiment object with wind-only forces (still-air subtracted).
        """

        # Align wind experiment with still air
        experiment_in_wind.align_with(experiment_in_still_air)

        # Copy and subtract still-air forces from wind test to isolate aerodynamic forces
        experiment_in_wind_still_air_forces_removed = deepcopy(experiment_in_wind)
        experiment_in_wind_still_air_forces_removed.substract(experiment_in_still_air)

        # Get harmonic test intervals
        starts, stops = experiment_in_wind_still_air_forces_removed.harmonic_groups()

        # Preallocate arrays for output data
        frequencies_of_motion = np.zeros(len(starts))
        reduced_velocities = np.zeros(len(starts))
        mean_wind_speeds = np.zeros(len(starts))
        
        # Shape: [damping/stiffness, force type, test index, load cell index]
        normalized_coefficient_matrix = np.zeros((2, 3, len(starts), 4))
        
        # Storage for predicted forces
        forces_predicted_by_ads = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape[0], 24))

        # Loop over all single harmonic tests in the time series
        for k in range(len(starts)):
            # Compute sampling frequency based on time step
            sampling_frequency = 1 / (experiment_in_still_air.time[1] - experiment_in_still_air.time[0])

            # Apply a Butterworth filter to motion signals to clean high-frequency noise
            sos = spsp.butter(filter_order, cutoff_frequency, fs=sampling_frequency, output="sos")
            motions = experiment_in_wind_still_air_forces_removed.motion
            motions = spsp.sosfiltfilt(sos, motions, axis=0)

            # Compute time derivatives of the motions
            time_derivative_motions = np.vstack((np.array([0, 0, 0]), np.diff(motions, axis=0))) * sampling_frequency

            # Identify motion type (0 = horizontal, 1 = vertical, 2 = torsional)
            motion_type = experiment_in_wind_still_air_forces_removed.motion_type()

            # Perform FFT to extract motion frequency
            motion_segment = motions[starts[k]:stops[k], motion_type]
            fourier_amplitudes = np.fft.fft(motion_segment - np.mean(motion_segment))
            time_step = experiment_in_wind_still_air_forces_removed.time[1] - experiment_in_wind_still_air_forces_removed.time[0]
            frequencies = np.fft.fftfreq(len(fourier_amplitudes), time_step)

            # Find dominant motion frequency
            peak_index = np.argmax(np.abs(fourier_amplitudes[:len(fourier_amplitudes) // 2]))
            frequency_of_motion = frequencies[peak_index]
            frequencies_of_motion[k] = frequency_of_motion

            # Regressor matrix: [motion_dot, motion] for the active motion
            regressor_matrix = np.vstack((time_derivative_motions[starts[k]:stops[k], motion_type], motions[starts[k]:stops[k], motion_type])).T
            
            # Compute pseudoinverse of the regressor matrix (X^+)
            pseudo_inverse_regressor_matrix = spla.pinv(regressor_matrix)
            selected_forces = np.array([0, 2, 4])  # q_x, q_z, q_theta indices per load cell

            # Compute mean wind speed in current segment
            mean_wind_speed = np.mean(experiment_in_wind_still_air_forces_removed.wind_speed[starts[k]:stops[k]])
            mean_wind_speeds[k] = mean_wind_speed

            # Compute reduced frequency and reduced velocity
            reduced_frequency = frequency_of_motion * 2 * np.pi * section_width / mean_wind_speed
            reduced_velocities[k] = 1 / reduced_frequency

            # Loop through each load cell (4 total)
            for m in range(4):
                # Extract forces for current load cell
                forces = experiment_in_wind_still_air_forces_removed.forces_global_center[starts[k]:stops[k], selected_forces + 6 * m]

                # Remove mean wind force component
                forces_mean_wind_removed = forces - np.mean(
                    experiment_in_wind_still_air_forces_removed.forces_global_center[:400, selected_forces + 6 * m],
                    axis=0)

                # Compute raw coefficients: E = X^+ * q
                coefficient_matrix = pseudo_inverse_regressor_matrix @ forces_mean_wind_removed

                # Normalize the coefficients to aerodynamic derivative form
                normalized_coefficient_matrix[:, :, k, m] = np.copy(coefficient_matrix)
                normalized_coefficient_matrix[0, :, k, m] *= 2 / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed / reduced_frequency / section_width / section_length
                normalized_coefficient_matrix[1, :, k, m] *= 2 / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed ** 2 / reduced_frequency ** 2 / section_length
                normalized_coefficient_matrix[:, 2, k, m] /= section_width

                if motion_type == 2: 
                    # Special case: normalize all terms by section width for torsional tests
                    normalized_coefficient_matrix[:, :, k, m] /= section_width

                # Compute predicted forces from the model and add wind mean
                forces_predicted_by_ads[starts[k]:stops[k], selected_forces + 6 * m] += (regressor_matrix @ coefficient_matrix + np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[:400, selected_forces + 6 * m], axis=0))

        # Construct Experiment object for model prediction based on fitted aerodynamic model
        obj1 = experiment_in_wind_still_air_forces_removed
        obj2 = experiment_in_still_air
        model_prediction = Experiment(obj1.name, obj1.time, obj1.temperature, obj1.air_density, obj1.wind_speed, [], forces_predicted_by_ads, obj2.motion)

        # Instantiate empty derivative objects
        h1 = AerodynamicDerivative2x2()
        h2 = AerodynamicDerivative2x2()
        h3 = AerodynamicDerivative2x2()
        h4 = AerodynamicDerivative2x2()
        a1 = AerodynamicDerivative2x2()
        a2 = AerodynamicDerivative2x2()
        a3 = AerodynamicDerivative2x2()
        a4 = AerodynamicDerivative2x2()

        meanV = np.mean(mean_wind_speeds)

        # Based on the motion type, assign relevant aerodynamic derivatives
        if motion_type == 0:
            row = 0  # Placeholder case — not handled here

        elif motion_type == 1:  # Vertical
            row = 0
            col = 1
            h1 = AerodynamicDerivative2x2("H_1^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            col = 2
            a1 = AerodynamicDerivative2x2("A_1^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            row = 1
            col = 1
            h4 = AerodynamicDerivative2x2("H_4^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            col = 2
            a4 = AerodynamicDerivative2x2("A_4^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)

        elif motion_type == 2:  # Torsional
            row = 0
            col = 1
            h2 = AerodynamicDerivative2x2("H_2^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            col = 2
            a2 = AerodynamicDerivative2x2("A_2^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            row = 1
            col = 1
            h3 = AerodynamicDerivative2x2("H_3^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
            col = 2
            a3 = AerodynamicDerivative2x2("A_3^*", reduced_velocities, normalized_coefficient_matrix[row, col, :, 0], normalized_coefficient_matrix[row, col, :, 1], mean_wind_speeds, frequencies_of_motion)
        
        return cls(h1, h2, h3, h4, a1, a2, a3, a4, meanV), model_prediction, experiment_in_wind_still_air_forces_removed
    
    #Ikke endret og ikke benyttet
    @classmethod
    def from_Theodorsen(cls,vred):
        
        vred[vred==0] = 1.0e-10
        
        k = 0.5/np.abs(vred)

        j0 = spspes.jv(0,k)
        j1 = spspes.jv(1,k)
        y0 = spspes.yn(0,k)
        y1 = spspes.yn(1,k)

        a = j1 + y0
        b = y1-j0
        c = a**2 + b**2

        f = (j1*a + y1*b)/c
        g = -(j1*j0 + y1*y0)/c
        
        h1_value = -2*np.pi*f*np.abs(vred)
        h2_value = np.pi/2*(1+f+4*g*np.abs(vred))*np.abs(vred)
        h3_value = 2*np.pi*(f*np.abs(vred)-g/4)*np.abs(vred)
        h4_value = np.pi/2*(1+4*g*np.abs(vred))
        
        
        a1_value = -np.pi/2*f*np.abs(vred)
        a2_value = -np.pi/8*(1-f-4*g*np.abs(vred))*np.abs(vred)
        a3_value = np.pi/2*(f*np.abs(vred)-g/4)*np.abs(vred)
        a4_value = np.pi/2*g*np.abs(vred)
           
        h1 = AerodynamicDerivative2x2("H_1^*",vred, h1_value/2, h1_value/2, vred*0, vred*0)
        h2 = AerodynamicDerivative2x2("H_2^*",vred, h2_value/2, h2_value/2, vred*0, vred*0)
        h3 = AerodynamicDerivative2x2("H_3^*",vred, h3_value/2, h3_value/2, vred*0, vred*0)
        h4 = AerodynamicDerivative2x2("H_4^*",vred, h4_value/2, h4_value/2, vred*0, vred*0)
      
        a1 = AerodynamicDerivative2x2("A_1^*",vred, a1_value/2, a1_value/2, vred*0, vred*0)
        a2 = AerodynamicDerivative2x2("A_2^*",vred, a2_value/2, a2_value/2, vred*0, vred*0)
        a3 = AerodynamicDerivative2x2("A_3^*",vred, a3_value/2, a3_value/2, vred*0, vred*0)
        a4 = AerodynamicDerivative2x2("A_4^*",vred, a4_value/2, a4_value/2, vred*0, vred*0)
           
        return cls(h1, h2, h3, h4, a1, a2, a3, a4)
    
    #Ikke benyttet
    @classmethod
    def from_poly_k(cls,poly_k,k_range, vred):
        vred[vred==0] = 1.0e-10
        uit_step = lambda k,kc: 1./(1 + np.exp(-2*20*(k-kc)))
        fit = lambda p,k,k1c,k2c : np.polyval(p,k)*uit_step(k,k1c)*(1-uit_step(k,k2c)) + np.polyval(p,k1c)*(1-uit_step(k,k1c)) + np.polyval(p,k2c)*(uit_step(k,k2c))
        
        damping_ad = np.array([True, True, False, False, True, True, False, False])
        labels = ["H_1^*", "H_2^*", "H_3^*", "H_4^*",    "A_1^*", "A_2^*", "A_3^*", "A_4^*"]
        ads = []
        for k in range(8):
                      
            if damping_ad[k] == True:
                ad_value = np.abs(vred)*fit(poly_k[k,:],np.abs(1/vred),k_range[k,0],k_range[k,1])
            else:
                ad_value = np.abs(vred)**2*fit(poly_k[k,:],np.abs(1/vred),k_range[k,0],k_range[k,1])
                
            ads.append(AerodynamicDerivative2x2(labels[k],vred,ad_value/2 , ad_value/2 , vred*0, vred*0))
            
             
        return cls(ads[0], ads[1], ads[2], ads[3], ads[4], ads[5], ads[6], ads[7])
    
    
    def append(self,ads):
        """
        Appends the aerodynamic data from another AerodynamicDerivatives instance to the current instance.

        This method combines the time-series data and test parameters (load cell measurements, frequencies,
        mean wind speeds, and reduced velocities) from a second `AerodynamicDerivatives` object into the 
        corresponding attributes of the current object. The data from each of the eight internal modes 
        (h1–h4, a1–a4) is concatenated element-wise.

        Parameters:
        -----------
        ads : AerodynamicDerivatives
            An instance of the AerodynamicDerivatives class whose data will be appended to this instance.
        """

        objs1 = [self.h1, self.h2, self.h3, self.h4, self.a1, self.a2, self.a3, self.a4]
        objs2 = [ads.h1, ads.h2, ads.h3, ads.h4, ads.a1, ads.a2, ads.a3, ads.a4]
        
        for k in range(len(objs1)):
            objs1[k].ad_load_cell_a = np.append(objs1[k].ad_load_cell_a,objs2[k].ad_load_cell_a)
            objs1[k].ad_load_cell_b = np.append(objs1[k].ad_load_cell_b,objs2[k].ad_load_cell_b) 
            
            objs1[k].frequencies = np.append(objs1[k].frequencies,objs2[k].frequencies) 
            objs1[k].mean_wind_speeds = np.append(objs1[k].mean_wind_speeds,objs2[k].mean_wind_speeds) 
            objs1[k].reduced_velocities = np.append(objs1[k].reduced_velocities,objs2[k].reduced_velocities) 
            
    @property
    def ad_matrix(self):
        """ Returns a matrix of aerodynamic derivatives and reduced velocities
        
        Returns
        -------
        ads : float
        
        a matrix of aerodynamic derivatives [18 x N reduced velocities]
        
        vreds : float
        
        a matrix of reduced velocities [18 x N reduced velocities]
        
        """
        ads = np.zeros((8,self.h1.reduced_velocities.shape[0]))
        vreds = np.zeros((8,self.h1.reduced_velocities.shape[0]))
        ads[0,:] = self.h1.value
        ads[1,:] = self.h2.value
        ads[2,:] = self.h3.value
        ads[3,:] = self.h4.value
        
        ads[4,:] = self.a1.value
        ads[5,:] = self.a2.value
        ads[6,:] = self.a3.value
        ads[7,:] = self.a4.value
        
        vreds[0,:] = self.h1.reduced_velocities
        vreds[1,:] = self.h2.reduced_velocities
        vreds[2,:] = self.h3.reduced_velocities
        vreds[3,:] = self.h4.reduced_velocities
        
        vreds[4,:] = self.a1.reduced_velocities
        vreds[5,:] = self.a2.reduced_velocities
        vreds[6,:] = self.a3.reduced_velocities
        vreds[7,:] = self.a4.reduced_velocities
        
        return ads, vreds
    
    #Ikke endret og ikke benyttet
    def frf_mat(self,mean_wind_velocity = 1.0, section_width = 1.0, air_density = 1.25):
        
        
        frf_mat = np.zeros((2,2,len(self.p1.reduced_velocities)),dtype=complex)

        frf_mat[0,0,:] = 1/2*air_density*mean_wind_velocity**2 * (1/self.h1.reduced_velocities)**2 * (self.h1.value*1j + self.h4.value)
        frf_mat[0,1,:] = 1/2*air_density*mean_wind_velocity**2 * section_width*(1/self.h3.reduced_velocities)**2 * (self.h2.value*1j + self.h3.value)
        
        frf_mat[1,0,:] = 1/2*air_density*mean_wind_velocity**2 * section_width*(1/self.a1.reduced_velocities)**2 * (self.a1.value*1j + self.a4.value)
        frf_mat[1,1,:] = 1/2*air_density*mean_wind_velocity**2 * section_width**2*(1/self.a2.reduced_velocities)**2 * (self.a2.value*1j + self.a3.value)
        
        return frf_mat
    
    #Ikke benyttet
    def fit_poly_k(self,orders = np.ones(8,dtype=int)*2):
        ad_matrix, vreds = self.ad_matrix
        
        poly_coeff = np.zeros((8,np.max(orders)+1))
        k_range = np.zeros((8,2))
        
        damping_ad = np.array([True, True, False, False,  True, True, False, False])
        
        
        for k in range(8):
            k_range[k,0] = 1/np.max(vreds)
            k_range[k,1] = 1/np.min(vreds)
            
            if damping_ad[k] == True:
                poly_coeff[k,-orders[k]-1:] = np.polyfit(1/vreds[k,:],1/vreds[k,:]*ad_matrix[k,:],orders[k])
            elif damping_ad[k] == False:
                poly_coeff[k,-orders[k]-1:] = np.polyfit(1/vreds[k,:],(1/vreds[k,:])**2*ad_matrix[k,:],orders[k])
        
        return poly_coeff, k_range
    

    def fit_poly(self, orders=np.ones(8, dtype=int)*2):
        """
        Fits polynomial curves to the aerodynamic derivative data using specified polynomial orders.

        Returns:
        --------
        poly_coeff : ndarray of shape (8, max_order+1)
            Coefficient matrix for the fitted polynomials, one row per aerodynamic derivative.
        v_range : ndarray of shape (8, 2)
            Minimum and maximum reduced velocity values used for fitting each derivative.
        """
        
        # Retrieve aerodynamic derivatives (8xN) and corresponding reduced velocities (8xN)
        ad_matrix, vreds = self.ad_matrix

        # Initialize array to store polynomial coefficients
        poly_coeff = np.zeros((8, np.max(orders) + 1))
        
        # Initialize array to store [vmin, vmax] for each aerodynamic derivative
        v_range = np.zeros((8, 2))

        # Boolean mask to indicate which derivatives are damping and which are stiffness
        damping_ad = np.array([True, True, False, False, True, True, False, False])

        # Loop through each of the 8 aerodynamic derivatives
        for k in range(8):
            # Store min and max of reduced velocity used for fitting (same for all currently)
            v_range[k, 1] = np.max(vreds)
            v_range[k, 0] = np.min(vreds)

            # Fit a polynomial of order `orders[k]` to the k-th derivative
            poly_coeff[k, :] = np.polyfit(vreds[k, :], ad_matrix[k, :], orders[k])

        # Return the polynomial coefficients and velocity fitting ranges
        return poly_coeff, v_range


    #Ikke benyttet
    def to_excel(self,section_name, section_height=0, section_width=0, section_length=0):
        """
        Parameters
        ----------
        section_name : string
            section_name
        section_height : float64, optional
            Section height. The default is 0.
        section_width : float64, optional
            section width. The default is 0.
        section_length : float 64, optional
            section length. The default is 0.

        Returns
        -------
        None.

        """
        
        ad_value = pd.DataFrame({"H_1": self.h1.value,
                                 "H_2": self.h2.value,
                                 "H_3": self.h3.value,
                                 "H_4": self.h4.value,

                                 "A_1": self.a1.value,
                                 "A_2": self.a2.value,
                                 "A_3": self.a3.value,
                                 "A_4": self.a4.value,
                                 })

        ad_reduced_velocity = pd.DataFrame({"H_1": self.h1.reduced_velocities,
                                            "H_2": self.h2.reduced_velocities,
                                            "H_3": self.h3.reduced_velocities,
                                            "H_4": self.h4.reduced_velocities,

                                            "A_1": self.a1.reduced_velocities,
                                            "A_2": self.a2.reduced_velocities,
                                            "A_3": self.a3.reduced_velocities,
                                            "A_4": self.a4.reduced_velocities,
                                            })

        ad_mean_wind_speeds = pd.DataFrame({"H_1": self.h1.mean_wind_speeds,
                                            "H_2": self.h2.mean_wind_speeds,
                                            "H_3": self.h3.mean_wind_speeds,
                                            "H_4": self.h4.mean_wind_speeds,

                                            "A_1": self.a1.mean_wind_speeds,
                                            "A_2": self.a2.mean_wind_speeds,
                                            "A_3": self.a3.mean_wind_speeds,
                                            "A_4": self.a4.mean_wind_speeds,
                                            })

        geometry = pd.DataFrame({"D": [section_height],
                                 "B": [section_width],
                                 "L": [section_length]
                                 })

        with pd.ExcelWriter("ADs_" + section_name + '.xlsx') as writer:
            geometry.to_excel(writer, sheet_name="Dim section model")
            ad_value.to_excel(writer, sheet_name='Aerodynamic derivatives')
            ad_reduced_velocity.to_excel(
                writer, sheet_name='Reduced velocities')
            ad_mean_wind_speeds.to_excel(
                writer, sheet_name='Mean wind velocity')


    def polyfit_to_excel(self, test_name, save_path, orders=np.ones(8, dtype=int)*2):
        """
        Fits polynomial models to aerodynamic derivatives and exports the coefficients to an Excel file.

        Parameters:
        -----------
        test_name : str
            Name of the test case, used in the Excel file title.
        save_path : str
            Directory path where the Excel file will be saved.
        orders : np.ndarray of int, optional
            Polynomial order for each aerodynamic derivative (default is second-order for all 32 terms).

        The resulting Excel file will contain:
            - Label of each aerodynamic derivative (AD)
            - Polynomial order used
            - Fitting range of reduced velocity (V̂_min, V̂_max)
            - Polynomial coefficients (a₀, a₁, ..., aₙ)
        """

        # Fit polynomials to all aerodynamic derivatives using the specified polynomial orders
        poly_coeff, v_range = self.fit_poly(orders=orders)
        
        # Labels for each aerodynamic derivative
        labels = ["H_1*", "H_2*", "H_3*", "H_4*","A_1*", "A_2*", "A_3*", "A_4*"]
        
        data = []
        for i, label in enumerate(labels):
            # Get the polynomial coefficients for the i-th derivative and reverse their order (highest degree first)
            coeff_row = list(poly_coeff[i])[::-1]
            # Get the reduced velocity fitting range for this derivative
            vmin, vmax = v_range[i]
            # Create a row with label, polynomial order, velocity range, and coefficients
            row = [label, orders[i], vmin, vmax] + coeff_row
            data.append(row)

        # Determine the maximum polynomial order used (to know how many coefficient columns to create)
        max_order = max(orders)
        # Define column names including coefficient names a0, a1, ..., an
        col_names = ["AD label", "Order", "V̂_min", "V̂_max"] + [f"a{i}" for i in range(max_order+1)]
        # Create a pandas DataFrame from the data
        df = pd.DataFrame(data, columns=col_names)

        # Define filename and full output path
        filename = f"{test_name}_AD_polyfit.xlsx"
        full_path = os.path.join(save_path, filename)

        # Write DataFrame to Excel with some metadata and formula description
        with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, startrow=2)  # Write data starting from row 3 (0-based index)
            worksheet = writer.sheets["Sheet1"]
            worksheet.cell(row=1, column=1).value = f"Test Name: {test_name}"  # Add test name in row 1
            worksheet.cell(row=1, column=4).value = "f(V̂) = a₀ + a₁·V̂ + a₂·V̂² + ..."  # Add formula description

        print(f"Saved polynomial fit data to {full_path}")

        
    def plot(self, fig_damping=[], fig_stiffness=[], conv='normal', mode='total', orders=np.ones(8, dtype=int)*2):
        """
        Plots all 8 aerodynamic derivatives (4 damping + 4 stiffness) with optional polynomial fits.

        Parameters:
        -----------
        fig_damping : matplotlib.figure.Figure or list (optional)
            Figure object for damping plots. If not provided, a new figure is created with 2x2 subplots.

        fig_stiffness : matplotlib.figure.Figure or list (optional)
            Figure object for stiffness plots. If not provided, a new figure is created with 2x2 subplots.

        conv : str, optional
            String that controls the type of data conversion ('normal' is default).

        mode : str, optional
            Plot mode: 'total' for raw data only, or modes like 'all', 'decks', or 'velocity' depending on use case.

        orders : np.ndarray, optional
            Array of polynomial orders for each aerodynamic derivative (length 8).
        """

        # Create new figure and subplots for damping if not supplied
        if not bool(fig_damping):
            fig_damping = plt.figure()
            for k in range(4):
                fig_damping.add_subplot(2, 2, k+1)

        # Create new figure and subplots for stiffness if not supplied
        if not bool(fig_stiffness):
            fig_stiffness = plt.figure()
            for k in range(4):
                fig_stiffness.add_subplot(2, 2, k+1)

        # Boolean array indicating whether each component is a damping (True) or stiffness (False) term
        damping_ad = np.array([True, True, False, False, True, True, False, False])

        # Get axes from both figures
        axs_damping = fig_damping.get_axes()
        axs_stiffness = fig_stiffness.get_axes()

        # --- Plot Damping Derivatives ---
        self.h1.plot(mode=mode, conv=conv, ax=axs_damping[0], V=self.meanV, damping=damping_ad[0], order=orders[0])
        self.h2.plot(mode=mode, conv=conv, ax=axs_damping[1], V=self.meanV, damping=damping_ad[1], order=orders[1])
        self.a1.plot(mode=mode, conv=conv, ax=axs_damping[2], V=self.meanV, damping=damping_ad[2], order=orders[2])
        self.a2.plot(mode=mode, conv=conv, ax=axs_damping[3], V=self.meanV, damping=damping_ad[3], order=orders[3])

        # --- Plot Stiffness Derivatives ---
        self.h4.plot(mode=mode, conv=conv, ax=axs_stiffness[0], V=self.meanV, damping=damping_ad[4], order=orders[4])
        self.h3.plot(mode=mode, conv=conv, ax=axs_stiffness[1], V=self.meanV, damping=damping_ad[5], order=orders[5])
        self.a4.plot(mode=mode, conv=conv, ax=axs_stiffness[2], V=self.meanV, damping=damping_ad[6], order=orders[6])
        self.a3.plot(mode=mode, conv=conv, ax=axs_stiffness[3], V=self.meanV, damping=damping_ad[7], order=orders[7])

        # Remove x-axis labels from top 2 subplots for visual clarity
        for k in range(2):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")

        # Resize figures for better layout
        fig_damping.set_size_inches(20/2.54, 15/2.54)
        fig_stiffness.set_size_inches(20/2.54, 15/2.54)

        # Adjust layout to avoid overlap
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()


    def polyfit_to_excel(self, test_name, save_path, orders=np.ones(8, dtype=int)*2):
        """
        Fits polynomial models to aerodynamic derivatives and exports the coefficients to an Excel file.

        Parameters:
        -----------
        test_name : str
            Name of the test case, used in the Excel file title.
        save_path : str
            Directory path where the Excel file will be saved.
        orders : np.ndarray of int, optional
            Polynomial order for each aerodynamic derivative (default is second-order for all 32 terms).

        The resulting Excel file will contain:
            - Label of each aerodynamic derivative (AD)
            - Polynomial order used
            - Fitting range of reduced velocity (V̂_min, V̂_max)
            - Polynomial coefficients (a₀, a₁, ..., aₙ)
        """

        # Fit polynomials to all aerodynamic derivatives using the specified polynomial orders
        poly_coeff, v_range = self.fit_poly(orders=orders)
        
        # Labels for each aerodynamic derivative
        labels = ["H_1*", "H_2*", "H_3*", "H_4*","A_1*", "A_2*", "A_3*", "A_4*"]
        
        data = []
        for i, label in enumerate(labels):
            # Get the polynomial coefficients for the i-th derivative and reverse their order (highest degree first)
            coeff_row = list(poly_coeff[i])[::-1]
            # Get the reduced velocity fitting range for this derivative
            vmin, vmax = v_range[i]
            # Create a row with label, polynomial order, velocity range, and coefficients
            row = [label, orders[i], vmin, vmax] + coeff_row
            data.append(row)

        # Determine the maximum polynomial order used (to know how many coefficient columns to create)
        max_order = max(orders)
        # Define column names including coefficient names a0, a1, ..., an
        col_names = ["AD label", "Order", "V̂_min", "V̂_max"] + [f"a{i}" for i in range(max_order+1)]
        # Create a pandas DataFrame from the data
        df = pd.DataFrame(data, columns=col_names)

        # Define filename and full output path
        filename = f"{test_name}_AD_polyfit.xlsx"
        full_path = os.path.join(save_path, filename)

        # Write DataFrame to Excel with some metadata and formula description
        with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, startrow=2)  # Write data starting from row 3 (0-based index)
            worksheet = writer.sheets["Sheet1"]
            worksheet.cell(row=1, column=1).value = f"Test Name: {test_name}"  # Add test name in row 1
            worksheet.cell(row=1, column=4).value = "f(V̂) = a₀ + a₁·V̂ + a₂·V̂² + ..."  # Add formula description

        print(f"Saved polynomial fit data to {full_path}")


    def get_points(self):
        all_points = []

        for ad_id, ad in zip(
            ['h1', 'h2', 'h3', 'h4', 'a1', 'a2', 'a3', 'a4'],
            [self.h1, self.h2, self.h3, self.h4, self.a1, self.a2, self.a3, self.a4]
        ):
            x_array, y_array = ad.get_points()
            for x, y in zip(x_array, y_array):
                all_points.append([ad_id, x, y])

        with open('single_AD_points.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ad_id', 'x_value', 'y_value'])
            writer.writerows(all_points)


    def plot_to_compare(self, fig_damping=[], fig_stiffness=[], conv='normal', mode='poly only', orders=np.ones(8, dtype=int)*2):     
        """
        Plots aerodynamic derivatives for comparison purposes using polynomial fits only.

        This method is useful for comparing multiple datasets or polynomial fits by plotting them
        on the same axes. Each aerodynamic derivative is plotted using a custom label indicating 
        what test the data corresponds to.

        Parameters:
        -----------
        fig_damping : matplotlib.figure.Figure or list (optional)
            Figure object for damping plots. If not provided, a new 4x4 subplot figure is created.

        fig_stiffness : matplotlib.figure.Figure or list (optional)
            Figure object for stiffness plots. If not provided, a new 4x4 subplot figure is created.

        conv : str, optional
            String that controls the type of data conversion ('normal' is default).

        mode : str, optional
            Plot mode 'poly only' for comparison of polynomial fits.

        orders : np.ndarray, optional
            Array of length 8 specifying the polynomial fit order for each derivative.

        label : str, optional
            Label for the plotted data (used in legends to distinguish different datasets).
        """

        # Create new damping figure with 16 subplots if none is provided
        if not bool(fig_damping):
            fig_damping = plt.figure()
            for k in range(16):
                fig_damping.add_subplot(4, 4, k+1)

        # Create new stiffness figure with 16 subplots if none is provided
        if not bool(fig_stiffness):
            fig_stiffness = plt.figure()
            for k in range(16):
                fig_stiffness.add_subplot(4, 4, k+1)

        # Define which coefficients are damping (True), and which are stiffness (False) ---
        damping_ad = np.array([True, True, False, False,  True, True, False, False])

       # Retrieve subplot axes for damping figure
        axs_damping = fig_damping.get_axes()

        # --- Damping Plots ---
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[0], damping=damping_ad[0], order=orders[0], ADlabel="c_{z_1z_1}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[10], damping=damping_ad[0], order=orders[0], ADlabel="c_{z_2z_2}^*")

        self.h2.plot2(mode=mode, conv=conv, ax=axs_damping[1], damping=damping_ad[1], order=orders[1], ADlabel="c_{z_1\\theta_1}^*")
        self.h2.plot2(mode=mode, conv=conv, ax=axs_damping[11], damping=damping_ad[1], order=orders[1], ADlabel="c_{z_2\\theta_2}^*")

        self.a1.plot2(mode=mode, conv=conv, ax=axs_damping[4], damping=damping_ad[2], order=orders[2], ADlabel="c_{\\theta_1z_1}^*")
        self.a1.plot2(mode=mode, conv=conv, ax=axs_damping[14], damping=damping_ad[2], order=orders[2], ADlabel="c_{\\theta_2z_2}^*")

        self.a2.plot2(mode=mode, conv=conv, ax=axs_damping[5], damping=damping_ad[3], order=orders[3], ADlabel="c_{\\theta_1\\theta_1}^*")
        self.a2.plot2(mode=mode, conv=conv, ax=axs_damping[15], damping=damping_ad[3], order=orders[3], ADlabel="c_{\\theta_2\\theta_2}^*")

        # Retrieve subplot axes for stiffness figure
        axs_stiffness = fig_stiffness.get_axes()

       # --- Stiffness Plots ---
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[0], damping=damping_ad[4], order=orders[4], ADlabel="k_{z_1z_1}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[10], damping=damping_ad[4], order=orders[4], ADlabel="k_{z_2z_2}^*")

        self.h3.plot2(mode=mode, conv=conv, ax=axs_stiffness[1], damping=damping_ad[5], order=orders[5], ADlabel="k_{z_1\\theta_1}^*")
        self.h3.plot2(mode=mode, conv=conv, ax=axs_stiffness[11], damping=damping_ad[5], order=orders[5], ADlabel="k_{z_2\\theta_2}^*")

        self.a4.plot2(mode=mode, conv=conv, ax=axs_stiffness[4], damping=damping_ad[6], order=orders[6], ADlabel="k_{\\theta_1z_1}^*")
        self.a4.plot2(mode=mode, conv=conv, ax=axs_stiffness[14], damping=damping_ad[6], order=orders[6], ADlabel="k_{\\theta_2z_2}^*")

        self.a3.plot2(mode=mode, conv=conv, ax=axs_stiffness[5], damping=damping_ad[7], order=orders[7], ADlabel="k_{\\theta_1\\theta_1}^*")
        self.a3.plot2(mode=mode, conv=conv, ax=axs_stiffness[15], damping=damping_ad[7], order=orders[7], ADlabel="k_{\\theta_2\\theta_2}^*")

        # --- Plot INVALID Terms (shown as empty for visual completeness) ---
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[2], valid=False, ADlabel="c_{z_1z_2}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[3], valid=False, ADlabel="c_{z_1\\theta_2}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[6], valid=False, ADlabel="c_{\\theta_1z_2}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[7], valid=False, ADlabel="c_{\\theta_1\\theta_2}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[8], valid=False, ADlabel="c_{z_2z_1}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[9], valid=False, ADlabel="c_{z_2\\theta_1}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[12], valid=False, ADlabel="c_{\\theta_2z_1}^*")
        self.h1.plot2(mode=mode, conv=conv, ax=axs_damping[13], valid=False, ADlabel="c_{\\theta_2\\theta_1}^*")

        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[2], valid=False, ADlabel="k_{z_1z_2}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[3], valid=False, ADlabel="k_{z_1\\theta_2}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[6], valid=False, ADlabel="k_{\\theta_1z_2}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[7], valid=False, ADlabel="k_{\\theta_1\\theta_2}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[8], valid=False, ADlabel="k_{z_2z_1}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[9], valid=False, ADlabel="k_{z_2\\theta_1}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[12], valid=False, ADlabel="k_{\\theta_2z_1}^*")
        self.h4.plot2(mode=mode, conv=conv, ax=axs_stiffness[13], valid=False, ADlabel="k_{\\theta_2\\theta_1}^*")

        # Remove x-axis labels on top 12 subplots for cleaner layout
        for k in range(12):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")

        # Resize figures
        scal = 1.8
        fig_damping.set_size_inches(20/scal, 15/scal)
        fig_stiffness.set_size_inches(20/scal, 15/scal)

        # Optimize subplot layout
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()
