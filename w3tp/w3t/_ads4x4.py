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

__all__ = ["AerodynamicDerivatives4x4","AerodynamicDerivative4x4",]


class AerodynamicDerivative4x4:
    """ 
    A class used to represent a aerodynamic derivative
    
    Arguments
    ---------
    reduced_velocities  : float
        reduced velocities
    ad_load_cell_1      : float
        contribution to aerodynamic derivative from load cell 1
    ad_load_cell_2      : float
        contribution to aerodynamic derivative from load cell 2
    ad_load_cell_3      : float
        contribution to aerodynamic derivative from load cell 3
    ad_load_cell_4      : float
        contribution to aerodynamic derivative from load cell 4
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
            Indicates whether the AD corresponds to a damping term (True) or a stiffness term (False).
            (Currently used for potential future differentiation.)

        order : int, optional
            Order of the polynomial to fit.

        Returns:
        --------
        poly_coeff : np.ndarray or None
            Polynomial coefficients of the fitted curve.

        v_range : np.ndarray or None
            Min and max reduced velocity used in the fitting.
        """
        # Initialize arrays to store aerodynamic derivatives and reduced velocities
        ads = self.value
        vreds = self.reduced_velocities

        # Proceed only if data is available
        if ads.size > 0:
            poly_coeff = np.zeros(np.max(order) + 1)
            v_range = np.zeros(2)

            # Store range of reduced velocities
            v_range[0] = np.min(vreds)
            v_range[1] = np.max(vreds)

            # Fit a polynomial to the data
            poly_coeff = np.polyfit(vreds, ads, order)

            return poly_coeff, v_range
        else:
            # Return None if there’s no data to fit
            return None, None

    

    def plot(self, mode="total", conv="normal", ax=[], damping=True, order=2):
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
        """
        # Fit a polynomial to the data
        poly_coeff, v_range = self.poly_fit(damping=damping, order=order)

        # If no data is available, use placeholders
        if poly_coeff is None:
            V = 0
            y = 0
        else:
            # Evaluate polynomial over a fine velocity range
            p = np.poly1d(poly_coeff)
            V = np.linspace(v_range[0], v_range[1], 200)
            y = p(V)

        # If no axis is provided, create a new figure and axis
        if not bool(ax):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Plot total measured aerodynamic derivative
        if mode == "total":
            ax.plot(self.reduced_velocities, self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
            ax.set_ylabel(f"${self.label}$")
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.grid(True)

        # Plot both total values and polynomial fit
        elif mode == "total+poly":
            ax.plot(self.reduced_velocities, self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
            ax.plot(V, y, label="Poly Fit")
            ax.set_ylabel(f"${self.label}$")
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.grid(True)


    def plot2(self, mode="poly only", conv="normal", ax=[], damping=True, order=2, label='i'):
        """
        Plots only the polynomial fit of the aerodynamic derivative.

        Parameters:
        -----------
        mode : str, optional
            Plotting mode: "poly only" is the default and only supported option.

        conv : str, optional
            Convention flag for how data is interpreted (currently unused).

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If not provided, a new figure and axes are created.

        damping : bool, optional
            Indicates whether the AD is associated with damping (True) or stiffness (False).

        order : int, optional
            Order of the polynomial fit.

        label : str, optional
            Label to be used in the plot legend (useful for comparisons).
        """
        # Fit a polynomial to the data
        poly_coeff, v_range = self.poly_fit(damping=damping, order=order)

        if poly_coeff is None:
            V = 0
            y = 0
        else:
            # Evaluate polynomial over reduced velocity range
            p = np.poly1d(poly_coeff)
            V = np.linspace(v_range[0], v_range[1], 200)
            y = p(V)

        # Create plot if no axes were provided
        if not bool(ax):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Plot only the polynomial fit
        if mode == "poly only":
            ax.plot(V, y, label=f'{label}')
            ax.set_ylabel(f"${self.label}$")
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.legend()
            ax.grid(True)
                

class AerodynamicDerivatives4x4:
    """
    A class used to represent all aerodynamic derivatives for a 4 dof motion
    
    parameters:
    ----------
    c_z1z1 ... c_theta2theta2 : obj
        aerodynamic derivatives related to the self-excited force asociated with damping
    k_z1z1 ... k_theta2theta2 : obj
        aerodynamic derivatives related to the self-excited force asociated with stiffness
    
    methods:
    -------
    .fromWTT()
        obtains aerodynamic derivatives from a sequence of single harmonic wind tunnel tests
    .append()
        appends an instance of the class AerodynamicDerivtives to self    
    .plot()
        plots all aerodynamic derivatives    
    
    
    
    """
    def __init__(self, 
                 c_z1z1=None, c_z1theta1=None, c_z1z2=None, c_z1theta2=None,
                 c_theta1z1=None, c_theta1theta1=None, c_theta1z2=None, c_theta1theta2=None,
                 c_z2z1=None, c_z2theta1=None, c_z2z2=None, c_z2theta2=None,
                 c_theta2z1=None, c_theta2theta1=None, c_theta2z2=None, c_theta2theta2=None,
                 k_z1z1=None, k_z1theta1=None, k_z1z2=None, k_z1theta2=None,
                 k_theta1z1=None, k_theta1theta1=None, k_theta1z2=None, k_theta1theta2=None,
                 k_z2z1=None, k_z2theta1=None, k_z2z2=None, k_z2theta2=None,
                 k_theta2z1=None, k_theta2theta1=None, k_theta2z2=None, k_theta2theta2=None):

        """
        parameters:
        ----------
        c_z1z1 ... c_theta2theta2 : obj
            aerodynamic derivatives related to the self-excited force asociated with damping
        k_z1z1 ... k_theta2theta2 : obj
            aerodynamic derivatives related to the self-excited force asociated with stiffness
        ---------
        """
        
        self.c_z1z1 = c_z1z1 or AerodynamicDerivative4x4(label="c_{z_1z_1}^*")
        self.c_z1theta1 = c_z1theta1 or AerodynamicDerivative4x4(label="c_{z_1\\theta_1}^*")
        self.c_z1z2 = c_z1z2 or AerodynamicDerivative4x4(label="c_{z_1z_2}^*")
        self.c_z1theta2 = c_z1theta2 or AerodynamicDerivative4x4(label="c_{z_1\\theta_2}^*")

        self.c_theta1z1 = c_theta1z1 or AerodynamicDerivative4x4(label="c_{\\theta_1z_1}^*")
        self.c_theta1theta1 = c_theta1theta1 or AerodynamicDerivative4x4(label="c_{\\theta_1\\theta_1}^*")
        self.c_theta1z2 = c_theta1z2 or AerodynamicDerivative4x4(label="c_{\\theta_1z_2}^*")
        self.c_theta1theta2 = c_theta1theta2 or AerodynamicDerivative4x4(label="c_{\\theta_1\\theta_2}^*")

        self.c_z2z1 = c_z2z1 or AerodynamicDerivative4x4(label="c_{z_2z_1}^*")
        self.c_z2theta1 = c_z2theta1 or AerodynamicDerivative4x4(label="c_{z_2\\theta_1}^*")
        self.c_z2z2 = c_z2z2 or AerodynamicDerivative4x4(label="c_{z_2z_2}^*")
        self.c_z2theta2 = c_z2theta2 or AerodynamicDerivative4x4(label="c_{z_2\\theta_2}^*")

        self.c_theta2z1 = c_theta2z1 or AerodynamicDerivative4x4(label="c_{\\theta_2z_1}^*")
        self.c_theta2theta1 = c_theta2theta1 or AerodynamicDerivative4x4(label="c_{\\theta_2\\theta_1}^*")
        self.c_theta2z2 = c_theta2z2 or AerodynamicDerivative4x4(label="c_{\\theta_2z_2}^*")
        self.c_theta2theta2 = c_theta2theta2 or AerodynamicDerivative4x4(label="c_{\\theta_2\\theta_2}^*")

        self.k_z1z1 = k_z1z1 or AerodynamicDerivative4x4(label="k_{z_1z_1}^*")
        self.k_z1theta1 = k_z1theta1 or AerodynamicDerivative4x4(label="k_{z_1\\theta_1}^*")
        self.k_z1z2 = k_z1z2 or AerodynamicDerivative4x4(label="k_{z_1z_2}^*")
        self.k_z1theta2 = k_z1theta2 or AerodynamicDerivative4x4(label="k_{z_1\\theta_2}^*")

        self.k_theta1z1 = k_theta1z1 or AerodynamicDerivative4x4(label="k_{\\theta_1z_1}^*")
        self.k_theta1theta1 = k_theta1theta1 or AerodynamicDerivative4x4(label="k_{\\theta_1\\theta_1}^*")
        self.k_theta1z2 = k_theta1z2 or AerodynamicDerivative4x4(label="k_{\\theta_1z_2}^*")
        self.k_theta1theta2 = k_theta1theta2 or AerodynamicDerivative4x4(label="k_{\\theta_1\\theta_2}^*")

        self.k_z2z1 = k_z2z1 or AerodynamicDerivative4x4(label="k_{z_2z_1}^*")
        self.k_z2theta1 = k_z2theta1 or AerodynamicDerivative4x4(label="k_{z_2\\theta_1}^*")
        self.k_z2z2 = k_z2z2 or AerodynamicDerivative4x4(label="k_{z_2z_2}^*")
        self.k_z2theta2 = k_z2theta2 or AerodynamicDerivative4x4(label="k_{z_2\\theta_2}^*")

        self.k_theta2z1 = k_theta2z1 or AerodynamicDerivative4x4(label="k_{\\theta_2z_1}^*")
        self.k_theta2theta1 = k_theta2theta1 or AerodynamicDerivative4x4(label="k_{\\theta_2\\theta_1}^*")
        self.k_theta2z2 = k_theta2z2 or AerodynamicDerivative4x4(label="k_{\\theta_2z_2}^*")
        self.k_theta2theta2 = k_theta2theta2 or AerodynamicDerivative4x4(label="k_{\\theta_2\\theta_2}^*")


    @classmethod
    def fromWTT(cls, experiment_in_still_air, experiment_in_wind, section_width, section_length_1, section_length_2, upstream_in_rig=True, filter_order=6, cutoff_frequency=7):
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
        section_length_1 : float
            Length of the upstream section model.
        section_length_2 : float
            Length of the downstream section model.
        upstream_in_rig : bool, optional
            Indicates whether the upstream section is placed in the rig (default is True).
        filter_order : int, optional
            Order of the Butterworth filter used for signal processing (default is 6).
        cutoff_frequency : float, optional
            Cutoff frequency of the Butterworth filter (default is 7 Hz).

        Returns:
        --------
        tuple:
            - An instance of AerodynamicDerivatives4x4 containing identified coefficients.
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

            # Design Butterworth filter
            sos = spsp.butter(filter_order, cutoff_frequency, fs=sampling_frequency, output="sos")

            # Motions of the section in the rig: [r_x1, r_z1, r_theta1]
            motions_bridge_in_rig = experiment_in_wind_still_air_forces_removed.motion

            # Motions of the section on the wall: [r_x2, r_z2, r_theta2] (assumed zero)
            motions_bridge_on_wall = np.zeros_like(motions_bridge_in_rig)

            # Full motion vector: [r_x1, r_z1, r_theta1, r_x2, r_z2, r_theta2]
            motions = np.hstack((motions_bridge_in_rig, motions_bridge_on_wall))

            # Filter the motions
            motions = spsp.sosfiltfilt(sos, motions, axis=0)

            # Compute time derivatives of the motions
            time_derivative_motions = np.vstack((np.array([0, 0, 0, 0, 0, 0]), np.diff(motions, axis=0))) * sampling_frequency

            # Identify motion type (0 = horizontal, 1 = vertical, 2 = torsional)
            motion_type = experiment_in_wind_still_air_forces_removed.motion_type()

            # Perform FFT to extract motion frequency
            fourier_amplitudes = np.fft.fft(motions[starts[k]:stops[k], motion_type] - np.mean(motions[starts[k]:stops[k], motion_type]))
            time_step = experiment_in_wind_still_air_forces_removed.time[1] - experiment_in_wind_still_air_forces_removed.time[0]
            frequencies = np.fft.fftfreq(len(fourier_amplitudes), time_step)

            # Find dominant motion frequency
            peak_index = np.argmax(np.abs(fourier_amplitudes[0:int(len(fourier_amplitudes)/2)]))
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
                section_length = section_length_1 if m in [0, 1] else section_length_2

                # Extract forces for current load cell
                forces = experiment_in_wind_still_air_forces_removed.forces_global_center[starts[k]:stops[k], selected_forces + 6 * m]

                # Remove mean wind force component
                forces_mean_wind_removed = forces - np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400, selected_forces + 6 * m], axis=0)

                # Compute raw coefficients: E = X^+ * q
                coefficient_matrix = pseudo_inverse_regressor_matrix @ forces_mean_wind_removed

                # Normalize coefficients to obtain c* and k*
                normalized_coefficient_matrix[:, :, k, m] = coefficient_matrix
                normalized_coefficient_matrix[0, :, k, m] *= 2 / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed / reduced_frequency / section_width / section_length
                normalized_coefficient_matrix[1, :, k, m] *= 2 / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed**2 / reduced_frequency**2 / section_length
                normalized_coefficient_matrix[:, 2, k, m] /= section_width

                if motion_type == 2:
                    # Special case: normalize all terms by section width for torsional tests
                    normalized_coefficient_matrix[:, :, k, m] /= section_width

                # Compute predicted forces from the model and add wind mean
                forces_predicted_by_ads[starts[k]:stops[k], selected_forces + 6 * m] += regressor_matrix @ coefficient_matrix + np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400, selected_forces + 6 * m], axis=0)

        # Construct Experiment object for model prediction based on fitted aerodynamic model
        obj1 = experiment_in_wind_still_air_forces_removed
        obj2 = experiment_in_still_air
        model_prediction = Experiment(obj1.name, obj1.time, obj1.temperature, obj1.air_density, obj1.wind_speed,[],forces_predicted_by_ads,obj2.motion)
                 
        # Instantiate empty derivative objects
        c_z1z1 = AerodynamicDerivative4x4()
        c_z1theta1 = AerodynamicDerivative4x4()
        c_z1z2 = AerodynamicDerivative4x4()
        c_z1theta2 = AerodynamicDerivative4x4()
        
        c_theta1z1 = AerodynamicDerivative4x4()
        c_theta1theta1 = AerodynamicDerivative4x4()
        c_theta1z2 = AerodynamicDerivative4x4()
        c_theta1theta2 = AerodynamicDerivative4x4()
        
        c_z2z1 = AerodynamicDerivative4x4()
        c_z2theta1 = AerodynamicDerivative4x4()
        c_z2z2 = AerodynamicDerivative4x4()
        c_z2theta2 = AerodynamicDerivative4x4()
        
        c_theta2z1 = AerodynamicDerivative4x4()
        c_theta2theta1 = AerodynamicDerivative4x4()
        c_theta2z2 = AerodynamicDerivative4x4()
        c_theta2theta2 = AerodynamicDerivative4x4()
        
        k_z1z1 = AerodynamicDerivative4x4()
        k_z1theta1 = AerodynamicDerivative4x4()
        k_z1z2 = AerodynamicDerivative4x4()
        k_z1theta2 = AerodynamicDerivative4x4()
        
        k_theta1z1 = AerodynamicDerivative4x4()
        k_theta1theta1 = AerodynamicDerivative4x4()
        k_theta1z2 = AerodynamicDerivative4x4()
        k_theta1theta2 = AerodynamicDerivative4x4()
        
        k_z2z1 = AerodynamicDerivative4x4()
        k_z2theta1 = AerodynamicDerivative4x4()
        k_z2z2 = AerodynamicDerivative4x4()
        k_z2theta2 = AerodynamicDerivative4x4()
        
        k_theta2z1 = AerodynamicDerivative4x4()
        k_theta2theta1 = AerodynamicDerivative4x4()
        k_theta2z2 = AerodynamicDerivative4x4()
        k_theta2theta2 = AerodynamicDerivative4x4()
          

        if upstream_in_rig == True:    
            if motion_type == 0:
                mat = 0
            elif motion_type == 1:
                mat = 0 # C(damping)
                col = 1
                c_z1z1 = AerodynamicDerivative4x4("c_{z_1z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                c_z2z1 = AerodynamicDerivative4x4("c_{z_2z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                col = 2
                c_theta1z1 = AerodynamicDerivative4x4("c_{\\theta_1z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                c_theta2z1 = AerodynamicDerivative4x4("c_{\\theta_2z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)

                mat = 1 # K(stiffness)
                col = 1
                k_z1z1 = AerodynamicDerivative4x4("k_{z_1z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                k_z2z1 = AerodynamicDerivative4x4("k_{z_2z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                col = 2
                k_theta1z1 = AerodynamicDerivative4x4("k_{\\theta_1z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                k_theta2z1 = AerodynamicDerivative4x4("k_{\\theta_2z_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)

            elif motion_type == 2:
                mat = 0 # C(damping)
                col = 1
                c_z1theta1 = AerodynamicDerivative4x4("c_{z_1\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                c_z2theta1 = AerodynamicDerivative4x4("c_{z_2\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                col = 2
                c_theta1theta1 = AerodynamicDerivative4x4("c_{\\theta_1\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                c_theta2theta1 = AerodynamicDerivative4x4("c_{\\theta_2\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)

                mat = 1 # K(stiffness)
                col = 1
                k_z1theta1 = AerodynamicDerivative4x4("k_{z_1\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                k_z2theta1 = AerodynamicDerivative4x4("k_{z_2\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                col = 2
                k_theta1theta1 = AerodynamicDerivative4x4("k_{\\theta_1\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                k_theta2theta1 = AerodynamicDerivative4x4("k_{\\theta_2\\theta_1}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                        
        elif upstream_in_rig == False:
            if motion_type == 0:
                mat = 0
            elif motion_type == 1:
                mat = 0 #C(damping)
                col = 1
                c_z1z2 = AerodynamicDerivative4x4("c_{z_1z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                c_z2z2 = AerodynamicDerivative4x4("c_{z_2z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                col = 2
                c_theta1z2 = AerodynamicDerivative4x4("c_{\\theta_1z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                c_theta2z2 = AerodynamicDerivative4x4("c_{\\theta_2z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)

                mat = 1 #K(stiffness)
                col = 1
                k_z1z2 = AerodynamicDerivative4x4("k_{z_1z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                k_z2z2 = AerodynamicDerivative4x4("k_{z_2z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                col = 2
                k_theta1z2 = AerodynamicDerivative4x4("k_{\\theta_1z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                k_theta2z2 = AerodynamicDerivative4x4("k_{\\theta_2z_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)

            elif motion_type == 2:
                mat = 0 #C(damping)
                col = 1
                c_z1theta2 = AerodynamicDerivative4x4("c_{z_1\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                c_z2theta2 = AerodynamicDerivative4x4("c_{z_2\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                col = 2
                c_theta1theta2 = AerodynamicDerivative4x4("c_{\\theta_1\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                c_theta2theta2 = AerodynamicDerivative4x4("c_{\\theta_2\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)

                mat = 1 #K(stiffness)
                col = 1
                k_z1theta2 = AerodynamicDerivative4x4("k_{z_1\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                k_z2theta2 = AerodynamicDerivative4x4("k_{z_2\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)
                col = 2
                k_theta1theta2 = AerodynamicDerivative4x4("k_{\\theta_1\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 2], normalized_coefficient_matrix[mat, col, :, 3], mean_wind_speeds, frequencies_of_motion)
                k_theta2theta2 = AerodynamicDerivative4x4("k_{\\theta_2\\theta_2}^*", reduced_velocities, normalized_coefficient_matrix[mat, col, :, 0], normalized_coefficient_matrix[mat, col, :, 1], mean_wind_speeds, frequencies_of_motion)   

        return cls(c_z1z1, c_z1theta1, c_z1z2, c_z1theta2, c_theta1z1, c_theta1theta1, c_theta1z2, c_theta1theta2, c_z2z1, c_z2theta1, c_z2z2, c_z2theta2, c_theta2z1, c_theta2theta1, c_theta2z2, c_theta2theta2, k_z1z1, k_z1theta1, k_z1z2, k_z1theta2, k_theta1z1, k_theta1theta1, k_theta1z2, k_theta1theta2, k_z2z1, k_z2theta1, k_z2z2, k_z2theta2, k_theta2z1, k_theta2theta1, k_theta2z2, k_theta2theta2), model_prediction, experiment_in_wind_still_air_forces_removed
    
    @classmethod
    def from_poly_k(cls,poly_k, k_range, vred):
        vred[vred==0] = 1.0e-10
        uit_step = lambda k,kc: 1./(1 + np.exp(-2*20*(k-kc)))
        fit = lambda p,k,k1c,k2c : np.polyval(p,k)*uit_step(k,k1c)*(1-uit_step(k,k2c)) + np.polyval(p,k1c)*(1-uit_step(k,k1c)) + np.polyval(p,k2c)*(uit_step(k,k2c))
        
        damping_ad = np.array([True, True, True, True, True, True, True, True,
                               True, True, True, True, True, True, True, True,
                               False, False, False, False, False, False, False, False,
                               False, False, False, False, False, False, False, False])
        
        labels = ["c_{z_1z_1}^*", "c_{z_1\\theta_1}^*", "c_{z_1z_2}^*", "c_{z_1\\theta_2}^*",
                  "c_{\\theta_1z_1}^*", "c_{\\theta_1\\theta_1}^*", "c_{\\theta_1z_2}^*", "c_{\\theta_1\\theta_2}^*",
                  "c_{z_2z_1}^*", "c_{z_2\\theta_1}^*", "c_{z_2z_2}^*", "c_{z_2\\theta_2}^*",
                  "c_{\\theta_2z_1}^*", "c_{\\theta_2\\theta_1}^*", "c_{\\theta_2z_2}^*", "c_{\\theta_2\\theta_2}^*",
                  "k_{z_1z_1}^*", "k_{z_1\\theta_1}^*", "k_{z_1z_2}^*", "k_{z_1\\theta_2}^*",
                  "k_{\\theta_1z_1}^*", "k_{\\theta_1\\theta_1}^*", "k_{\\theta_1z_2}^*", "k_{\\theta_1\\theta_2}^*",
                  "k_{z_2z_1}^*", "k_{z_2\\theta_1}^*", "k_{z_2z_2}^*", "k_{z_2\\theta_2}^*",
                  "k_{\\theta_2z_1}^*", "k_{\\theta_2\\theta_1}^*", "k_{\\theta_2z_2}^*", "k_{\\theta_2\\theta_2}^*"]

        ads = []
        for k in range(32):
                      
            if damping_ad[k] == True:
                ad_value = np.abs(vred)*fit(poly_k[k,:],np.abs(1/vred),k_range[k,0],k_range[k,1])
            else:
                ad_value = np.abs(vred)**2*fit(poly_k[k,:],np.abs(1/vred),k_range[k,0],k_range[k,1])
                
            ads.append(AerodynamicDerivative4x4(labels[k],vred,ad_value/2 , ad_value/2 , vred*0, vred*0))
            
             
        return cls(ads[0],ads[1],ads[2],ads[3],ads[4],ads[5],ads[6],ads[7],ads[8],ads[9],ads[10],ads[11],ads[12],ads[13],ads[14],ads[15],ads[16],ads[17],ads[18],ads[19],ads[20],ads[21],ads[22],ads[23],ads[24],ads[25],ads[26],ads[27],ads[28],ads[29],ads[30],ads[31])
    
    def append(self,ads):
        """
        Appends the aerodynamic data from another AerodynamicDerivatives4x4 instance to the current instance.

        This method combines the time-series data and test parameters (load cell measurements, frequencies,
        mean wind speeds, and reduced velocities) from a second `AerodynamicDerivatives2x2` object into the 
        corresponding attributes of the current object. The data from each of the 32 aerodynamic derivatives 
        (16 damping terms and 16 stiffness terms) is concatenated element-wise.

        Parameters:
        -----------
        ads : AerodynamicDerivatives4x4
            An instance of the AerodynamicDerivatives2x2 class whose data will be appended to this instance.
        """

        objs1 = [self.c_z1z1, self.c_z1theta1, self.c_z1z2, self.c_z1theta2,
                 self.c_theta1z1, self.c_theta1theta1, self.c_theta1z2, self.c_theta1theta2,
                 self.c_z2z1, self.c_z2theta1, self.c_z2z2, self.c_z2theta2,
                 self.c_theta2z1, self.c_theta2theta1, self.c_theta2z2, self.c_theta2theta2,
                 self.k_z1z1, self.k_z1theta1, self.k_z1z2, self.k_z1theta2,
                 self.k_theta1z1, self.k_theta1theta1, self.k_theta1z2, self.k_theta1theta2,
                 self.k_z2z1, self.k_z2theta1, self.k_z2z2, self.k_z2theta2,
                 self.k_theta2z1, self.k_theta2theta1, self.k_theta2z2, self.k_theta2theta2]

        objs2 = [ads.c_z1z1, ads.c_z1theta1, ads.c_z1z2, ads.c_z1theta2,
                 ads.c_theta1z1, ads.c_theta1theta1, ads.c_theta1z2, ads.c_theta1theta2,
                 ads.c_z2z1, ads.c_z2theta1, ads.c_z2z2, ads.c_z2theta2,
                 ads.c_theta2z1, ads.c_theta2theta1, ads.c_theta2z2, ads.c_theta2theta2,
                 ads.k_z1z1, ads.k_z1theta1, ads.k_z1z2, ads.k_z1theta2,
                 ads.k_theta1z1, ads.k_theta1theta1, ads.k_theta1z2, ads.k_theta1theta2,
                 ads.k_z2z1, ads.k_z2theta1, ads.k_z2z2, ads.k_z2theta2,
                 ads.k_theta2z1, ads.k_theta2theta1, ads.k_theta2z2, ads.k_theta2theta2]

        for k in range(len(objs1)):
            objs1[k].ad_load_cell_a = np.append(objs1[k].ad_load_cell_a,objs2[k].ad_load_cell_a)
            objs1[k].ad_load_cell_b = np.append(objs1[k].ad_load_cell_b,objs2[k].ad_load_cell_b) 
            
            objs1[k].frequencies = np.append(objs1[k].frequencies,objs2[k].frequencies) 
            objs1[k].mean_wind_speeds = np.append(objs1[k].mean_wind_speeds,objs2[k].mean_wind_speeds) 
            objs1[k].reduced_velocities = np.append(objs1[k].reduced_velocities,objs2[k].reduced_velocities) 
            
    @property
    def ad_matrix(self):
        """
        Returns:
        --------
        ads : np.ndarray
            A matrix of aerodynamic derivative values of shape (32, N), where each row corresponds
            to one of the 32 aerodynamic derivatives and each column to a test point (reduced velocity).

        vreds : np.ndarray
            A matrix of reduced velocities with shape (32, N), corresponding to each aerodynamic derivative.
        """

        # Number of data points (assumed the same for all derivatives)
        N = self.c_z1z1.reduced_velocities.shape[0]

        # Initialize matrices for aerodynamic derivatives and reduced velocities
        ads = np.zeros((32, N))
        vreds = np.zeros((32, N))

        # Fill damping aerodynamic derivatives (0–15)
        ads[0, :] = self.c_z1z1.value
        ads[1, :] = self.c_z1theta1.value
        ads[2, :] = self.c_z1z2.value
        ads[3, :] = self.c_z1theta2.value
        ads[4, :] = self.c_theta1z1.value
        ads[5, :] = self.c_theta1theta1.value
        ads[6, :] = self.c_theta1z2.value
        ads[7, :] = self.c_theta1theta2.value
        ads[8, :] = self.c_z2z1.value
        ads[9, :] = self.c_z2theta1.value
        ads[10, :] = self.c_z2z2.value
        ads[11, :] = self.c_z2theta2.value
        ads[12, :] = self.c_theta2z1.value
        ads[13, :] = self.c_theta2theta1.value
        ads[14, :] = self.c_theta2z2.value
        ads[15, :] = self.c_theta2theta2.value

        # Fill stiffness aerodynamic derivatives (16–31)
        ads[16, :] = self.k_z1z1.value
        ads[17, :] = self.k_z1theta1.value
        ads[18, :] = self.k_z1z2.value
        ads[19, :] = self.k_z1theta2.value
        ads[20, :] = self.k_theta1z1.value
        ads[21, :] = self.k_theta1theta1.value
        ads[22, :] = self.k_theta1z2.value
        ads[23, :] = self.k_theta1theta2.value
        ads[24, :] = self.k_z2z1.value
        ads[25, :] = self.k_z2theta1.value
        ads[26, :] = self.k_z2z2.value
        ads[27, :] = self.k_z2theta2.value
        ads[28, :] = self.k_theta2z1.value
        ads[29, :] = self.k_theta2theta1.value
        ads[30, :] = self.k_theta2z2.value
        ads[31, :] = self.k_theta2theta2.value

        # Fill reduced velocities for damping terms
        vreds[0, :] = self.c_z1z1.reduced_velocities
        vreds[1, :] = self.c_z1theta1.reduced_velocities
        vreds[2, :] = self.c_z1z2.reduced_velocities
        vreds[3, :] = self.c_z1theta2.reduced_velocities
        vreds[4, :] = self.c_theta1z1.reduced_velocities
        vreds[5, :] = self.c_theta1theta1.reduced_velocities
        vreds[6, :] = self.c_theta1z2.reduced_velocities
        vreds[7, :] = self.c_theta1theta2.reduced_velocities
        vreds[8, :] = self.c_z2z1.reduced_velocities
        vreds[9, :] = self.c_z2theta1.reduced_velocities
        vreds[10, :] = self.c_z2z2.reduced_velocities
        vreds[11, :] = self.c_z2theta2.reduced_velocities
        vreds[12, :] = self.c_theta2z1.reduced_velocities
        vreds[13, :] = self.c_theta2theta1.reduced_velocities
        vreds[14, :] = self.c_theta2z2.reduced_velocities
        vreds[15, :] = self.c_theta2theta2.reduced_velocities

        # Fill reduced velocities for stiffness terms
        vreds[16, :] = self.k_z1z1.reduced_velocities
        vreds[17, :] = self.k_z1theta1.reduced_velocities
        vreds[18, :] = self.k_z1z2.reduced_velocities
        vreds[19, :] = self.k_z1theta2.reduced_velocities
        vreds[20, :] = self.k_theta1z1.reduced_velocities
        vreds[21, :] = self.k_theta1theta1.reduced_velocities
        vreds[22, :] = self.k_theta1z2.reduced_velocities
        vreds[23, :] = self.k_theta1theta2.reduced_velocities
        vreds[24, :] = self.k_z2z1.reduced_velocities
        vreds[25, :] = self.k_z2theta1.reduced_velocities
        vreds[26, :] = self.k_z2z2.reduced_velocities
        vreds[27, :] = self.k_z2theta2.reduced_velocities
        vreds[28, :] = self.k_theta2z1.reduced_velocities
        vreds[29, :] = self.k_theta2theta1.reduced_velocities
        vreds[30, :] = self.k_theta2z2.reduced_velocities
        vreds[31, :] = self.k_theta2theta2.reduced_velocities

        return ads, vreds
    
    def ad_matrix_jagged(self):
        """
        Constructs jagged arrays of aerodynamic derivatives and corresponding reduced velocities, accommodating derivatives with varying lengths.

        This property iterates through a predefined list of aerodynamic derivative objects, extracting their values and reduced velocities 
        into two separate lists. Each entry in these lists can vary in length, effectively creating jagged arrays to handle non-uniform data.

        Returns:
        --------
        ads : List[np.ndarray]
            A list containing 32 arrays of aerodynamic derivative values. Each array can have a different length, corresponding to each aerodynamic derivative.

        vreds : List[np.ndarray]
            A list containing 32 arrays of reduced velocities, corresponding to each aerodynamic derivative.
        """

        # Initialize lists for aerodynamic derivatives and reduced velocities
        ads = []
        vreds = []

        # List of aerodynamic derivative objects
        derivatives = [
            self.c_z1z1, self.c_z1theta1, self.c_z1z2, self.c_z1theta2,
            self.c_theta1z1, self.c_theta1theta1, self.c_theta1z2, self.c_theta1theta2,
            self.c_z2z1, self.c_z2theta1, self.c_z2z2, self.c_z2theta2,
            self.c_theta2z1, self.c_theta2theta1, self.c_theta2z2, self.c_theta2theta2,
            self.k_z1z1, self.k_z1theta1, self.k_z1z2, self.k_z1theta2,
            self.k_theta1z1, self.k_theta1theta1, self.k_theta1z2, self.k_theta1theta2,
            self.k_z2z1, self.k_z2theta1, self.k_z2z2, self.k_z2theta2,
            self.k_theta2z1, self.k_theta2theta1, self.k_theta2z2, self.k_theta2theta2
        ]

        # Populate the jagged arrays
        for derivative in derivatives:
            ads.append(np.array(derivative.value))
            vreds.append(np.array(derivative.reduced_velocities))

        return ads, vreds


    #Ikke benyttet
    def fit_poly_k(self,orders = np.ones(32,dtype=int)*2):
        ad_matrix, vreds = self.ad_matrix
        
        poly_coeff = np.zeros((32,np.max(orders)+1))
        k_range = np.zeros((32,2))
        
        damping_ad = np.array([True, True, True, True, True, True, True, True,
                               True, True, True, True, True, True, True, True,
                               False, False, False, False, False, False, False, False,
                               False, False, False, False, False, False, False, False])
        
        
        for k in range(32):
            k_range[k,0] = 1/np.max(vreds)
            k_range[k,1] = 1/np.min(vreds)
            
            if damping_ad[k] == True:
                poly_coeff[k,-orders[k]-1:] = np.polyfit(1/vreds[k,:],1/vreds[k,:]*ad_matrix[k,:],orders[k])
            elif damping_ad[k] == False:
                poly_coeff[k,-orders[k]-1:] = np.polyfit(1/vreds[k,:],(1/vreds[k,:])**2*ad_matrix[k,:],orders[k])
            
                
        return poly_coeff, k_range
    

    def fit_poly(self, orders=np.ones(32, dtype=int)*2):
        """
        Fits polynomial curves to each of the 32 aerodynamic derivative components in the ad_matrix
        as a function of reduced velocity. The polynomial order for each component can be specified.

        Parameters:
            orders (np.ndarray): Array of length 32 specifying the polynomial order for each component.
                                Defaults to second-order (2) for all components.

        Returns:
            poly_coeff (List[np.ndarray]): List containing arrays of polynomial coefficients for each component.
            v_range (np.ndarray): Min and max reduced velocity values used for fitting, for each component.
        """

        ad_matrix, vreds = self.ad_matrix_jagged()  # jagged arrays (lists of np.ndarray)

        poly_coeff = np.zeros((32, np.max(orders) + 1))                     # list to store polynomial coefficients for each component
        v_range = np.zeros((32, 2))         # Initialize velocity range array

        for k in range(32):
            v_k = vreds[k]        # Reduced velocities for current component
            ad_k = ad_matrix[k]   # Aerodynamic derivative data for current component

            # Store the min and max reduced velocity for each component
            v_range[k, 0] = np.min(v_k)
            v_range[k, 1] = np.max(v_k)

            # Fit polynomial of specified order to current component
            poly_coeff[k, :] = np.polyfit(v_k, ad_k, orders[k])

        return poly_coeff, v_range


    #Ikke endret og ikke benyttet
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
        
        ad_value = pd.DataFrame({"P_1": self.p1.value,
                                 "P_2": self.p2.value,
                                 "P_3": self.p3.value,
                                 "P_4": self.p4.value,
                                 "P_5": self.p5.value,
                                 "P_6": self.p6.value,

                                 "H_1": self.h1.value,
                                 "H_2": self.h2.value,
                                 "H_3": self.h3.value,
                                 "H_4": self.h4.value,
                                 "H_5": self.h5.value,
                                 "H_6": self.h6.value,

                                 "A_1": self.a1.value,
                                 "A_2": self.a2.value,
                                 "A_3": self.a3.value,
                                 "A_4": self.a4.value,
                                 "A_5": self.a5.value,
                                 "A_6": self.a6.value,
                                 })

        ad_reduced_velocity = pd.DataFrame({"P_1": self.p1.reduced_velocities,
                                            "P_2": self.p2.reduced_velocities,
                                            "P_3": self.p3.reduced_velocities,
                                            "P_4": self.p4.reduced_velocities,
                                            "P_5": self.p5.reduced_velocities,
                                            "P_6": self.p6.reduced_velocities,

                                            "H_1": self.h1.reduced_velocities,
                                            "H_2": self.h2.reduced_velocities,
                                            "H_3": self.h3.reduced_velocities,
                                            "H_4": self.h4.reduced_velocities,
                                            "H_5": self.h5.reduced_velocities,
                                            "H_6": self.h6.reduced_velocities,

                                            "A_1": self.a1.reduced_velocities,
                                            "A_2": self.a2.reduced_velocities,
                                            "A_3": self.a3.reduced_velocities,
                                            "A_4": self.a4.reduced_velocities,
                                            "A_5": self.a5.reduced_velocities,
                                            "A_6": self.a6.reduced_velocities,
                                            })

        ad_mean_wind_speeds = pd.DataFrame({"P_1": self.p1.mean_wind_speeds,
                                            "P_2": self.p2.mean_wind_speeds,
                                            "P_3": self.p3.mean_wind_speeds,
                                            "P_4": self.p4.mean_wind_speeds,
                                            "P_5": self.p5.mean_wind_speeds,
                                            "P_6": self.p6.mean_wind_speeds,

                                            "H_1": self.h1.mean_wind_speeds,
                                            "H_2": self.h2.mean_wind_speeds,
                                            "H_3": self.h3.mean_wind_speeds,
                                            "H_4": self.h4.mean_wind_speeds,
                                            "H_5": self.h5.mean_wind_speeds,
                                            "H_6": self.h6.mean_wind_speeds,

                                            "A_1": self.a1.mean_wind_speeds,
                                            "A_2": self.a2.mean_wind_speeds,
                                            "A_3": self.a3.mean_wind_speeds,
                                            "A_4": self.a4.mean_wind_speeds,
                                            "A_5": self.a5.mean_wind_speeds,
                                            "A_6": self.a6.mean_wind_speeds,
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


    def polyfit_to_excel(self, test_name, save_path, orders=np.ones(32, dtype=int)*2):
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

        # Labels for the aerodynamic damping (c_*) and stiffness (k_*) coefficients
        labels = ["c_z1z1*", "c_z1θ1*", "c_z1z2*", "c_z1θ2*",
                "c_θ1z1*", "c_θ1θ1*", "c_θ1z2*", "c_θ1θ2*",
                "c_z2z1*", "c_z2θ1*", "c_z2z2*", "c_z2θ2*",
                "c_θ2z1*", "c_θ2θ1*", "c_θ2z2*", "c_θ2θ2*",
                "k_z1z1*", "k_z1θ1*", "k_z1z2*", "k_z1θ2*",
                "k_θ1z1*", "k_θ1θ1*", "k_θ1z2*", "k_θ1θ2*",
                "k_z2z1*", "k_z2θ1*", "k_z2z2*", "k_z2θ2*",
                "k_θ2z1*", "k_θ2θ1*", "k_θ2z2*", "k_θ2θ2*"]

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
        col_names = ["AD label", "Order", "V̂_min", "V̂_max"] + [f"a{i}" for i in range(max_order + 1)]
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
        

    def plot(self, fig_damping=[], fig_stiffness=[], conv='normal', mode='total', orders=np.ones(32, dtype=int)*2):
        """
        Plots all 32 aerodynamic derivatives (16 damping + 16 stiffness) with optional polynomial fits.

        Parameters:
        -----------
        fig_damping : matplotlib.figure.Figure or list (optional)
            Figure object for damping plots. If not provided, a new figure is created with 4x4 subplots.

        fig_stiffness : matplotlib.figure.Figure or list (optional)
            Figure object for stiffness plots. If not provided, a new figure is created with 4x4 subplots.

        conv : str, optional
            String that controls the type of data conversion ('normal' is default).

        mode : str, optional
            Plot mode: 'total' for raw data only or 'total+poly' to also include polynomial fits.

        orders : np.ndarray, optional
            Array of polynomial orders for each aerodynamic derivative (length 32).
        """

        # Create new figure and subplots for damping if not supplied
        if not bool(fig_damping):
            fig_damping = plt.figure()
            for k in range(16):
                fig_damping.add_subplot(4, 4, k+1)

        # Create new figure and subplots for stiffness if not supplied
        if not bool(fig_stiffness):
            fig_stiffness = plt.figure()
            for k in range(16):
                fig_stiffness.add_subplot(4, 4, k+1)

        # Boolean array indicating whether each component is a damping (True) or stiffness (False) term
        damping_ad = np.array([True]*16 + [False]*16)

        # Get axes from both figures
        axs_damping = fig_damping.get_axes()
        axs_stiffness = fig_stiffness.get_axes()

        # --- Plot Damping Derivatives ---
        self.c_z1z1.plot(mode=mode, conv=conv, ax=axs_damping[0], damping=damping_ad[0], order=orders[0])
        self.c_z1theta1.plot(mode=mode, conv=conv, ax=axs_damping[1], damping=damping_ad[1], order=orders[1])
        self.c_z1z2.plot(mode=mode, conv=conv, ax=axs_damping[2], damping=damping_ad[2], order=orders[2])
        self.c_z1theta2.plot(mode=mode, conv=conv, ax=axs_damping[3], damping=damping_ad[3], order=orders[3])

        self.c_theta1z1.plot(mode=mode, conv=conv, ax=axs_damping[4], damping=damping_ad[4], order=orders[4])
        self.c_theta1theta1.plot(mode=mode, conv=conv, ax=axs_damping[5], damping=damping_ad[5], order=orders[5])
        self.c_theta1z2.plot(mode=mode, conv=conv, ax=axs_damping[6], damping=damping_ad[6], order=orders[6])
        self.c_theta1theta2.plot(mode=mode, conv=conv, ax=axs_damping[7], damping=damping_ad[7], order=orders[7])

        self.c_z2z1.plot(mode=mode, conv=conv, ax=axs_damping[8], damping=damping_ad[8], order=orders[8])
        self.c_z2theta1.plot(mode=mode, conv=conv, ax=axs_damping[9], damping=damping_ad[9], order=orders[9])
        self.c_z2z2.plot(mode=mode, conv=conv, ax=axs_damping[10], damping=damping_ad[10], order=orders[10])
        self.c_z2theta2.plot(mode=mode, conv=conv, ax=axs_damping[11], damping=damping_ad[11], order=orders[11])

        self.c_theta2z1.plot(mode=mode, conv=conv, ax=axs_damping[12], damping=damping_ad[12], order=orders[12])
        self.c_theta2theta1.plot(mode=mode, conv=conv, ax=axs_damping[13], damping=damping_ad[13], order=orders[13])
        self.c_theta2z2.plot(mode=mode, conv=conv, ax=axs_damping[14], damping=damping_ad[14], order=orders[14])
        self.c_theta2theta2.plot(mode=mode, conv=conv, ax=axs_damping[15], damping=damping_ad[15], order=orders[15])

        # --- Plot Stiffness Derivatives ---
        self.k_z1z1.plot(mode=mode, conv=conv, ax=axs_stiffness[0], damping=damping_ad[16], order=orders[16])
        self.k_z1theta1.plot(mode=mode, conv=conv, ax=axs_stiffness[1], damping=damping_ad[17], order=orders[17])
        self.k_z1z2.plot(mode=mode, conv=conv, ax=axs_stiffness[2], damping=damping_ad[18], order=orders[18])
        self.k_z1theta2.plot(mode=mode, conv=conv, ax=axs_stiffness[3], damping=damping_ad[19], order=orders[19])

        self.k_theta1z1.plot(mode=mode, conv=conv, ax=axs_stiffness[4], damping=damping_ad[20], order=orders[20])
        self.k_theta1theta1.plot(mode=mode, conv=conv, ax=axs_stiffness[5], damping=damping_ad[21], order=orders[21])
        self.k_theta1z2.plot(mode=mode, conv=conv, ax=axs_stiffness[6], damping=damping_ad[22], order=orders[22])
        self.k_theta1theta2.plot(mode=mode, conv=conv, ax=axs_stiffness[7], damping=damping_ad[23], order=orders[23])

        self.k_z2z1.plot(mode=mode, conv=conv, ax=axs_stiffness[8], damping=damping_ad[24], order=orders[24])
        self.k_z2theta1.plot(mode=mode, conv=conv, ax=axs_stiffness[9], damping=damping_ad[25], order=orders[25])
        self.k_z2z2.plot(mode=mode, conv=conv, ax=axs_stiffness[10], damping=damping_ad[26], order=orders[26])
        self.k_z2theta2.plot(mode=mode, conv=conv, ax=axs_stiffness[11], damping=damping_ad[27], order=orders[27])

        self.k_theta2z1.plot(mode=mode, conv=conv, ax=axs_stiffness[12], damping=damping_ad[28], order=orders[28])
        self.k_theta2theta1.plot(mode=mode, conv=conv, ax=axs_stiffness[13], damping=damping_ad[29], order=orders[29])
        self.k_theta2z2.plot(mode=mode, conv=conv, ax=axs_stiffness[14], damping=damping_ad[30], order=orders[30])
        self.k_theta2theta2.plot(mode=mode, conv=conv, ax=axs_stiffness[15], damping=damping_ad[31], order=orders[31])

        # Remove x-axis labels from top 12 subplots for visual clarity
        for k in range(12):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")

        # Resize figures for better layout
        scal = 1.8
        fig_damping.set_size_inches(20/scal, 15/scal)
        fig_stiffness.set_size_inches(20/scal, 15/scal)

        # Adjust layout to avoid overlap
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()


    def plot_to_compare(self, fig_damping=[], fig_stiffness=[], conv='normal', mode='poly only', orders=np.ones(32, dtype=int)*2, label='i'):
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
            Array of length 32 specifying the polynomial fit order for each derivative.

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

        # Boolean array defining which derivatives are damping (first 16) and which are stiffness (last 16)
        damping_ad = np.array([True]*16 + [False]*16)

        # Retrieve subplot axes for both damping and stiffness figures
        axs_damping = fig_damping.get_axes()
        axs_stiffness = fig_stiffness.get_axes()

        # --- Damping Plots ---
        self.c_z1z1.plot2(mode=mode, conv=conv, ax=axs_damping[0], damping=damping_ad[0], order=orders[0], label=label)
        self.c_z1theta1.plot2(mode=mode, conv=conv, ax=axs_damping[1], damping=damping_ad[1], order=orders[1], label=label)
        self.c_z1z2.plot2(mode=mode, conv=conv, ax=axs_damping[2], damping=damping_ad[2], order=orders[2], label=label)
        self.c_z1theta2.plot2(mode=mode, conv=conv, ax=axs_damping[3], damping=damping_ad[3], order=orders[3], label=label)

        self.c_theta1z1.plot2(mode=mode, conv=conv, ax=axs_damping[4], damping=damping_ad[4], order=orders[4], label=label)
        self.c_theta1theta1.plot2(mode=mode, conv=conv, ax=axs_damping[5], damping=damping_ad[5], order=orders[5], label=label)
        self.c_theta1z2.plot2(mode=mode, conv=conv, ax=axs_damping[6], damping=damping_ad[6], order=orders[6], label=label)
        self.c_theta1theta2.plot2(mode=mode, conv=conv, ax=axs_damping[7], damping=damping_ad[7], order=orders[7], label=label)

        self.c_z2z1.plot2(mode=mode, conv=conv, ax=axs_damping[8], damping=damping_ad[8], order=orders[8], label=label)
        self.c_z2theta1.plot2(mode=mode, conv=conv, ax=axs_damping[9], damping=damping_ad[9], order=orders[9], label=label)
        self.c_z2z2.plot2(mode=mode, conv=conv, ax=axs_damping[10], damping=damping_ad[10], order=orders[10], label=label)
        self.c_z2theta2.plot2(mode=mode, conv=conv, ax=axs_damping[11], damping=damping_ad[11], order=orders[11], label=label)

        self.c_theta2z1.plot2(mode=mode, conv=conv, ax=axs_damping[12], damping=damping_ad[12], order=orders[12], label=label)
        self.c_theta2theta1.plot2(mode=mode, conv=conv, ax=axs_damping[13], damping=damping_ad[13], order=orders[13], label=label)
        self.c_theta2z2.plot2(mode=mode, conv=conv, ax=axs_damping[14], damping=damping_ad[14], order=orders[14], label=label)
        self.c_theta2theta2.plot2(mode=mode, conv=conv, ax=axs_damping[15], damping=damping_ad[15], order=orders[15], label=label)

        # --- Stiffness Plots ---
        self.k_z1z1.plot2(mode=mode, conv=conv, ax=axs_stiffness[0], damping=damping_ad[16], order=orders[16], label=label)
        self.k_z1theta1.plot2(mode=mode, conv=conv, ax=axs_stiffness[1], damping=damping_ad[17], order=orders[17], label=label)
        self.k_z1z2.plot2(mode=mode, conv=conv, ax=axs_stiffness[2], damping=damping_ad[18], order=orders[18], label=label)
        self.k_z1theta2.plot2(mode=mode, conv=conv, ax=axs_stiffness[3], damping=damping_ad[19], order=orders[19], label=label)

        self.k_theta1z1.plot2(mode=mode, conv=conv, ax=axs_stiffness[4], damping=damping_ad[20], order=orders[20], label=label)
        self.k_theta1theta1.plot2(mode=mode, conv=conv, ax=axs_stiffness[5], damping=damping_ad[21], order=orders[21], label=label)
        self.k_theta1z2.plot2(mode=mode, conv=conv, ax=axs_stiffness[6], damping=damping_ad[22], order=orders[22], label=label)
        self.k_theta1theta2.plot2(mode=mode, conv=conv, ax=axs_stiffness[7], damping=damping_ad[23], order=orders[23], label=label)

        self.k_z2z1.plot2(mode=mode, conv=conv, ax=axs_stiffness[8], damping=damping_ad[24], order=orders[24], label=label)
        self.k_z2theta1.plot2(mode=mode, conv=conv, ax=axs_stiffness[9], damping=damping_ad[25], order=orders[25], label=label)
        self.k_z2z2.plot2(mode=mode, conv=conv, ax=axs_stiffness[10], damping=damping_ad[26], order=orders[26], label=label)
        self.k_z2theta2.plot2(mode=mode, conv=conv, ax=axs_stiffness[11], damping=damping_ad[27], order=orders[27], label=label)

        self.k_theta2z1.plot2(mode=mode, conv=conv, ax=axs_stiffness[12], damping=damping_ad[28], order=orders[28], label=label)
        self.k_theta2theta1.plot2(mode=mode, conv=conv, ax=axs_stiffness[13], damping=damping_ad[29], order=orders[29], label=label)
        self.k_theta2z2.plot2(mode=mode, conv=conv, ax=axs_stiffness[14], damping=damping_ad[30], order=orders[30], label=label)
        self.k_theta2theta2.plot2(mode=mode, conv=conv, ax=axs_stiffness[15], damping=damping_ad[31], order=orders[31], label=label)

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


def plot_compare(AD_single, AD_1D, AD_2D, AD_3D, AD_4D, AD_5D):
    """
    Compares aerodynamic derivative polynomial fits across different gap distances (1D to 5D and single).

    Parameters:
    -----------
    AD_single : object
        Aerodynamic derivative object for the single-section case.

    AD_1D to AD_5D : objects
        Aerodynamic derivative objects for each gap distance (1D to 5D).

    Returns:
    --------
    fig_damping : matplotlib.figure.Figure
        Figure object containing damping comparison plots.

    fig_stiffness : matplotlib.figure.Figure
        Figure object containing stiffness comparison plots.
    """

    # Create figure with 4x4 subplots for damping terms
    fig_damping = plt.figure()
    for k in range(16):
        fig_damping.add_subplot(4, 4, k + 1)

    # Create figure with 4x4 subplots for stiffness terms
    fig_stiffness = plt.figure()
    for k in range(16):
        fig_stiffness.add_subplot(4, 4, k + 1)

    # Plot polynomial fits for each model (1D to 5D and single) into the same axes for comparison
    AD_1D.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness, label='1D')
    AD_2D.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness, label='2D')
    AD_3D.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness, label='3D')
    AD_4D.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness, label='4D')
    AD_5D.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness, label='5D')
    AD_single.plot_to_compare(fig_damping=fig_damping, fig_stiffness=fig_stiffness)

    # Extract legend handles and labels from the first subplot in the damping figure
    handles, labels = fig_damping.axes[0].get_legend_handles_labels()

    # Remove individual legends from each subplot to avoid clutter
    for ax in fig_damping.get_axes():
        ax.legend().remove()
    for ax in fig_stiffness.get_axes():
        ax.legend().remove()

    # Add a single shared legend above each figure
    fig_damping.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=6, frameon=False)
    fig_stiffness.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         ncol=6, frameon=False)

    # Adjust layout to leave space for the shared legend
    fig_damping.tight_layout(rect=[0, 0, 1, 0.97])
    fig_stiffness.tight_layout(rect=[0, 0, 1, 0.97])

    # Display the plots
    plt.show()

    return fig_damping, fig_stiffness


        
