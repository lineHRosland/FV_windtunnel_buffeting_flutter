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
    plot()
        plots the aerodynamic derivative        
    
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
    
    #Definining a function which fits a polynomial to the aerodynamic derivative
    def poly_fit(self, damping=True, order=2):
        ads = np.zeros(self.reduced_velocities.shape[0])
        vreds = np.zeros(self.reduced_velocities.shape[0])
        
        ads = self.value
        vreds = self.reduced_velocities

        if ads.size > 0:
            poly_coeff = np.zeros(np.max(order)+1)
            v_range = np.zeros(2)
            
            v_range[1] = np.max(vreds)
            v_range[0] = np.min(vreds)
                
            if damping == True:
                poly_coeff = np.polyfit(vreds,ads,order)
            elif damping == False:
                poly_coeff = np.polyfit(vreds,ads,order)    
                    
            return poly_coeff, v_range
        else:
            return None, None
 
    #Defining a function which plots the aerodynamic derivative
    def plot(self, mode = "total", conv = "normal", ax=[], damping=True, order=2):
        """ plots the aerodynamic derivative
        
        The method plots the aerodynamic derivative as function of the reduced 
        velocity. Two optimal modes are available.
        
        parameters:
        ----------
        mode : str, optional
            selects the plot mode
        conv: str, optional
            selects which convention to use when plotting
        ax : pyplot figure instance
        damping : bool, optional
            selects to clarify if the AD is associated with damping or stiffness if a polynomial fit is to be performed 
        order : int, optional
            selects the order of the polynomial fit if a polynomial fit is to be performed 
        ---------        
        
        """
        #Using the poly_fit function to fit a polynomial to the aerodynamic derivative
        poly_coeff, v_range = self.poly_fit(damping=damping, order=order)
        if poly_coeff is None:
            V = 0
            y = 0
        else: 
            p = np.poly1d(poly_coeff)
            V = np.linspace(v_range[0], v_range[1], 200)
            y = p(V)

        if bool(ax) == False:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        #Plotting of the aerodynamic derivative found from each test 
        if mode == "total":
            ax.plot(self.reduced_velocities,self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
            ax.set_ylabel(("$" + self.label + "$"))
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.grid(True)

        #Plotting of the aerodynamic derivative found from each test and the polynomial fit
        elif mode == "total+poly":
            ax.plot(self.reduced_velocities,self.ad_load_cell_a + self.ad_load_cell_b, "o", label="Total")
            ax.plot(V, y)
            ax.set_ylabel(("$" + self.label + "$"))
            ax.set_xlabel(r"Reduced velocity $\hat{V}$")
            ax.grid(True)            

    #Defining a function which plots only the polynomial fit of the aerodynamic derivative
    def plot2(self, mode = "poly only", conv = "normal", ax=[], damping=True, order=2, label='i'):
        """ plots the polynomial fit of the aerodynamic derivative
        
        The method plots the polynomial fit of the aerodynamic derivative as function of the 
        reduced velocity. One optimal mode is available.
        
        parameters:
        ----------
        mode : str, optional
            selects the plot mode
        conv: str, optional
            selects which convention to use when plotting
        ax : pyplot figure instance
        damping : bool, optional
            selects to clarify if the AD is associated with damping or stiffness if a polynomial fit is to be performed 
        order : int, optional
            selects the order of the polynomial fit if a polynomial fit is to be performed
        label : str, optional
            selects the label of the plot 
        ---------        
        
        """
        #Using the poly_fit function to fit a polynomial to the aerodynamic derivative
        poly_coeff, v_range = self.poly_fit(damping=damping, order=order)
        if poly_coeff is None:
            V = 0
            y = 0
        else: 
            p = np.poly1d(poly_coeff)
            V = np.linspace(v_range[0], v_range[1], 200)
            y = p(V)

        if bool(ax) == False:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
    
        #Plotting of only the polynomial fit of the aerodynamic derivative
        if mode == "poly only":
            ax.plot(V, y, label=f'{label}')
            ax.set_ylabel(("$" + self.label + "$"))
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
    def fromWTT(cls,experiment_in_still_air,experiment_in_wind,section_width, section_length_1, section_length_2, upstream_in_rig = True, filter_order = 6, cutoff_frequency = 7):
        """ obtains an instance of the class Aerodynamic derivatives from a wind tunnel experiment
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind      : instance of the class experiment
        section_width           : width of the bridge deck section model
        section_length_1        : length of the upstream section model
        section_length_2        : length of the downstream section model
        upstream_in_rig         : bool
            determines if the upstream section is in the rig or not
        filter_order            : int
            order of the butterworth filter
        cutoff_frequency        : float
            cutoff frequency of the butterworth filter
        ---------
        
        returns:
        --------
        an instance of the class AerodynamicDerivatives4x4
        to instances of the class Experiment, one for model predictions and one for data used to fit the model
        
        
        """
        experiment_in_wind.align_with(experiment_in_still_air)
        experiment_in_wind_still_air_forces_removed = deepcopy(experiment_in_wind)
        experiment_in_wind_still_air_forces_removed.substract(experiment_in_still_air)
        starts, stops = experiment_in_wind_still_air_forces_removed.harmonic_groups()
        
        
        frequencies_of_motion = np.zeros(len(starts))
        reduced_velocities = np.zeros(len(starts))
        mean_wind_speeds = np.zeros(len(starts))
        
        normalized_coefficient_matrix = np.zeros((2,3,len(starts),4))
        
        forces_predicted_by_ads = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape[0],24))
        #model_forces = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape[0],3))
        
        # loop over all single harmonic test in the time series
        # 6 single harmonic tests are performed for each motion type
        for k in range(len(starts)):           

            sampling_frequency = 1/(experiment_in_still_air.time[1]- experiment_in_still_air.time[0])
       
            sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
           
            #r_x,1 r_z,1 r_theta,1 (bru i rigg, avlest av riggen)
            motions_bridge_in_rig = experiment_in_wind_still_air_forces_removed.motion 

            #r_x,2 r_z,2 r_theta,2 (bru på vegg, ingen bevegelser)
            motions_bridge_on_wall = np.zeros_like(motions_bridge_in_rig)

            #r_x,1 r_z,1 r_theta,1 r_x,2 r_z,2 r_theta,2
            motions = np.hstack((motions_bridge_in_rig, motions_bridge_on_wall))
            
            motions = spsp.sosfiltfilt(sos,motions,axis=0)

            #r_x,1\dot r_z,1\dot r_theta,1\dot r_x,2\dot r_z,2\dot r_theta,2\dot
            time_derivative_motions = np.vstack((np.array([0,0,0,0,0,0]),np.diff(motions,axis=0)))*sampling_frequency
            
            motion_type = experiment_in_wind_still_air_forces_removed.motion_type()

            #print("Motion type: " + str(motion_type))
            fourier_amplitudes = np.fft.fft(motions[starts[k]:stops[k],motion_type]-np.mean(motions[starts[k]:stops[k],motion_type]))
            
            
            time_step = experiment_in_wind_still_air_forces_removed.time[1]- experiment_in_wind_still_air_forces_removed.time[0]
            
            peak_index = np.argmax(np.abs(fourier_amplitudes[0:int(len(fourier_amplitudes)/2)]))
            
            frequencies = np.fft.fftfreq(len(fourier_amplitudes),time_step)
            
            frequency_of_motion = frequencies[peak_index]
            frequencies_of_motion[k] = frequency_of_motion
         
            #X-matrise, men bare det som ikke er null --> r_i\dot r_i, i er motion type for testen som behandles
            regressor_matrix = np.vstack((time_derivative_motions[starts[k]:stops[k],motion_type],motions[starts[k]:stops[k],motion_type])).T

            #Pseudoinvers av X-matrise  --> X^+           
            pseudo_inverse_regressor_matrix = spla.pinv(regressor_matrix) 
            selected_forces = np.array([0,2,4])
            
            mean_wind_speed = np.mean(experiment_in_wind_still_air_forces_removed.wind_speed[starts[k]:stops[k]])
            mean_wind_speeds[k] = mean_wind_speed
                
            reduced_frequency  = frequency_of_motion*2*np.pi*section_width/mean_wind_speed
            
            reduced_velocities[k] = 1/reduced_frequency

            section_length = 0
            
            #model_forces = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape))
            
            # Loop over all load cells
            for m in range(4):
                if m == 0 or m == 1:
                    section_length = section_length_1
                elif m == 2 or m == 3:
                    section_length = section_length_2

                #q_x, q_z, q_theta for the cell being considered
                forces = experiment_in_wind_still_air_forces_removed.forces_global_center[starts[k]:stops[k],selected_forces + 6*m]
                forces_mean_wind_removed = forces - np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400,selected_forces + 6*m],axis= 0)

                #E = X^+ * q
                coefficient_matrix = pseudo_inverse_regressor_matrix @ forces_mean_wind_removed
                                
                #Normalisering av koeffiesienter --> løse ut c^* og k^* fra E-matrisen
                normalized_coefficient_matrix[:,:,k,m] = np.copy(coefficient_matrix)
                normalized_coefficient_matrix[0,:,k,m] = normalized_coefficient_matrix[0,:,k,m]*2  / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed / reduced_frequency / section_width / section_length
                normalized_coefficient_matrix[1,:,k,m] = normalized_coefficient_matrix[1,:,k,m]*2  /experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed**2 / reduced_frequency**2 /section_length
                normalized_coefficient_matrix[:,2,k,m] = normalized_coefficient_matrix[:,2,k,m]/section_width
                
                if motion_type ==2:
                    normalized_coefficient_matrix[:,:,k,m] = normalized_coefficient_matrix[:,:,k,m]/section_width 
                
                forces_predicted_by_ads[starts[k]:stops[k],selected_forces + 6*m] = forces_predicted_by_ads[starts[k]:stops[k],selected_forces + 6*m]  + regressor_matrix @ coefficient_matrix + np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
            
               
        # Make Experiment object for simulation of model
        obj1 = experiment_in_wind_still_air_forces_removed
        obj2 = experiment_in_still_air
        model_prediction = Experiment(obj1.name, obj1.time, obj1.temperature, obj1.air_density, obj1.wind_speed,[],forces_predicted_by_ads,obj2.motion)
                 
        
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
        """ appends and instance of AerodynamicDerivatives2x2 to self
        
        Arguments:
        ----------
        ads         : an instance of the class AerodynamicDerivatives2x2
        
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
        """ Returns a matrix of aerodynamic derivatives and reduced velocities
        
        Returns
        -------
        ads : float
        
        a matrix of aerodynamic derivatives [32 x N reduced velocities]
        
        vreds : float
        
        a matrix of reduced velocities [32 x N reduced velocities]
        
        
        
        """
        ads = np.zeros((32,self.c_z1z1.reduced_velocities.shape[0]))
        vreds = np.zeros((32,self.c_z1z1.reduced_velocities.shape[0]))
        
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
    
    def fit_poly(self,orders = np.ones(32,dtype=int)*2):
        ad_matrix, vreds = self.ad_matrix
        
        poly_coeff = np.zeros((32,np.max(orders)+1))
        v_range = np.zeros((32,2))
        
        damping_ad = np.array([True, True, True, True, True, True, True, True,
                               True, True, True, True, True, True, True, True,
                               False, False, False, False, False, False, False, False,
                               False, False, False, False, False, False, False, False])
        
        
        for k in range(32):
            v_range[k,1] = np.max(vreds)
            v_range[k,0] = np.min(vreds)
            
            if damping_ad[k] == True:
                poly_coeff[k,:] = np.polyfit(vreds[k,:],ad_matrix[k,:],orders[k])
            elif damping_ad[k] == False:
                poly_coeff[k,:] = np.polyfit(vreds[k,:],ad_matrix[k,:],orders[k])
            
                
        return poly_coeff, v_range  
    
    #Ikke endret
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


    def plot(self, fig_damping=[],fig_stiffness=[],conv='normal', mode='total', orders = np.ones(32,dtype=int)*2):
        
        """ plots all aerodynamic derivatives
        
        Arguments:
        ----------
        fig_damping     : figure object
        
        fig_stiffness   : figure object
        
        conv            : normal or zasso
        
        mode            : total, all or decks        
        
        """
        
        # Make figure objects if not given
        if bool(fig_damping) == False:
            fig_damping = plt.figure()
            for k in range(16):
                fig_damping.add_subplot(4,4,k+1)
        
        if bool(fig_stiffness) == False:
            fig_stiffness = plt.figure()
            for k in range(16):
                fig_stiffness.add_subplot(4,4,k+1)
        
        #Get the poly fit for the ADs
        damping_ad = np.array([True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True,
                            False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False])
 

        axs_damping = fig_damping.get_axes()
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

        axs_stiffness = fig_stiffness.get_axes()
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

        for k in range(12):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")

        scal = 1.8
        fig_damping.set_size_inches(20/scal,15/scal)
        fig_stiffness.set_size_inches(20/scal,15/scal)

        '''
        fig_damping.set_size_inches(20/2.54,15/2.54)
        fig_stiffness.set_size_inches(20/2.54,15/2.54)
        '''
        
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()

    def plot_to_compare(self, fig_damping=[],fig_stiffness=[],conv='normal', mode='poly only', orders = np.ones(32,dtype=int)*2, label='i'):
        # Make figure objects if not given
        if bool(fig_damping) == False:
            fig_damping = plt.figure()
            for k in range(16):
                fig_damping.add_subplot(4,4,k+1)
        
        if bool(fig_stiffness) == False:
            fig_stiffness = plt.figure()
            for k in range(16):
                fig_stiffness.add_subplot(4,4,k+1)
        
        #Get the poly fit for the ADs
        damping_ad = np.array([True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True,
                            False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False])

        axs_damping = fig_damping.get_axes()
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

        axs_stiffness = fig_stiffness.get_axes()
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

        for k in range(12):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")

        scal = 1.8
        fig_damping.set_size_inches(20/scal,15/scal)
        fig_stiffness.set_size_inches(20/scal,15/scal)

        '''
        fig_damping.set_size_inches(20/2.54,15/2.54)
        fig_stiffness.set_size_inches(20/2.54,15/2.54)
        '''
        
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()



def plot_compare(AD_single, AD_1D, AD_2D, AD_3D, AD_4D, AD_5D):
    fig_damping = plt.figure()
    for k in range(16):
        fig_damping.add_subplot(4, 4, k + 1)
    fig_stiffness = plt.figure()
    for k in range(16):
        fig_stiffness.add_subplot(4, 4, k + 1)

    AD_1D.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='1D')
    AD_2D.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='2D')
    AD_3D.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='3D')
    AD_4D.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='4D')
    AD_5D.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='5D')
    AD_single.plot_to_compare(fig_damping = fig_damping, fig_stiffness=fig_stiffness, label='Single')


    # Get legend handles/labels from the first subplot
    handles, labels = fig_damping.axes[0].get_legend_handles_labels()

    # Remove legends from individual subplots
    for ax in fig_damping.get_axes():
        ax.legend().remove()
    for ax in fig_stiffness.get_axes():
        ax.legend().remove()

    # Add single shared legend to each figure
    fig_damping.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=6, frameon=False)
    fig_stiffness.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         ncol=6, frameon=False)

    fig_damping.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for legend
    fig_stiffness.tight_layout(rect=[0, 0, 1, 0.97])

    # Show plots
    plt.show()

    return fig_damping, fig_stiffness


        
