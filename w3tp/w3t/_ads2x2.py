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



__all__ = ["AerodynamicDerivatives2x2","AerodynamicDerivative2x2",]

    
   
class AerodynamicDerivative2x2:
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
    def __init__(self,label="x",reduced_velocities=[],ad_load_cell_1=[],ad_load_cell_2=[],ad_load_cell_3=[],ad_load_cell_4=[],mean_wind_speeds=[], frequencies=[]):
        """  
            
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
        
        """
        self.reduced_velocities = reduced_velocities
        self.ad_load_cell_1 = ad_load_cell_1
        self.ad_load_cell_2 = ad_load_cell_2
        self.ad_load_cell_3 = ad_load_cell_3
        self.ad_load_cell_4 = ad_load_cell_4
        self.mean_wind_speeds = mean_wind_speeds
        self.frequencies = frequencies
        self.label = label
    
    @property    
    def value(self):
        return self.ad_load_cell_1 + self.ad_load_cell_2
    
    def poly_fit(self, damping=True, order=2):
        ads = np.zeros(self.reduced_velocities.shape[0])
        vreds = np.zeros(self.reduced_velocities.shape[0])
        
        ads = self.value
        vreds = self.reduced_velocities

        if ads.size > 0:
            poly_coeff = np.zeros(np.max(order)+1)
            k_range = np.zeros(2)
            
            k_range[0] = 1/np.max(vreds)
            k_range[1] = 1/np.min(vreds)
                
            if damping == True:
                poly_coeff = np.polyfit(vreds,ads,order)
            elif damping == False:
                poly_coeff = np.polyfit(vreds,ads,order)    
                    
            return poly_coeff, k_range
        else:
            return None, None
    
        
        
    def plot(self, mode = "all", conv = "normal", ax=[], V=1.0, damping=True, order=2):
        """ plots the aerodynamic derivative
        
        The method plots the aerodynamic derivative as function of the mean 
        wind speed. Four optimal modes are abailable.
        
        parameters:
        ----------
        mode : str, optional
            selects the plot mode
        conv: str, optional
            selects which convention to use when plotting
        fig : pyplot figure instance    
        ---------        
        
        """
        #Poly fit
        poly_coeff, k_range = self.poly_fit(damping=True, order=2)
        if poly_coeff is None:
            V = 0
            y = 0
        else: 
            p = np.poly1d(poly_coeff)
            V = np.linspace(1/k_range[1], 1/k_range[0], 200)
            y = p(V)


        if bool(ax) == False:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        
        
        if conv == "normal":
            if mode == "all":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
                ax.plot(self.reduced_velocities,self.ad_load_cell_1, "o", label="Load cell 1", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_2, "o", label="Load cell 2", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_3, "o", label="Load cell 3", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_4, "o", label="Load cell 4", alpha = 0.5)
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)
            
            elif mode == "decks":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2, "o", label="Upwind deck", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Downwind deck", alpha = 0.5)
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid(True)
                ax.legend()
                
            elif mode == "total":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2, "o", label="Total")
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid(True)

            elif mode == "velocity":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2, "o", label=f"V = {V:.1f} m/s")
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)

            elif mode == "total+poly":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_1, "o", label="Total")
                ax.plot(V, y)
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid(True)            

            #plt.tight_layout()
                
        elif conv == "zasso" and len(self.reduced_velocities) != 0:
            damping_ads =["P_1^*","P_2^*", "P_5^*", "H_1^*", "H_2^*", "H_5^*", "A_1^*", "A_2^*", "A_5^*" ]
            stiffness_ads =["P_3^*","P_4^*", "P_6^*", "H_3^*", "H_4^*", "H_6^*", "A_3^*", "A_4^*", "A_6^*" ]
            
             
            if self.label in damping_ads:
                factor = 1.0/self.reduced_velocities
                K_label = "K"
            elif self.label in stiffness_ads:
                factor = 1.0/self.reduced_velocities**2
                K_label = "K^2"
            else:
                print("ERROR")

            
            if mode == "all":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_1, "o", label="Load cell 1", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_2, "o", label="Load cell 2", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_3, "o", label="Load cell 3", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_4, "o", label="Load cell 4", alpha = 0.5)
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)
            
            elif mode == "decks":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2), "o", label="Upwind deck", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Downwind deck", alpha = 0.5)
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid(True)
                
            elif mode == "total":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.grid(True)
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
        #plt.tight_layout()
                

class AerodynamicDerivatives2x2:
    """
    A class used to represent all aerodynamic derivatives for a 3 dof motion
    
    parameters:
    ----------
    p1...p6 : obj
        aerodynamic derivatives related to the horizontal self-excited force
    h1...h6 : obj
        aerodynamic derivatives related to the vertical self-excited force
    a1...a6 : obj
        aerodynamic derivative related to the pitchingmoment
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
        parameters:
        ----------
        p1...p6 : obj
         aerodynamic derivatives related to the horizontal self-excited force
        h1...h6 : obj
         aerodynamic derivatives related to the vertical self-excited force
        a1...a6 : obj
         aerodynamic derivative related to the pitchingmoment
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
    def fromWTT(cls,experiment_in_still_air,experiment_in_wind,section_width,section_length, filter_order = 6, cutoff_frequency = 7):
        """ obtains an instance of the class Aerodynamic derivatives from a wind tunnel experiment
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind   : instance of the class experiment
        section_width        : width of the bridge deck section model
        section_length       : length of the section model
        ---------
        
        returns:
        --------
        an instance of the class AerodynamicDerivatives
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
        for k in range(len(starts)):    # 6 frequencies       

            sampling_frequency = 1/(experiment_in_still_air.time[1]- experiment_in_still_air.time[0])
       
            sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
           
            motions = experiment_in_wind_still_air_forces_removed.motion
            motions = spsp.sosfiltfilt(sos,motions,axis=0)
            
            time_derivative_motions = np.vstack((np.array([0,0,0]),np.diff(motions,axis=0)))*sampling_frequency
            
            #max_hor_vert_pitch_motion = [np.max(motions[:,0]), np.max(motions[:,1]), np.max(motions[:,2]) ]
            motion_type = experiment_in_wind_still_air_forces_removed.motion_type()
            #print("Motion type: " + str(motion_type))
            fourier_amplitudes = np.fft.fft(motions[starts[k]:stops[k],motion_type]-np.mean(motions[starts[k]:stops[k],motion_type]))
            
            time_step = experiment_in_wind_still_air_forces_removed.time[1]- experiment_in_wind_still_air_forces_removed.time[0]
            
            peak_index = np.argmax(np.abs(fourier_amplitudes[0:int(len(fourier_amplitudes)/2)]))
            
            frequencies = np.fft.fftfreq(len(fourier_amplitudes),time_step)
            
            frequency_of_motion = frequencies[peak_index]
            frequencies_of_motion[k] = frequency_of_motion
         
            regressor_matrix = np.vstack((time_derivative_motions[starts[k]:stops[k],motion_type],motions[starts[k]:stops[k],motion_type])).T
                        
            pseudo_inverse_regressor_matrix = spla.pinv(regressor_matrix) 
            selected_forces = np.array([0,2,4])
            
            
            mean_wind_speed = np.mean(experiment_in_wind_still_air_forces_removed.wind_speed[starts[k]:stops[k]])
            mean_wind_speeds[k] = mean_wind_speed
                
            reduced_frequency  = frequency_of_motion*2*np.pi*section_width/mean_wind_speed
            
            reduced_velocities[k] = 1/reduced_frequency
            
            #model_forces = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape))
            
            # Loop over all load cells
            for m in range(4):            
                forces = experiment_in_wind_still_air_forces_removed.forces_global_center[starts[k]:stops[k],selected_forces + 6*m]
                froces_mean_wind_removed = forces - np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
                                
                coefficient_matrix = pseudo_inverse_regressor_matrix @ froces_mean_wind_removed
                                
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
            
        h1 = AerodynamicDerivative2x2()
        h2 = AerodynamicDerivative2x2()
        h3 = AerodynamicDerivative2x2()
        h4 = AerodynamicDerivative2x2()
            
        a1 = AerodynamicDerivative2x2()
        a2 = AerodynamicDerivative2x2()
        a3 = AerodynamicDerivative2x2()
        a4 = AerodynamicDerivative2x2()

        meanV = np.mean(mean_wind_speeds)

        zero_matrix = np.zeros_like(normalized_coefficient_matrix[0,0,:,0]) 

        if motion_type ==0:
            row = 0
        elif motion_type ==1:
            row = 0
            col = 1
            h1 = AerodynamicDerivative2x2("H_1^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
            col = 2
            a1 = AerodynamicDerivative2x2("A_1^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 1
            h4 = AerodynamicDerivative2x2("H_4^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
            col = 2
            a4 = AerodynamicDerivative2x2("A_4^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
        elif motion_type ==2:
            row = 0
            col = 1
            h2 = AerodynamicDerivative2x2("H_2^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
            col = 2
            a2 = AerodynamicDerivative2x2("A_2^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 1
            h3 = AerodynamicDerivative2x2("H_3^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
            col = 2
            a3 = AerodynamicDerivative2x2("A_3^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],zero_matrix,zero_matrix,mean_wind_speeds,frequencies_of_motion)
              
        return cls(h1, h2, h3, h4, a1, a2, a3, a4, meanV), model_prediction, experiment_in_wind_still_air_forces_removed
    
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
        """ appends and instance of AerodynamicDerivatives to self
        
        Arguments:
        ----------
        ads         : an instance of the class AerodynamicDerivatives
        
        """
        objs1 = [self.h1, self.h2, self.h3, self.h4, self.a1, self.a2, self.a3, self.a4]
        objs2 = [ads.h1, ads.h2, ads.h3, ads.h4, ads.a1, ads.a2, ads.a3, ads.a4]
        
        for k in range(len(objs1)):
            objs1[k].ad_load_cell_1 = np.append(objs1[k].ad_load_cell_1,objs2[k].ad_load_cell_1)
            objs1[k].ad_load_cell_2 = np.append(objs1[k].ad_load_cell_2,objs2[k].ad_load_cell_2) 
            objs1[k].ad_load_cell_3 = np.append(objs1[k].ad_load_cell_3,objs2[k].ad_load_cell_3) 
            objs1[k].ad_load_cell_4 = np.append(objs1[k].ad_load_cell_4,objs2[k].ad_load_cell_4) 
            
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
    
    
    def frf_mat(self,mean_wind_velocity = 1.0, section_width = 1.0, air_density = 1.25):
        
        
        frf_mat = np.zeros((2,2,len(self.p1.reduced_velocities)),dtype=complex)

        frf_mat[0,0,:] = 1/2*air_density*mean_wind_velocity**2 * (1/self.h1.reduced_velocities)**2 * (self.h1.value*1j + self.h4.value)
        frf_mat[0,1,:] = 1/2*air_density*mean_wind_velocity**2 * section_width*(1/self.h3.reduced_velocities)**2 * (self.h2.value*1j + self.h3.value)
        
        frf_mat[1,0,:] = 1/2*air_density*mean_wind_velocity**2 * section_width*(1/self.a1.reduced_velocities)**2 * (self.a1.value*1j + self.a4.value)
        frf_mat[1,1,:] = 1/2*air_density*mean_wind_velocity**2 * section_width**2*(1/self.a2.reduced_velocities)**2 * (self.a2.value*1j + self.a3.value)
        
        return frf_mat
    

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


        
    def plot(self, fig_damping=[],fig_stiffness=[],conv='normal', mode='total', orders = np.ones(8,dtype=int)*2):
        
        """ plots all aerodynamic derivatives
        
        Arguments:
        ----------
        fig_damping     : figure object
        
        fig_stiffness   : figure object
        
        conv            : normal or zasso
        
        mode            : total, all, decks, velocity        
        
        """
        
        # Make figure objects if not given
        if bool(fig_damping) == False:
            fig_damping = plt.figure()
            for k in range(4):
                fig_damping.add_subplot(2,2,k+1)
        
        if bool(fig_stiffness) == False:
            fig_stiffness = plt.figure()
            for k in range(4):
                fig_stiffness.add_subplot(2,2,k+1)

        damping_ad = np.array([True, True, False, False,  True, True, False, False])
        
        axs_damping = fig_damping.get_axes()
#        
        self.h1.plot(mode=mode, conv=conv, ax=axs_damping[0], V=self.meanV, damping=damping_ad[0], order=orders[0])
        self.h2.plot(mode=mode, conv=conv, ax=axs_damping[1], V=self.meanV, damping=damping_ad[1], order=orders[1])

        self.a1.plot(mode=mode, conv=conv, ax=axs_damping[2], V=self.meanV, damping=damping_ad[2], order=orders[2])
        self.a2.plot(mode=mode, conv=conv, ax=axs_damping[3], V=self.meanV, damping=damping_ad[3], order=orders[3])
        
        axs_stiffness = fig_stiffness.get_axes()

        self.h4.plot(mode=mode, conv=conv, ax=axs_stiffness[0], V=self.meanV, damping=damping_ad[4], order=orders[4])
        self.h3.plot(mode=mode, conv=conv, ax=axs_stiffness[1], V=self.meanV, damping=damping_ad[5], order=orders[5])
        
        self.a4.plot(mode=mode, conv=conv, ax=axs_stiffness[2], V=self.meanV, damping=damping_ad[6], order=orders[6])
        self.a3.plot(mode=mode, conv=conv, ax=axs_stiffness[3], V=self.meanV, damping=damping_ad[7], order=orders[7])
        
        for k in range(2):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")
        
        fig_damping.set_size_inches(20/2.54,15/2.54)
        fig_stiffness.set_size_inches(20/2.54,15/2.54)
        
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()
         
      
