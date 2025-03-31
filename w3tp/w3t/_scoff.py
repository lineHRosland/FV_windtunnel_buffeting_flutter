# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:18:50 2022

@author: oiseth
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:09:00 2021

@author: oiseth
"""
import numpy as np
from scipy import signal as spsp
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import os
import copy  

__all__ = ["StaticCoeff",]
  
            
        
class StaticCoeff:
    """
    A class used to represent static force coefficients of a bridge deck
    
    Attributes:
    -----------
    drag_coeff : float
     drag coefficient (normalized drag force).
    lift_coeff : float
     lift coefficient (normalized lift force).
    pitch_coeff : float
     pitching moment coefficient (normalized pitching motion).
    pitch_motion : float
     pitch motion used in the wind tunnel tests.   
     
     
    Methods:
    ........
     
     fromWTT()
       obtains the static coefficients from a wind tunnel test.
     plot_drag()
       plots the drag coefficeint as function of the pitching motion     
     plot_lift()
       plots the lift coefficeint as function of the pitching motion      
     plot_pitch()
       plots the pitching coefficeint as function of the pitching motion 
    
    """
    def __init__(self,drag_coeff,lift_coeff,pitch_coeff,pitch_motion,mean_wind=[]):
        """
        parameters:
        -----------
        drag_coeff : float
          drag coefficient (normalized drag force).
        lift_coeff : float
          lift coefficient (normalized lift force).
        pitch_coeff : float
          pitching moment coefficient (normalized pitching motion).
        pitch_motion : float
          pitch motion used in the wind tunnel tests.   
        mean_wind : float
          the mean wind in which the static coefficeints have been obtained
    
    """
        self.drag_coeff = drag_coeff
        self.lift_coeff = lift_coeff
        self.pitch_coeff = pitch_coeff
        self.pitch_motion = pitch_motion
        self.mean_wind = mean_wind
        
    @classmethod
    def fromWTT(cls,experiment_in_still_air,experiment_in_wind,section_width,section_height,section_length_in_rig, section_length_on_wall, upwind_in_rig=True):  
        """ fromWTT obtains an instance of the class StaticCoeff from two experiments
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind : instance of the class experiment
        section_width : width of the bridge deck section model
        section_height : height of the bridge deck section model
        section_length_in_rig : length of the bridge deck section model in the wind tunnel rig
        section_length_on_wall : length of the bridge deck section model on the wall
        upwind_in_rig : boolean, optional
            True if the upwind deck is in the rig, False if the downwind deck is in the rig. The default is True.
                
        returns:
        -------
        instance of the class StaticCoeff
        
        """
        experiment_in_wind.align_with(experiment_in_still_air) #align the two experiments
        experiment_in_wind_still_air_forces_removed = deepcopy(experiment_in_wind)
        experiment_in_wind_still_air_forces_removed.substract(experiment_in_still_air) #remove the forces in still air 
        
        filter_order = 6
        cutoff_frequency = 1.0
        sampling_frequency = 1/(experiment_in_still_air.time[1]-experiment_in_still_air.time[0])
        
        sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos") #butterworth filtering of wind speed
        
        filtered_wind = np.nanmean(spsp.sosfiltfilt(sos,experiment_in_wind_still_air_forces_removed.wind_speed))
        #need a representative mean wind velocity U for the calculation of the coefficients

        # drag_coeff = forces * 2 / (rho * U^2 * h * L)
        drag_coeff_rig = experiment_in_wind_still_air_forces_removed.forces_global_center[:,0:12:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_height/section_length_in_rig
        drag_coeff_wall = experiment_in_wind_still_air_forces_removed.forces_global_center[:,12:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_height/section_length_on_wall
        #0:12:6 - horisontal dir: kolonne 0, 6 (from load cell nr. 1, 2)
        #12:24:6 - horisontal dir: kolonne 12, 18 (from load cell nr. 3, 4)

        # lift_coeff = forces * 2 / (rho * U^2 * b * L)
        lift_coeff_rig = experiment_in_wind_still_air_forces_removed.forces_global_center[:,2:12:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width/section_length_in_rig
        lift_coeff_wall = experiment_in_wind_still_air_forces_removed.forces_global_center[:,14:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width/section_length_on_wall
        #2:24:6 - vertical dir

        # pitch_coeff = moments * 2 / (rho * U^2 * b^2 * L)
        pitch_coeff_rig = experiment_in_wind_still_air_forces_removed.forces_global_center[:,4:12:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width**2/section_length_in_rig
        pitch_coeff_wall = experiment_in_wind_still_air_forces_removed.forces_global_center[:,16:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width**2/section_length_on_wall
        #4:24:6 - rotation dir

        pitch_motion = experiment_in_wind_still_air_forces_removed.motion[:,2]

        if upwind_in_rig == True:
            drag_coeff = np.concatenate((drag_coeff_rig,drag_coeff_wall),axis=1)
            lift_coeff = np.concatenate((lift_coeff_rig,lift_coeff_wall),axis=1)
            pitch_coeff = np.concatenate((pitch_coeff_rig,pitch_coeff_wall),axis=1)

        elif upwind_in_rig == False:
            drag_coeff = np.concatenate((drag_coeff_wall,drag_coeff_rig),axis=1)
            lift_coeff = np.concatenate((lift_coeff_wall,lift_coeff_rig),axis=1)
            pitch_coeff = np.concatenate((pitch_coeff_wall,pitch_coeff_rig),axis=1)
                
        return cls(drag_coeff,lift_coeff,pitch_coeff,pitch_motion,filtered_wind)
    
    #Plot limits
    ymin_drag = 0
    ymax_drag = 1

    ymin_lift = -0.75
    ymax_lift = 0.75

    ymin_pitch = -0.175
    ymax_pitch = 0.2
    
    def to_excel(self,section_name, sheet_name='Test' ,section_width=0,section_height=0,section_length_in_rig = 0, section_length_on_wall=0, upwind_in_rig=True):
        """
        
        Parameters
        ----------
        section_name : string
            section name.
            Distance: 1D, 2D, 3D, 4D, 5D. Follow same structure always.
        sheet_name : string 
            name of the excel sheet that the data is stored in.
            Options: MUS, MDS, Single.
        section_width : float64, optional
            Width of the section model. The default is 0.
        section_height : float64, optional
            Height of the section model. The default is 0.
        section_length_in rig : float64, optional
            Length of the section model in rig. The default is 0.
        section_length_on_wall : float
            Length of the section model on wall. The default is 0.
        upwind_in_rig : bool
            True dersom upwind-dekket er montert i rigg, False dersom downwind er i rigg.

        Returns
        -------
        None.

        """
        
        C_D_upwind = self.drag_coeff[0:-1:10,0] + self.drag_coeff[0:-1:10,1]
        C_D_downwind  =self.drag_coeff[0:-1:10,2] + self.drag_coeff[0:-1:10,3]
        C_L_upwind = self.lift_coeff[0:-1:10,0] + self.lift_coeff[0:-1:10,1]
        C_L_downwind = self.lift_coeff[0:-1:10,2] + self.lift_coeff[0:-1:10,3]
        C_M_upwind = self.pitch_coeff[0:-1:10,0] + self.pitch_coeff[0:-1:10,1]
        C_M_downwind = self.pitch_coeff[0:-1:10,2] + self.pitch_coeff[0:-1:10,3]

        # Create results dataframe

        static_coeff = pd.DataFrame({"pitch motion": self.pitch_motion[0:-1:10],
                                     "alpha [deg]": self.pitch_motion[0:-1:10] * 360 / (2 * np.pi),
                                "C_D_upwind": C_D_upwind,
                                "C_D_downwind": C_D_downwind,
                                "C_L_upwind": C_L_upwind,
                                "C_L_downwind": C_L_downwind,
                                "C_M_upwind": C_M_upwind,
                                "C_M_downwind": C_M_downwind,
                                 }).round(3)

        # Geometry/documentation DataFrame
        setUp = pd.DataFrame({
            "D": [section_height],
            "B": [section_width],
            "L in rig": [section_length_in_rig],
            "L on wall": [section_length_on_wall],
            "Upwind in rig": [upwind_in_rig]
        }).round(3)

        # Write to excel
        filename = os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Excel",
                    f"Static_coeff_{section_name}.xlsx")
        if os.path.exists(filename):# Add sheet/owerwrite sheet in excisting file
           with pd.ExcelWriter(filename,mode="a",engine="openpyxl",if_sheet_exists="replace") as writer:
               setUp.to_excel(writer, sheet_name)
               static_coeff.to_excel(writer, sheet_name=sheet_name)
        else: #create new excel file
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                setUp.to_excel(writer, sheet_name)
                static_coeff.to_excel(writer, sheet_name=sheet_name)
    
    def to_excel_mean(self,section_name, sheet_name='Test' ,section_width=0,section_height=0,section_length_in_rig = 0, section_length_on_wall=0, upwind_in_rig=True):
        """
        Export mean static coefficients and setup info to Excel.

        Parameters
        ----------
        section_name : string
            section name.
            Distance: 1D, 2D, 3D, 4D, 5D. Follow same structure always.
        sheet_name : string 
            name of the excel sheet that the data is stored in.
            Options: MUS, MDS, Single.
        section_width : float64, optional
            Width of the section model. The default is 0.
        section_height : float64, optional
            Height of the section model. The default is 0.
        section_length_in rig : float64, optional
            Length of the section model in rig. The default is 0.
        section_length_on_wall : float
            Length of the section model on wall. The default is 0.
        upwind_in_rig : bool
            True dersom upwind-dekket er montert i rigg, False dersom downwind er i rigg.

        Returns
        -------
        None.

        """
        alpha = np.round(self.pitch_motion * 360 / (2 * np.pi), 1)
        unique_alphas = np.unique(alpha)

        # Beregn middelverdi per alpha
        C_D_upwind = np.array([np.nanmean(self.drag_coeff[:,0][alpha == val]) + np.nanmean(self.drag_coeff[:,1][alpha == val]) for val in unique_alphas])
        C_D_downwind = np.array([np.nanmean(self.drag_coeff[:,2][alpha == val]) + np.nanmean(self.drag_coeff[:,3][alpha == val]) for val in unique_alphas])
        C_L_upwind = np.array([np.nanmean(self.lift_coeff[:,0][alpha == val]) + np.nanmean(self.lift_coeff[:,1][alpha == val]) for val in unique_alphas])
        C_L_downwind = np.array([np.nanmean(self.lift_coeff[:,2][alpha == val]) + np.nanmean(self.lift_coeff[:,3][alpha == val]) for val in unique_alphas])
        C_M_upwind = np.array([np.nanmean(self.pitch_coeff[:,0][alpha == val]) + np.nanmean(self.pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
        C_M_downwind = np.array([np.nanmean(self.pitch_coeff[:,2][alpha == val]) + np.nanmean(self.pitch_coeff[:,3][alpha == val]) for val in unique_alphas])

        # Create results dataframe
        static_coeff = pd.DataFrame({"alpha [deg]": unique_alphas,
                                "C_D_upwind": C_D_upwind,
                                "C_D_downwind": C_D_downwind,
                                "C_L_upwind": C_L_upwind,
                                "C_L_downwind": C_L_downwind,
                                "C_M_upwind": C_M_upwind,
                                "C_M_downwind": C_M_downwind,
                                 }).round(3)

        # Geometry/documentation DataFrame
        setUp = pd.DataFrame({
            "D": [section_height],
            "B": [section_width],
            "L in rig": [section_length_in_rig],
            "L on wall": [section_length_on_wall],
            "Upwind in rig": [upwind_in_rig]
        }).round(3)

        # Write to excel
        filename = os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Excel",
                    f"Static_coeff_{section_name}.xlsx")
        if os.path.exists(filename):# Add sheet/owerwrite sheet in excisting file
           with pd.ExcelWriter(filename,mode="a",engine="openpyxl",if_sheet_exists="replace") as writer:
               setUp.to_excel(writer, sheet_name)
               static_coeff.to_excel(writer, sheet_name=sheet_name)
        else: #create new excel file
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                setUp.to_excel(writer, sheet_name)
                static_coeff.to_excel(writer, sheet_name=sheet_name)             
            
    def plot_drag(self,mode="decks", upwind_in_rig=True, ax=None):
        """ plots the drag coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color1 = "#F15854"
            color2= "#990000"
        else: 
            color1 = "#006400"
            color2 ="#60BD68"
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if mode == "all": #individual load cells + total sum
            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
                # Sum all load cells for each time step            
            for k in range(self.drag_coeff.shape[1]):
                ax.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,k],label=("Load cell " + str(k+1)),alpha =0.5)
                # each load cell plottet separately
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)
        
        elif mode == "decks": #upwind and downwind deck + total sum
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
            ax.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,0]+self.drag_coeff[:,1],label=("Upwind deck"), color=color1)
            ax.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,2]+self.drag_coeff[:,3],label=("Downwind deck"), color=color2)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)

        elif mode == "total": #only total sum
            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)

        elif mode == "single": #single deck

            ax.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,0]+self.drag_coeff[:,1],label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)

        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
                 
    def plot_lift(self,mode="decks", upwind_in_rig=True, ax=None):
        """ plots the lift coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color1 = "#F15854"
            color2= "#990000"
        else: 
            color1 = "#006400"
            color2 ="#60BD68"
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))        
        if mode == "all":
            print("Lift coeff shape:", self.lift_coeff.shape)

            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                ax.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,k],label=("Load cell " + str(k+1)),alpha=0.5)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        elif mode == "decks":
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
            ax.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,0]+self.lift_coeff[:,1],label=("Upwind deck"), color=color1)
            ax.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,2]+self.lift_coeff[:,3],label=("Downwind deck"), color=color2)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        elif mode == "total":
            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)

        elif mode == "single": #single deck

            ax.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,0]+self.lift_coeff[:,1],label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
        
    def plot_pitch(self,mode="decks", upwind_in_rig=True, ax=None):
        """ plots the pitch coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color1 = "#F15854"
            color2= "#990000"
        else: 
            color1 = "#006400"
            color2 ="#60BD68"
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))        
        if mode == "all":
            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
                            
            for k in range(self.drag_coeff.shape[1]):
                ax.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,k],label=("Load cell " + str(k+1)),alpha=0.5)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        elif mode == "decks":
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
            ax.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,0]+self.pitch_coeff[:,1],label=("Upwind deck"), color=color1)
            ax.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,2]+self.pitch_coeff[:,3],label=("Downwind deck"), color=color2)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        elif mode == "total":
            ax.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        elif mode == "single": #single deck

            ax.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,0]+self.pitch_coeff[:,1],label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
    
    def plot_drag_mean(self,mode="total", upwind_in_rig=True, ax=None):
        """ plots the drag coefficient mean
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        
        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color = "#F15854"
            linestyle1 = "-"
            linestyle2 = "--"
        else: 
            color ="#60BD68"
            linestyle1 = "--"
            linestyle2 = "-"
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)      

        if mode == "all": #individual load cells + total sum

            cd_total = np.sum(self.drag_coeff,axis=1)
            cd_total_mean = np.array([np.nanmean(cd_total[alpha == val]) for val in unique_alphas])
            
            ax.plot(unique_alphas,cd_total_mean,label = "Total")
                      
            for k in range(self.drag_coeff.shape[1]):
                cd_k = self.drag_coeff[:,k]
                cd_k_mean = np.array([np.nanmean(cd_k[alpha == val]) for val in unique_alphas])
                ax.plot(unique_alphas,cd_k_mean,label=("Load cell " + str(k+1)),alpha =0.5)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)
        
        elif mode == "decks": #upwind and downwind deck + total sum
            cd_upwind_mean = np.array([
                np.nanmean(self.drag_coeff[:,0][alpha == val]) + np.nanmean(self.drag_coeff[:,1][alpha == val])
                for val in unique_alphas
            ])
            cd_downwind_mean = np.array([
                np.nanmean(self.drag_coeff[:,2][alpha == val]) + np.nanmean(self.drag_coeff[:,3][alpha == val])
                for val in unique_alphas
            ])
            ax.plot(unique_alphas,cd_upwind_mean,label=("Upwind deck"), color=color, linestyle = linestyle1, drawstyle='default') # Switch upwind and downwind deck. For Downstream files the load cells are switched.
            ax.plot(unique_alphas,cd_downwind_mean,label=("Downwind deck"), color=color, linestyle = linestyle2, drawstyle='default')
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)

        elif mode == "total": #only total sum
            cd_total_mean = np.array([np.nanmean(np.sum(self.drag_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            ax.plot(unique_alphas,cd_total_mean,label = "Total")    
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)
        
        elif mode == "single": #single deck

            cd_single_mean = np.array([
                np.nanmean(self.drag_coeff[:,0][alpha == val]) + np.nanmean(self.drag_coeff[:,1][alpha == val])
                for val in unique_alphas
            ])

            
            ax.plot(unique_alphas,cd_single_mean,label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_D(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_drag,ymax=self.ymax_drag)

        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
                 
    def plot_lift_mean(self,mode="total",upwind_in_rig=True, ax=None):
        """ plots the lift coefficient mean
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color = "#F15854"
            linestyle1 = "-"
            linestyle2 = "--"
        else: 
            color ="#60BD68"
            linestyle1 = "--"
            linestyle2 = "-" 
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6)) 
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)   

        if mode == "all":
            print("Lift coeff shape:", self.lift_coeff.shape)

            cl_total_mean = np.array([np.nanmean(np.sum(self.lift_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            ax.plot(unique_alphas,cl_total_mean,label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                cl_k = self.lift_coeff[:,k]
                cl_k_mean = np.array([np.nanmean(cl_k[alpha == val]) for val in unique_alphas])
                ax.plot(unique_alphas,cl_k_mean,label=("Load cell " + str(k+1)),alpha=0.5)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        elif mode == "decks":
            cl_upwind_mean = np.array([
                np.nanmean(self.lift_coeff[:,0][alpha == val]) + np.nanmean(self.lift_coeff[:,1][alpha == val])
                for val in unique_alphas
            ])
            cl_downwind_mean = np.array([
                np.nanmean(self.lift_coeff[:,2][alpha == val]) + np.nanmean(self.lift_coeff[:,3][alpha == val])
                for val in unique_alphas
            ])
            ax.plot(unique_alphas,cl_upwind_mean,label=("Upwind deck"), color = color, linestyle = linestyle1)
            ax.plot(unique_alphas,cl_downwind_mean,label=("Downwind deck"), color=color, linestyle = linestyle2)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        elif mode == "total":
            cl_total_mean = np.array([np.nanmean(np.sum(self.lift_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            ax.plot(unique_alphas,cl_total_mean)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        elif mode == "single": #single deck
            cl_single_mean = np.array([
                np.nanmean(self.lift_coeff[:,0][alpha == val]) + np.nanmean(self.lift_coeff[:,1][alpha == val])
                for val in unique_alphas
            ])
            
            ax.plot(unique_alphas,cl_single_mean,label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_L(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_lift,ymax=self.ymax_lift)
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
        
    def plot_pitch_mean(self,mode="total", upwind_in_rig=True, ax=None):
        """ plots the pitch coefficient mean 
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        upwind_in_rig: set up type
        """
        if upwind_in_rig:
            color = "#F15854"
            linestyle1 = "-"
            linestyle2 = "--"
        else: 
            color ="#60BD68"
            linestyle1 = "--"
            linestyle2 = "-" 
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)  
                
        if mode == "all":
            cm_total_mean = np.array([np.nanmean(np.sum(self.pitch_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            ax.plot(unique_alphas,cm_total_mean,label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                cm_k = self.pitch_coeff[:,k]
                cm_k_mean = np.array([np.nanmean(cm_k[alpha == val]) for val in unique_alphas])
                ax.plot(unique_alphas,cm_k_mean,label=("Load cell " + str(k+1)),alpha=0.5)
            
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        elif mode == "decks":
            cm_upwind_mean = np.array([
                np.nanmean(self.pitch_coeff[:,0][alpha == val]) + np.nanmean(self.pitch_coeff[:,1][alpha == val])
                for val in unique_alphas
            ])
            cm_downwind_mean = np.array([
                np.nanmean(self.pitch_coeff[:,2][alpha == val]) + np.nanmean(self.pitch_coeff[:,3][alpha == val])
                for val in unique_alphas
            ])
            ax.plot(unique_alphas,cm_upwind_mean,label=("Upwind deck"), color=color, linestyle = linestyle1)
            ax.plot(unique_alphas,cm_downwind_mean,label=("Downwind deck"), color=color, linestyle = linestyle2)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        elif mode == "total":
            cm_total_mean = np.array([np.nanmean(np.sum(self.pitch_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            ax.plot(unique_alphas,cm_total_mean)
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)

        elif mode == "single": #single deck

            cm_single_mean = np.array([
                            np.nanmean(self.pitch_coeff[:,0][alpha == val]) + np.nanmean(self.pitch_coeff[:,1][alpha == val])
                            for val in unique_alphas
                        ])

            
            ax.plot(unique_alphas,cm_single_mean,label=("Single deck"))
            ax.grid()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$C_M(\alpha)$")
            ax.legend()
            ax.set_ylim(ymin=self.ymin_pitch,ymax=self.ymax_pitch)
        
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )

    def save_mean_static_coeff(self, filename, path=".", mode="decks"):
        """
        Saves the static coefficients in a .npz file
        
        parameters:
        ----------
        filename : str
            name of the file
        path : str, optional
            path to the file. The default is ".".
        """

        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        if mode == "decks":
            cd_upwind_mean = np.array([np.nanmean(self.drag_coeff[:,0][alpha == val]) + np.nanmean(self.drag_coeff[:,1][alpha == val]) for val in unique_alphas])
            cd_downwind_mean = np.array([np.nanmean(self.drag_coeff[:,2][alpha == val]) + np.nanmean(self.drag_coeff[:,3][alpha == val]) for val in unique_alphas])
            cl_upwind_mean = np.array([np.nanmean(self.lift_coeff[:,0][alpha == val]) + np.nanmean(self.lift_coeff[:,1][alpha == val]) for val in unique_alphas])
            cl_downwind_mean = np.array([np.nanmean(self.lift_coeff[:,2][alpha == val]) + np.nanmean(self.lift_coeff[:,3][alpha == val]) for val in unique_alphas])
            cm_upwind_mean = np.array([np.nanmean(self.pitch_coeff[:,0][alpha == val]) + np.nanmean(self.pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
            cm_downwind_mean = np.array([np.nanmean(self.pitch_coeff[:,2][alpha == val]) + np.nanmean(self.pitch_coeff[:,3][alpha == val]) for val in unique_alphas])
            np.savez_compressed(filepath, alpha=unique_alphas, cd_in_rig=cd_upwind_mean, cd_on_wall=cd_downwind_mean, cl_in_rig=cl_upwind_mean, cl_on_wall=cl_downwind_mean, cm_in_rig=cm_upwind_mean, cm_on_wall=cm_downwind_mean)

        elif mode == "single":
            cd_upwind_mean = np.array([np.nanmean(self.drag_coeff[:,0][alpha == val]) + np.nanmean(self.drag_coeff[:,1][alpha == val]) for val in unique_alphas])
            cl_upwind_mean = np.array([np.nanmean(self.lift_coeff[:,0][alpha == val]) + np.nanmean(self.lift_coeff[:,1][alpha == val]) for val in unique_alphas])
            cm_upwind_mean = np.array([np.nanmean(self.pitch_coeff[:,0][alpha == val]) + np.nanmean(self.pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
            np.savez_compressed(filepath, alpha=unique_alphas, cd_in_rig=cd_upwind_mean, cl_in_rig=cl_upwind_mean, cm_in_rig=cm_upwind_mean)
    
    
def plot_compare_drag(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots drag coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    """
    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"

    }
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.drag_coeff[:,0] + static_coeff_single.drag_coeff[:,1], label=("Single deck"), color = colors["single"], linewidth = 2)
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.drag_coeff[:,0] + static_coeff_up.drag_coeff[:,1], label=("MUS: Upstream deck"), color = colors["mus"], linewidth = 2) #MUS er her riktig, altså motsatt av i excel arket.
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.drag_coeff[:,2] + static_coeff_up.drag_coeff[:,3], label=("MUS: Downstream deck"), color = colors["mus"],  alpha = 0.5, linewidth = 1.5)  
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.drag_coeff[:,2] + static_coeff_down.drag_coeff[:,3], label=("MDS: Downstream deck"), color = colors["mds"], linewidth = 2)
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.drag_coeff[:,0] + static_coeff_down.drag_coeff[:,1], label=("MDS: Upstream deck"), color = colors["mds"],alpha = 0.5, linewidth = 1.5)  
    
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_D(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    ax.legend()
    ax.set_title("Comparison of drag coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_drag,ymax=static_coeff_single.ymax_drag)


def plot_compare_lift(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots lift coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    
    """
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"
    }

    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.lift_coeff[:,0] + static_coeff_single.lift_coeff[:,1], label=("Single deck"), color = colors["single"], linewidth = 2)
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.lift_coeff[:,0] + static_coeff_up.lift_coeff[:,1], label=("MUS: Upstream deck"), color = colors["mus"], linewidth = 2)
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.lift_coeff[:,2] + static_coeff_up.lift_coeff[:,3], label=("MUS: Dowstream deck"), color = colors["mus"], alpha = 0.5, linewidth = 1.5)
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.lift_coeff[:,2] + static_coeff_down.lift_coeff[:,3], label=("MDS: Dowstream deck"), color = colors["mds"], linewidth = 2)
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.lift_coeff[:,0] + static_coeff_down.lift_coeff[:,1], label=("MDS: Upstream deck"), color = colors["mds"], alpha = 0.5, linewidth = 1.5)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_L(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    ax.legend()
    ax.set_title("Comparison of lift coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_lift,ymax=static_coeff_single.ymax_lift)

def plot_compare_pitch(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots pitch coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    """
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"
    }

    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.pitch_coeff[:,0] + static_coeff_single.pitch_coeff[:,1], label=("Single deck"), color = colors["single"], linewidth = 2)
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.pitch_coeff[:,0] + static_coeff_up.pitch_coeff[:,1], label=("MUS: Upstream deck"), color = colors["mus"], linewidth = 2)
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.pitch_coeff[:,0] + static_coeff_down.pitch_coeff[:,1], label=("MUS: Downstream deck"), color = colors["mus"],  alpha = 0.5, linewidth = 1.5)
    ax.plot(static_coeff_down.pitch_motion*360/2/np.pi, static_coeff_down.pitch_coeff[:,2] + static_coeff_down.pitch_coeff[:,3], label=("MDS: Downstream deck"), color = colors["mds"], linewidth = 2)
    ax.plot(static_coeff_up.pitch_motion*360/2/np.pi, static_coeff_up.pitch_coeff[:,2] + static_coeff_up.pitch_coeff[:,3], label=("MDS: Upstream deck"), color = colors["mds"],  alpha = 0.5, linewidth = 1.5)
  
    
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_M(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    ax.legend()
    ax.set_title("Comparison of pitch coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_pitch,ymax=static_coeff_single.ymax_pitch)

def plot_compare_drag_mean(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots drag mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    """
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"
    }
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha_up = np.round(static_coeff_up.pitch_motion*360/2/np.pi,1)
    unique_alphas_up = np.unique(alpha_up)
    alpha_down = np.round(static_coeff_down.pitch_motion*360/2/np.pi,1)
    unique_alphas_down = np.unique(alpha_down)

    cd_single_mean = np.array([np.nanmean(static_coeff_single.drag_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.drag_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cd_upwind_mean = np.array([np.nanmean(static_coeff_up.drag_coeff[:,0][alpha_up == val]) + np.nanmean(static_coeff_up.drag_coeff[:,1][alpha_up == val]) for val in unique_alphas_up])
    cd_downwind_mean = np.array([np.nanmean(static_coeff_down.drag_coeff[:,2][alpha_down == val]) + np.nanmean(static_coeff_down.drag_coeff[:,3][alpha_down == val]) for val in unique_alphas_down])
    cd_upDown_mean = np.array([np.nanmean(static_coeff_up.drag_coeff[:,2][alpha_up == val]) + np.nanmean(static_coeff_up.drag_coeff[:,3][alpha_up == val]) for val in unique_alphas_up])
    cd_downUp_mean = np.array([np.nanmean(static_coeff_down.drag_coeff[:,0][alpha_down == val]) + np.nanmean(static_coeff_down.drag_coeff[:,1][alpha_down == val]) for val in unique_alphas_down])
   


    ax.plot(unique_alphas_single, cd_single_mean, label="Single deck", color = colors["single"])
    ax.plot(unique_alphas_up, cd_upwind_mean, label="MUS: Upstream deck ", color = colors["mus"])
    ax.plot(unique_alphas_up, cd_upDown_mean, label="MUS: Downstream deck", color = colors["mus"], linestyle = "--")
    ax.plot(unique_alphas_down, cd_downwind_mean, label="MDS: Downstream deck", color = colors["mds"])
    ax.plot(unique_alphas_down, cd_downUp_mean, label="MDS: Upstream deck", color = colors["mds"], linestyle = "--")


   
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_D(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    ax.legend()
    ax.set_title("Comparison of mean drag coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_drag,ymax=static_coeff_single.ymax_drag)

def plot_compare_lift_mean(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots lift mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    
    """
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"
    }
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha_up = np.round(static_coeff_up.pitch_motion*360/2/np.pi,1)
    unique_alphas_up = np.unique(alpha_up)
    alpha_down = np.round(static_coeff_down.pitch_motion*360/2/np.pi,1)
    unique_alphas_down = np.unique(alpha_down)

    cl_single_mean = np.array([np.nanmean(static_coeff_single.lift_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.lift_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cl_upwind_mean = np.array([np.nanmean(static_coeff_up.lift_coeff[:,0][alpha_up == val]) + np.nanmean(static_coeff_up.lift_coeff[:,1][alpha_up == val]) for val in unique_alphas_up])
    cl_downwind_mean = np.array([np.nanmean(static_coeff_down.lift_coeff[:,2][alpha_down == val]) + np.nanmean(static_coeff_down.lift_coeff[:,3][alpha_down == val]) for val in unique_alphas_down])
    cl_upDown_mean = np.array([np.nanmean(static_coeff_up.lift_coeff[:,2][alpha_up == val]) + np.nanmean(static_coeff_up.lift_coeff[:,3][alpha_up == val]) for val in unique_alphas_up])
    cl_downUp_mean = np.array([np.nanmean(static_coeff_down.lift_coeff[:,0][alpha_down == val]) + np.nanmean(static_coeff_down.lift_coeff[:,1][alpha_down == val]) for val in unique_alphas_down])


    ax.plot(unique_alphas_single, cl_single_mean, label="Single deck", color = colors["single"])
    ax.plot(unique_alphas_up, cl_upwind_mean, label="MUS: Upstream deck", color = colors["mus"])
    ax.plot(unique_alphas_up, cl_upDown_mean, label="MUS: Downstream deck", color = colors["mus"], linestyle = "--")
    ax.plot(unique_alphas_down, cl_downwind_mean, label="MDS: Downstream deck", color = colors["mds"])
    ax.plot(unique_alphas_down, cl_downUp_mean, label="MDS: Upstream deck", color = colors["mds"], linestyle = "--")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_L(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    ax.legend()
    ax.set_title("Comparison of mean lift coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_lift,ymax=static_coeff_single.ymax_lift)

def plot_compare_pitch_mean(static_coeff_single, static_coeff_up, static_coeff_down, ax=None):
    """
    Plots pitch mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff_up : StaticCoeff object
        The StaticCoeff object for upwind deck.
    static_coeff_down : StaticCoeff object
        The StaticCoeff object for downwind deck.
    """
    # Calculate unique alpha values (pitch motion in degrees)
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha_up = np.round(static_coeff_up.pitch_motion*360/2/np.pi,1)
    unique_alphas_up = np.unique(alpha_up)
    alpha_down = np.round(static_coeff_down.pitch_motion*360/2/np.pi,1)
    unique_alphas_down = np.unique(alpha_down)

    cm_single_mean = np.array([np.nanmean(static_coeff_single.pitch_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cm_upwind_mean = np.array([np.nanmean(static_coeff_up.pitch_coeff[:,0][alpha_up == val]) + np.nanmean(static_coeff_up.pitch_coeff[:,1][alpha_up == val]) for val in unique_alphas_up])
    cm_downwind_mean = np.array([np.nanmean(static_coeff_down.pitch_coeff[:,2][alpha_down == val]) + np.nanmean(static_coeff_down.pitch_coeff[:,3][alpha_down == val]) for val in unique_alphas_down])
    cm_upDown_mean = np.array([np.nanmean(static_coeff_up.pitch_coeff[:,2][alpha_up == val]) + np.nanmean(static_coeff_up.pitch_coeff[:,3][alpha_up == val]) for val in unique_alphas_up])
    cm_downUp_mean = np.array([np.nanmean(static_coeff_down.pitch_coeff[:,0][alpha_down == val]) + np.nanmean(static_coeff_down.pitch_coeff[:,1][alpha_down == val]) for val in unique_alphas_down])

    colors = {
        "single": "#5DA5DA",
        "mus": "#F15854",
        "mds": "#60BD68"
    }

    ax.plot(unique_alphas_single, cm_single_mean, label="Single deck", color = colors["single"])
    ax.plot(unique_alphas_up, cm_upwind_mean, label="MUS: Upstream deck", color =colors["mus"])
    ax.plot(unique_alphas_up, cm_upDown_mean, label="MUS: Downstream deck", color = colors["mus"], linestyle = "--")

    ax.plot(unique_alphas_down, cm_downwind_mean, label="MDS: Downstream deck", color =colors["mds"])
    ax.plot(unique_alphas_down, cm_downUp_mean, label="MDS: Upstream deck", color = colors["mds"], linestyle = "--")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_M(\alpha)$")
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    ax.legend()
    ax.set_title("Comparison of mean pitch coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_pitch,ymax=static_coeff_single.ymax_pitch)

#%% Compare with single deck
def plot_compare_drag_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots drag coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS

    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color1 = "#F15854"
        color2= "#990000"
    else: 
        setUp_type = "MDS"
        color1 = "#006400"
        color2 ="#60BD68"
       
   
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))   


    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.drag_coeff[:,0] + static_coeff_single.drag_coeff[:,1], label=("Single deck"), color = "#5DA5DA")
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.drag_coeff[:,0] + static_coeff.drag_coeff[:,1], label=("Upstream deck"), color = color1)
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.drag_coeff[:,2] + static_coeff.drag_coeff[:,3], label=("Downstream deck"), color = color2)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_D(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of drag coefficients")
    ax.set_ylim(ymin=static_coeff_single.ymin_drag,ymax=static_coeff_single.ymax_drag)


def plot_compare_lift_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots lift coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS
    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color1 = "#F15854"
        color2= "#990000"
    else: 
        setUp_type = "MDS"
        color1 = "#006400"
        color2 ="#60BD68"
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.lift_coeff[:,0] + static_coeff_single.lift_coeff[:,1], label=("Single deck"), color = "#5DA5DA")
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.lift_coeff[:,0] + static_coeff.lift_coeff[:,1], label=("Upwind deck"), color = color1)
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.lift_coeff[:,2] + static_coeff.lift_coeff[:,3], label=("Downwind deck"), color = color2)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_L(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of lift coefficients ")
    ax.set_ylim(ymin=static_coeff_single.ymin_lift,ymax=static_coeff_single.ymax_lift)

def plot_compare_pitch_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots pitch coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS coeff
    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color1 = "#F15854"
        color2= "#990000"
    else: 
        setUp_type = "MDS"
        color1 = "#006400"
        color2 ="#60BD68"
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))   

    ax.plot(static_coeff_single.pitch_motion*360/2/np.pi, static_coeff_single.pitch_coeff[:,0] + static_coeff_single.pitch_coeff[:,1], label=("Single deck"), color = "#5DA5DA")
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.pitch_coeff[:,0] + static_coeff.pitch_coeff[:,1], label=("Upwind deck"), color = color1)
    ax.plot(static_coeff.pitch_motion*360/2/np.pi, static_coeff.pitch_coeff[:,2] + static_coeff.pitch_coeff[:,3], label=("Downwind deck"), color = color2)

    
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_M(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of pitch coefficients ")
    ax.set_ylim(ymin=static_coeff_single.ymin_pitch,ymax=static_coeff_single.ymax_pitch)

def plot_compare_drag_mean_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots drag mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS
    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color = "#F15854"
    else: 
        setUp_type = "MDS"
        color ="#60BD68"
   
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha = np.round(static_coeff.pitch_motion*360/2/np.pi,1)
    unique_alphas = np.unique(alpha)

    cd_single_mean = np.array([np.nanmean(static_coeff_single.drag_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.drag_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cd_upwind_mean = np.array([np.nanmean(static_coeff.drag_coeff[:,0][alpha == val]) + np.nanmean(static_coeff.drag_coeff[:,1][alpha == val]) for val in unique_alphas])
    cd_downwind_mean = np.array([np.nanmean(static_coeff.drag_coeff[:,2][alpha == val]) + np.nanmean(static_coeff.drag_coeff[:,3][alpha== val]) for val in unique_alphas])


    
    ax.plot(unique_alphas_single, cd_single_mean, label="Single deck", color = "#5DA5DA")
    ax.plot(unique_alphas, cd_upwind_mean, label="Upwind deck", color = color)
    ax.plot(unique_alphas, cd_downwind_mean, label="Downwind deck", color = color, linestyle = "--")


   
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_D(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of mean drag coefficients ")
    ax.set_ylim(ymin=static_coeff_single.ymin_drag,ymax=static_coeff_single.ymax_drag)

def plot_compare_lift_mean_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots lift mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS
    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color = "#F15854"
    else: 
        setUp_type = "MDS"
        color ="#60BD68"
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha = np.round(static_coeff.pitch_motion*360/2/np.pi,1)
    unique_alphas = np.unique(alpha)

    cl_single_mean = np.array([np.nanmean(static_coeff_single.lift_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.lift_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cl_upwind_mean = np.array([np.nanmean(static_coeff.lift_coeff[:,0][alpha == val]) + np.nanmean(static_coeff.lift_coeff[:,1][alpha == val]) for val in unique_alphas])
    cl_downwind_mean = np.array([np.nanmean(static_coeff.lift_coeff[:,2][alpha == val]) + np.nanmean(static_coeff.lift_coeff[:,3][alpha == val]) for val in unique_alphas])
   

    ax.plot(unique_alphas_single, cl_single_mean, label="Single deck", color = "#5DA5DA")
    ax.plot(unique_alphas, cl_upwind_mean, label="Upwind deck", color = color)
    ax.plot(unique_alphas, cl_downwind_mean, label="Downwind deck", color = color, linestyle = "--")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_L(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of mean lift coefficients ")
    ax.set_ylim(ymin=static_coeff_single.ymin_lift,ymax=static_coeff_single.ymax_lift)

def plot_compare_pitch_mean_only_single(static_coeff_single, static_coeff, upwind_in_rig=True, ax=None):
    """
    Plots pitch mean coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_single : StaticCoeff object
        The StaticCoeff object for single deck.
    static_coeff : MUS /MDS
    upwind_in_rig: set up type
        """
    if upwind_in_rig:
        setUp_type = "MUS"
        color = "#F15854"
    else: 
        setUp_type = "MDS"
        color ="#60BD68"
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.unique(alpha_single)
    alpha = np.round(static_coeff.pitch_motion*360/2/np.pi,1)
    unique_alphas = np.unique(alpha)

    cm_single_mean = np.array([np.nanmean(static_coeff_single.pitch_coeff[:,0][alpha_single == val]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][alpha_single == val]) for val in unique_alphas_single])
    cm_upwind_mean = np.array([np.nanmean(static_coeff.pitch_coeff[:,0][alpha == val]) + np.nanmean(static_coeff.pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
    cm_downwind_mean = np.array([np.nanmean(static_coeff.pitch_coeff[:,2][alpha == val]) + np.nanmean(static_coeff.pitch_coeff[:,3][alpha == val]) for val in unique_alphas])
    

    ax.plot(unique_alphas_single, cm_single_mean, label="Single deck", color = "#5DA5DA")
    ax.plot(unique_alphas, cm_upwind_mean, label="Upwind deck", color = color)
    ax.plot(unique_alphas, cm_downwind_mean, label="Downwind deck", color = color, linestyle = "--")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$C_M(\alpha)$")
    ax.grid()
    ax.legend()
    ax.set_title(f"{setUp_type}: Comparison of mean pitch coefficients ")
    ax.set_ylim(ymin=static_coeff_single.ymin_pitch,ymax=static_coeff_single.ymax_pitch)

#%% Wind speeds
def plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_med,
                                   static_coeff_single_high, static_coeff_low,
                                   static_coeff_med, static_coeff_high,
                                   scoff = "", ax=None):
    """
    Plots  coefficients for low and high wind speeds in same figure for each deck setup.

    Parameters
    ----------
    static_coeff_single_low : StaticCoeff
        Single deck, low wind speed
    static_coeff_single_high : StaticCoeff
        Single deck, high wind speed
    static_coeff_up_low : StaticCoeff
        Deck in rig (upwind or downwind), low wind speed
    static_coeff_up_high : StaticCoeff
        Same deck in rig, high wind speed
    static_coeff_down_low : StaticCoeff
        Deck on wall (opposite of upwind_in_rig), low wind speed
    static_coeff_down_high : StaticCoeff
        Deck on wall, high wind speed

    scoff : str
        "drag" or "ift" or "pitch"

    """
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    color1HWS = "#B22222"
    color2HWS= "#F15854"
    color3HWS="#FCA5A5"

    color1MWS = "#A6CEE3"
    color2MWS= "#5DA5DA"
    color3MWS="#1F4E79"     

    color1LWS = "#A1D99B"
    color2LWS ="#60BD68"
    color3LWS="#238B45"
    
    if scoff == "drag":
        axis = r"$C_D(\alpha)$"
        coeff = "drag_coeff"
        min = 0
        max = 1
    elif scoff == "lift":
        axis = r"$C_L(\alpha)$"
        coeff = "lift_coeff"
        min = -0.75
        max = 0.75
    elif scoff == "pitch":
        axis = r"$C_M(\alpha)$"
        coeff = "pitch_coeff"
        min = -0.175
        max = 0.2


    # Plot low wind speed
   
    ax.plot(static_coeff_single_low.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_single_low, coeff)[:,0] + getattr(static_coeff_single_low, coeff)[:,1],
             label=f"LWS - Single deck", color = color1LWS)
    ax.plot(static_coeff_low.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_low, coeff)[:,0] + getattr(static_coeff_low, coeff)[:,1],
             label=f"LWS -  Upstream deck", color = color2LWS)
    ax.plot(static_coeff_low.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_low, coeff)[:,2] + getattr(static_coeff_low, coeff)[:,3],
             label=f"LWS - Downstream deck", color = color3LWS)

    # Plot med wind speed
    #plt.plot(static_coeff_med.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_med, coeff)[:,0] + getattr(static_coeff_med, coeff)[:,1],
    #         label=f"MWS - Upstream deck", color = color2MWS)
    #plt.plot(static_coeff_med.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_med, coeff)[:,2] + getattr(static_coeff_med, coeff)[:,3],
    #         label=f"MWS - Downstream deck", color = color3MWS)

    # Plot high wind speed
    
    ax.plot(static_coeff_single_high.pitch_motion * 360 / (2 * np.pi),getattr(static_coeff_single_high, coeff)[:,0] + getattr(static_coeff_single_high, coeff)[:,1],
             label=f"HWS - Single deck", color = color1HWS)
    ax.plot(static_coeff_high.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_high, coeff)[:,0] + getattr(static_coeff_high, coeff)[:,1],
             label=f"HWS - Upstream deck", color = color2HWS)
    ax.plot(static_coeff_high.pitch_motion * 360 / (2 * np.pi), getattr(static_coeff_high, coeff)[:,2] + getattr(static_coeff_high, coeff)[:,3],
             label=f"HWS - Downstream deck", color = color3HWS)

    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(axis)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(min,max)
    ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

 

#%% Wind speeds
def plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_med,
                                   static_coeff_single_high, static_coeff_low,
                                   static_coeff_med, static_coeff_high,
                                    scoff = "", ax=None):
    """
    Plots  coefficients for low and high wind speeds in same figure for each deck setup.

    Parameters
    ----------
    static_coeff_single_low : StaticCoeff
        Single deck, low wind speed
    static_coeff_single_high : StaticCoeff
        Single deck, high wind speed
    static_coeff_up_low : StaticCoeff
        Deck in rig (upwind or downwind), low wind speed
    static_coeff_up_high : StaticCoeff
        Same deck in rig, high wind speed
    static_coeff_down_low : StaticCoeff
        Deck on wall (opposite of upwind_in_rig), low wind speed
    static_coeff_down_high : StaticCoeff
        Deck on wall, high wind speed
    
    scoff : str
        "drag" or "ift" or "pitch"

    """

    color1HWS = "#B22222"
    color2HWS= "#F15854"
    color3HWS="#FCA5A5"

    color1MWS = "#A6CEE3"
    color2MWS= "#5DA5DA"
    color3MWS="#1F4E79"    

    color1LWS = "#A1D99B"
    color2LWS ="#60BD68"
    color3LWS="#238B45"
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    if scoff == "drag":
        axis = r"$C_D(\alpha)$"
        coeff = "drag_coeff"
        min = 0
        max = 1
    elif scoff == "lift":
        axis = r"$C_L(\alpha)$"
        coeff = "lift_coeff"
        min = -0.75
        max = 0.75
    elif scoff == "pitch":
        axis = r"$C_M(\alpha)$"
        coeff = "pitch_coeff"
        min = -0.175
        max = 0.2

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_single_low = np.round(static_coeff_single_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_single_low = np.unique(alpha_single_low)
    alpha_single_med = np.round(static_coeff_single_med.pitch_motion*360/2/np.pi,1)
    unique_alphas_single_med = np.unique(alpha_single_med)
    alpha_single_high = np.round(static_coeff_single_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_single_high = np.unique(alpha_single_high)

    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
    unique_alphas_med = np.unique(alpha_med)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    single_mean_low = np.array([np.nanmean(getattr(static_coeff_single_low, coeff)[:,0][alpha_single_low == val]) + np.nanmean(getattr(static_coeff_single_low, coeff)[:,1][alpha_single_low == val]) for val in unique_alphas_single_low])
    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    single_mean_med = np.array([np.nanmean(getattr(static_coeff_single_med, coeff)[:,0][alpha_single_med == val]) + np.nanmean(getattr(static_coeff_single_med, coeff)[:,1][alpha_single_med == val]) for val in unique_alphas_single_med])
    upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
    downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])

    single_mean_high = np.array([np.nanmean(getattr(static_coeff_single_high, coeff)[:,0][alpha_single_high == val]) + np.nanmean(getattr(static_coeff_single_high, coeff)[:,1][alpha_single_high == val]) for val in unique_alphas_single_high])
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])



    # Plot low wind speed
    ax.plot(unique_alphas_single_low, single_mean_low,
             label=f"LWS - Single deck", color = color2LWS,linestyle=':')
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"LWS -  Upstream deck", color = color2LWS, linestyle='--')
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"LWS - Downstream deck", color = color2LWS, linestyle='-')


    # Plot med wind speed
  
    #plt.plot(unique_alphas_med, upwind_mean_med,
    #            label=f"MWS - Upstream deck", color = color2MWS, linestyle='--')
    #plt.plot(unique_alphas_med, downwind_mean_med,
    #            label=f"MWS - Downstream deck", color = color2MWS, linestyle='-')
    
    # Plot high wind speed
    ax.plot(unique_alphas_single_high, single_mean_high,
             label=f"HWS - Single deck", color = color2HWS,linestyle=':')
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"HWS - Upstream deck", color = color2HWS, linestyle='--')
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"HWS - Downstream deck", color = color2HWS, linestyle='-')
    
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(axis)
    ax.grid(True)
    ax.set_ylim(min,max)
    ax.legend()
    ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

 
def filter(static_coeff, threshold=0.3, scoff="", single=True):
    """
    Filters coefficient values at each alpha where the spread exceeds a given threshold.
    Sets outlier values to NaN to ensure cleaner plots and prevent invalid line connections.

    Parameters
    ----------
    static_coeff : StaticCoeff
        The static coefficient object containing aerodynamic data.
    threshold : float, optional
        Maximum allowed spread (max - min) at each alpha value. Defaults to 0.3.
    scoff : str, optional
        Type of coefficient to filter: 'drag', 'lift', or 'pitch'.
    single : bool, optional
        If True, only filters the upwind (single-deck) coefficient. If False, filters both upwind and downwind.

    Returns
    -------
    alpha : ndarray
        Rounded alpha values [deg].
    coeff_up_plot : ndarray
        Filtered upwind (or single-deck) coefficients with NaNs replacing outliers.
    coeff_down_plot : ndarray (only if single=False)
        Filtered downwind coefficients with NaNs replacing outliers.
    """
    if scoff == "drag":
        coeff_up = static_coeff.drag_coeff[:, 0] + static_coeff.drag_coeff[:, 1]
        coeff_down = static_coeff.drag_coeff[:, 2] + static_coeff.drag_coeff[:, 3]
    elif scoff == "lift":
        coeff_up = static_coeff.lift_coeff[:, 0] + static_coeff.lift_coeff[:, 1]
        coeff_down = static_coeff.lift_coeff[:, 2] + static_coeff.lift_coeff[:, 3]
    elif scoff == "pitch":
        coeff_up = static_coeff.pitch_coeff[:, 0] + static_coeff.pitch_coeff[:, 1]
        coeff_down = static_coeff.pitch_coeff[:, 2] + static_coeff.pitch_coeff[:, 3]
    else:
        raise ValueError("Invalid 'scoff' argument. Must be 'drag', 'lift', or 'pitch'.")

    alpha = np.round(static_coeff.pitch_motion * 360 / (2 * np.pi), 1)

    coeff_up_plot = coeff_up.copy()
    coeff_down_plot = coeff_down.copy()
    unique_alphas = np.unique(alpha)

    if single:
        # Filter upwind coefficients only
        for val in unique_alphas:
            idx = np.where(alpha == val)[0]
            spread = np.max(coeff_up[idx]) - np.min(coeff_up[idx])
            if spread > threshold:
                coeff_up_plot[idx] = np.nan
        return alpha, coeff_up_plot

    # Filter both upwind and downwind coefficients
    for val in unique_alphas:
        idx = np.where(alpha == val)[0]
        spread_up = np.max(coeff_up[idx]) - np.min(coeff_up[idx])
        spread_down = np.max(coeff_down[idx]) - np.min(coeff_down[idx])

        if spread_up > threshold:
            coeff_up_plot[idx] = np.nan
        if spread_down > threshold:
            coeff_down_plot[idx] = np.nan

    return alpha, coeff_up_plot, coeff_down_plot


def get_coeffs(static_coeff):
    alpha = np.round(static_coeff.pitch_motion * 360 / (2 * np.pi), 1)
    return alpha, static_coeff.drag_coeff.copy(), static_coeff.lift_coeff.copy(), static_coeff.pitch_coeff.copy()
     
def remove_after_jump(alpha_vals, coeff_array, threshold_jump=0.2, cols=(0, 1)):
    alpha_vals_rounded = np.round(alpha_vals, 1)
    unique_alpha = np.sort(np.unique(alpha_vals_rounded)) # Lager en sortert liste av unike α-verdier (avrundet til én desimal).

    for i, alpha in enumerate(unique_alpha): # fra lav til høy alpha
        idx = np.where(alpha_vals_rounded == alpha)[0]
        if len(idx) == 0:
            continue

        current_vals = coeff_array[idx, cols[0]] + coeff_array[idx, cols[1]] #upwind or downwind
        current_mean = np.nanmean(current_vals) #Tar mean i tilfelle ikke begge cellene har verdier for akkurat denne alpha verdien
        if np.isnan(current_mean):
            continue

        p = 1
        while i - p >= 0: #Går bakover i α-verdiene til den finner siste gyldige (ikke-NaN) datapunkt.
            prev_idx = np.where(alpha_vals_rounded == unique_alpha[i - p])[0]
            if len(prev_idx) == 0:
                p += 1
                continue
            prev_vals = coeff_array[prev_idx, cols[0]] + coeff_array[prev_idx, cols[1]]
            prev_mean = np.nanmean(prev_vals)
            if not np.isnan(prev_mean):
                break
            p += 1
        else:
            continue

        if np.abs(current_mean - prev_mean) > threshold_jump: #et hopp er oppdaget
            mask = alpha_vals_rounded > alpha # Fjerner alle verdier med høyere α enn der hoppet oppstod
            coeff_array[mask, cols[0]] = np.nan
            coeff_array[mask, cols[1]] = np.nan
            break # tar kun første hopp (dette er greit etter å ha studert datasettenes oppførsel)
    return coeff_array

def filter_by_reference(static_coeff_1, static_coeff_2, static_coeff_3=None, threshold=0.1, threshold_low=[0.05, 0.05, 0.05], threshold_med=[None, None, None], threshold_high=[0.05, 0.05, 0.05], single=False):
    """
    Filters drag, lift, and pitch coefficients in each dataset where values deviate too much from reference at a given alpha.
    Reference is chosen based on dataset with lowest spread per alpha.
    If single=True, filters only static_coeff_1 and static_coeff_2.
    Returns filtered StaticCoeff objects.
    """
    # Antall trinn bakover du vil sjekke (i grader)
    alpha_lookback_range = 2  # f.eks. 5 trinn på 0.1
    alpha_step = 0.1
    alpha_tolerance = 1e-6  # for å unngå float-problemer


 
    alpha_1, drag_1, lift_1, pitch_1 = get_coeffs(static_coeff_1)
    #ettersom alpha er rundet opp til 1 desimal, er alle alphaer til hver tidsserie like, og man trenger egt ikke skille mellom alphaene i forsøket. Men koden blir litt mer robust.
    alpha_2, drag_2, lift_2, pitch_2 = get_coeffs(static_coeff_2)
 
    drag_1_filt, lift_1_filt, pitch_1_filt = drag_1.copy(), lift_1.copy(), pitch_1.copy()
    drag_2_filt, lift_2_filt, pitch_2_filt = drag_2.copy(), lift_2.copy(), pitch_2.copy()
 
    if not single:
        alpha_3, drag_3, lift_3, pitch_3 = get_coeffs(static_coeff_3)
        drag_3_filt, lift_3_filt, pitch_3_filt = drag_3.copy(), lift_3.copy(), pitch_3.copy()
 
    coeff_names = ["drag", "lift", "pitch"]
    coeffs_1 = [drag_1, lift_1, pitch_1]
    coeffs_2 = [drag_2, lift_2, pitch_2]
    coeffs_1_filt = [drag_1_filt, lift_1_filt, pitch_1_filt] # I første omgang kun en kopi
    coeffs_2_filt = [drag_2_filt, lift_2_filt, pitch_2_filt]
 
    if not single:
        coeffs_3 = [drag_3, lift_3, pitch_3]
        coeffs_3_filt = [drag_3_filt, lift_3_filt, pitch_3_filt]
 
    unique_alpha = np.unique(alpha_1)
 
    for val in unique_alpha:
        idx1 = np.where(alpha_1 == val)[0]
        idx2 = np.where(alpha_2 == val)[0]
 
        if single:
            if not (len(idx1) and len(idx2)):
                continue
 
            for i, name in enumerate(coeff_names):
                this_threshold_low  = threshold_low[i] if threshold_low[i] is not None else threshold
                this_threshold_high = threshold_high[i] if threshold_high[i] is not None else threshold
 
                coeff_1 = coeffs_1[i]
                coeff_2 = coeffs_2[i]
                coeff_1_f = coeffs_1_filt[i]
                coeff_2_f = coeffs_2_filt[i]
 
                vals_1 = coeff_1[idx1, 0] + coeff_1[idx1, 1] #drag eller lift eller pitch totalen for ett helt brudekke
                vals_2 = coeff_2[idx2, 0] + coeff_2[idx2, 1] #samme for et annet forsøk
                spread_1 = np.max(vals_1) - np.min(vals_1)
                spread_2 = np.max(vals_2) - np.min(vals_2)
 
                coeff_check1 = filter(static_coeff_1, this_threshold_low, scoff=name, single=True)[1]
                coeff_check2 = filter(static_coeff_2, this_threshold_high, scoff=name, single=True)[1]
 
                nan_flags = []

                for check_array, idx_array, alpha_array in zip([coeff_check1, coeff_check2], [idx1, idx2], [alpha_1, alpha_2]):
                    has_nan_now = np.any(np.isnan(check_array[idx_array]))

                    if has_nan_now:
                        nan_flags.append(True)
                        continue

                    has_nan_back = False
                    current_alpha = np.round(val, 1)

                    for delta in np.arange(alpha_step, alpha_lookback_range + alpha_step, alpha_step):
                        prev_alpha = np.round(current_alpha - delta, 1)
                        idx_prev = np.where(np.abs(np.round(alpha_array, 1) - prev_alpha) < alpha_tolerance)[0]
                        if len(idx_prev) > 0 and np.any(np.isnan(check_array[idx_prev])):
                            has_nan_back = True
                            break

                    nan_flags.append(has_nan_back)

                has_nan_1, has_nan_2 = nan_flags
 
                if has_nan_1 and not has_nan_2:
                    ref_mean = np.mean(vals_2)
                elif has_nan_2 and not has_nan_1:
                    ref_mean = np.mean(vals_1)
                else:
                    ref_mean = np.mean(vals_1) if spread_1 <= spread_2 else np.mean(vals_2)
 
                for idx, coeff_array in zip([idx1, idx2], [coeff_1_f, coeff_2_f]):
                    summed = coeff_array[idx, 0] + coeff_array[idx, 1] #når singel, kun ett enkelt brudekke (val_1 og val_2)
                    mask = np.abs(summed - ref_mean) > threshold #markerer verdier som er for langt unna referansen
                    coeff_array[idx[mask], 0] = np.nan # setter verdien til nan dersom mask er true
                    coeff_array[idx[mask], 1] = np.nan
 
        else:
            idx3 = np.where(alpha_3 == val)[0]
            if not (len(idx1) and len(idx2) and len(idx3)):
                continue
 
            for i, name in enumerate(coeff_names):
                this_threshold_low  = threshold_low[i] if threshold_low[i] is not None else threshold
                this_threshold_med  = threshold_med[i] if threshold_med[i] is not None else threshold
                this_threshold_high = threshold_high[i] if threshold_high[i] is not None else threshold
 
                coeff_1 = coeffs_1[i]
                coeff_2 = coeffs_2[i]
                coeff_3 = coeffs_3[i]
                coeff_1_f = coeffs_1_filt[i]
                coeff_2_f = coeffs_2_filt[i]
                coeff_3_f = coeffs_3_filt[i]
 
                #UP
                vals_up = [coeff_1[idx1, 0] + coeff_1[idx1, 1],
                           coeff_2[idx2, 0] + coeff_2[idx2, 1],
                           coeff_3[idx3, 0] + coeff_3[idx3, 1]]
                #print(f"Upwind vals: {[np.mean(v) for v in vals_up]}")
                spreads_up = [np.max(v) - np.min(v) for v in vals_up]
 
 
                coeff_up_checks = [
                    filter(static_coeff_1, this_threshold_low, scoff=name, single=False)[1],
                    filter(static_coeff_2, this_threshold_med, scoff=name, single=False)[1],
                    filter(static_coeff_3, this_threshold_high, scoff=name, single=False)[1],
                ]
 
                nan_flags_up = []

                for check_array, idx_array, alpha_array in zip(coeff_up_checks, [idx1, idx2, idx3], [alpha_1, alpha_2, alpha_3]):
                    has_nan_now = np.any(np.isnan(check_array[idx_array]))  #Sjekker om det er for stor spredning i en av de tre forsøkene. Hvis spredningen er for stor er det et dårlig signal med denne vinkelen.

                    if has_nan_now:
                        nan_flags_up.append(True)
                        continue  # trenger ikke sjekke historikken hvis det allerede er dårlig nå

                    # --- Sjekk bakover i alpha ---
                    has_nan_back = False
                    current_alpha = np.round(val, 1)  # val = current alpha value in loop

                    for delta in np.arange(alpha_step, alpha_lookback_range + alpha_step, alpha_step):
                        prev_alpha = np.round(current_alpha - delta, 1)
                        idx_prev = np.where(np.abs(np.round(alpha_array, 1) - prev_alpha) < alpha_tolerance)[0]
                        if len(idx_prev) > 0:
                            if np.any(np.isnan(check_array[idx_prev])):
                                has_nan_back = True
                                break  # vi trenger bare én dårlig α bakover

                    nan_flags_up.append(has_nan_now or has_nan_back)

                # Liste med de datasett som er “clean” nå og tidligere
                clean_idxs_up = [i for i, nan in enumerate(nan_flags_up) if not nan]  #liste med index til de som ikke har nan (ok spredning i data), nå og litt tidligere

               
 
                if len(clean_idxs_up) == 1: #velger datasettet med minst spredning eller ingen nan
                    ref_idx_up = clean_idxs_up[0] #eneste signal uten for mye spredning 
                elif len(clean_idxs_up) == 2: # to signaler uten for mye spredning, lavest spredning best
                    ref_idx_up = clean_idxs_up[0] if spreads_up[clean_idxs_up[0]] < spreads_up[clean_idxs_up[1]] else clean_idxs_up[1]
                elif len(clean_idxs_up) == 3: # alle signaler ok
                   ref_idx_up = np.argmin(spreads_up) 
                else: #Alle datasett er for dårlig
                    ref_idx_up = None
                    #print("alle datasett er for dårlig")
                    for idx, coeff_array in zip([idx1, idx2, idx3], [coeff_1_f, coeff_2_f, coeff_3_f]):
                        coeff_array[idx, 0] = np.nan
                        coeff_array[idx, 1] = np.nan
                

                if ref_idx_up is not None:
                    ref_mean_up = np.mean(vals_up[ref_idx_up])
    
                    #print(f"Ref mean up = {ref_mean_up:.3f}")

                    for idx, coeff_array in zip([idx1, idx2, idx3], [coeff_1_f, coeff_2_f, coeff_3_f]):
                        summed = coeff_array[idx, 0] + coeff_array[idx, 1]
                        mask = np.abs(summed - ref_mean_up) > threshold
                        #print(f"Deck {i+1} — alpha = {val:.1f} — Removed {np.sum(mask)} points due to threshold filtering.")

                        coeff_array[idx[mask], 0] = np.nan
                        coeff_array[idx[mask], 1] = np.nan
    
                    
                # Same logic for downwind
                vals_down = [coeff_1[idx1, 2] + coeff_1[idx1, 3],
                             coeff_2[idx2, 2] + coeff_2[idx2, 3],
                             coeff_3[idx3, 2] + coeff_3[idx3, 3]]
                spreads_down = [np.max(v) - np.min(v) for v in vals_down]
 
                coeff_down_checks = [
                    filter(static_coeff_1, this_threshold_low, scoff=name, single=False)[2],
                    filter(static_coeff_2, this_threshold_med, scoff=name, single=False)[2],
                    filter(static_coeff_3, this_threshold_high, scoff=name, single=False)[2],
                ]
 
                nan_flags_down = []

                for check_array, idx_array, alpha_array in zip(coeff_down_checks, [idx1, idx2, idx3], [alpha_1, alpha_2, alpha_3]):
                    has_nan_now = np.any(np.isnan(check_array[idx_array]))  #Sjekker om det er for stor spredning i en av de tre forsøkene. Hvis spredningen er for stor er det et dårlig signal med denne vinkelen.

                    if has_nan_now:
                        nan_flags_down.append(True)
                        continue  # trenger ikke sjekke historikken hvis det allerede er dårlig nå

                    # --- Sjekk bakover i alpha ---
                    has_nan_back = False
                    current_alpha = np.round(val, 1)  # val = current alpha value in loop

                    for delta in np.arange(alpha_step, alpha_lookback_range + alpha_step, alpha_step):
                        prev_alpha = np.round(current_alpha - delta, 1)
                        idx_prev = np.where(np.abs(np.round(alpha_array, 1) - prev_alpha) < alpha_tolerance)[0]
                        if len(idx_prev) > 0:
                            if np.any(np.isnan(check_array[idx_prev])):
                                has_nan_back = True
                                break  # vi trenger bare én dårlig α bakover

                    nan_flags_down.append(has_nan_now or has_nan_back)

                # Liste med de datasett som er “clean” nå og tidligere
                clean_idxs_down = [i for i, nan in enumerate(nan_flags_down) if not nan]  #liste med index til de som ikke har nan (ok spredning i data), nå og litt tidligere

 
                if len(clean_idxs_down) == 1:
                    ref_idx_down = clean_idxs_down[0]
                elif len(clean_idxs_down) == 2:
                    ref_idx_down = clean_idxs_down[0] if spreads_down[clean_idxs_down[0]] < spreads_down[clean_idxs_down[1]] else clean_idxs_down[1]
                elif len(clean_idxs_down) == 3: # alle signaler ok
                   ref_idx_down = np.argmin(spreads_down)
                else:
                    ref_idx_down = None
                    for idx, coeff_array in zip([idx1, idx2, idx3], [coeff_1_f, coeff_2_f, coeff_3_f]):
                       coeff_array[idx, 2] = np.nan
                       coeff_array[idx, 3] = np.nan
                 
                if ref_idx_down is not None:
 
                    ref_mean_down = np.mean(vals_down[ref_idx_down])
    
                    for idx, coeff_array in zip([idx1, idx2, idx3], [coeff_1_f, coeff_2_f, coeff_3_f]):
                        summed = coeff_array[idx, 2] + coeff_array[idx, 3]
                        mask = np.abs(summed - ref_mean_down) > threshold
                        coeff_array[idx[mask], 2] = np.nan
                        coeff_array[idx[mask], 3] = np.nan

    for i, alpha in enumerate([alpha_1, alpha_2] if single else [alpha_1, alpha_2, alpha_3]):
        target_arrays = [coeffs_1_filt[i], coeffs_2_filt[i]] if single else [coeffs_1_filt[i], coeffs_2_filt[i], coeffs_3_filt[i]]
        for coeff_array in target_arrays:
            remove_after_jump(alpha, coeff_array, threshold_jump=1.2, cols=(0, 1))
            if not single:
                remove_after_jump(alpha, coeff_array, threshold_jump=1.2, cols=(2, 3))

    #samler sammen alt etter filtreringer
    static_coeff_1_f = copy.deepcopy(static_coeff_1)
    static_coeff_2_f = copy.deepcopy(static_coeff_2)
    for name, data in zip(coeff_names, [drag_1_filt, lift_1_filt, pitch_1_filt]): #legger til nan verdier
        setattr(static_coeff_1_f, f"{name}_coeff", data)
    for name, data in zip(coeff_names, [drag_2_filt, lift_2_filt, pitch_2_filt]):
        setattr(static_coeff_2_f, f"{name}_coeff", data)
 
    if single:
        return static_coeff_1_f, static_coeff_2_f
 
    static_coeff_3_f = copy.deepcopy(static_coeff_3)
    for name, data in zip(coeff_names, [drag_3_filt, lift_3_filt, pitch_3_filt]):
        setattr(static_coeff_3_f, f"{name}_coeff", data)
 
    return static_coeff_1_f, static_coeff_2_f, static_coeff_3_f



#####################################################################33


def plot_static_coeff_filtered_out_above_threshold(alpha,coeff_up_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.3, scoff="", ax=None):

    if upwind_in_rig:
        color1 = "#F15854"
        color2= "#990000"
    else:
        color1 = "#006400"
        color2 ="#60BD68"

    if scoff == "drag":
        ylabel = r"$C_D(\alpha)$"
        min = 0
        max = 1
    elif scoff == "lift":
        ylabel = r"$C_L(\alpha)$"
        min = -0.75
        max = 0.75
    elif scoff == "pitch":
        ylabel = r"$C_M(\alpha)$"
        min = -0.175
        max = 0.2
    if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.tight_layout(rect=[0, 0, 1, 0.85])

    # Plot

    if coeff_down_plot is None:
        ax.plot(alpha, coeff_up_plot, label="Single deck")  # alpha is unchanged, but coeff has NaNs
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
        ax.set_ylim(min,max)
        ax.set_title(f"Filtered {scoff} coefficients (threshold={threshold}) - Step 1")
        ax.get_tightbbox

        return 


    # Plot both decks
    ax.plot(alpha, coeff_up_plot, color = color1,label="Upwind deck")
    ax.plot(alpha, coeff_down_plot, color = color2,label="Downwind deck")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_ylim(min, max)
    ax.legend()
    ax.set_title(f"Filtered {scoff} coefficients (threshold={threshold}) - Step 1")