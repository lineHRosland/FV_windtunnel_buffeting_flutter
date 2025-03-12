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
        """ fromWTT obtains an instance of the class StaticCoeff from two eksperimenter
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind : instance of the class experiment
        section_width : width of the bridge deck section model
        section_height : height of the bridge deck section model
        section_length_in_rig : length of the bridge deck section model
        
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
        
        filtered_wind = np.mean(spsp.sosfiltfilt(sos,experiment_in_wind_still_air_forces_removed.wind_speed))
        #need a representative mean wind velocity U for the calculation of the coefficients

        # drag_coeff = forces * 2 / (rho * U^2 * h * L)
        drag_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,0:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_height/section_length
        #0:24:6 - horisontal dir: kolonne 0, 6, 12, 18 (from load cell nr. 1, 2, 3, 4)

        # lift_coeff = forces * 2 / (rho * U^2 * b * L)
        lift_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,2:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width/section_length
        #2:24:6 - vertical dir

        # pitch_coeff = moments * 2 / (rho * U^2 * b^2 * L)
        pitch_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,4:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind**2/section_width**2/section_length
        #4:24:6 - rotation dir

        pitch_motion = experiment_in_wind_still_air_forces_removed.motion[:,2]
                
        return cls(drag_coeff,lift_coeff,pitch_coeff,pitch_motion,filtered_wind)
    
    def to_excel(self,section_name, sheet_name='Test #' ,section_width=0,section_height=0,section_length_in_rig = 0, section_length_on_wall=0, upwind_in_rig=True):
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
        
        if upwind_in_rig: #upwind: cell 0,1
            C_D_upwind = self.drag_coeff[0:-1:10,0] + self.drag_coeff[0:-1:10,1]
            C_D_downwind  =self.drag_coeff[0:-1:10,2] + self.drag_coeff[0:-1:10,3]
            C_L_upwind = self.lift_coeff[0:-1:10,0] + self.lift_coeff[0:-1:10,1]
            C_L_downwind = self.lift_coeff[0:-1:10,2] + self.lift_coeff[0:-1:10,3]
            C_M_upwind = self.pitch_coeff[0:-1:10,0] + self.pitch_coeff[0:-1:10,1]
            C_M_downwind = self.pitch_coeff[0:-1:10,2] + self.pitch_coeff[0:-1:10,3]
        
        else: #downwind: cell 1,2
            C_D_upwind = self.drag_coeff[0:-1:10,2] + self.drag_coeff[0:-1:10,3]
            C_D_downwind = self.drag_coeff[0:-1:10,0] + self.drag_coeff[0:-1:10,1]
            C_L_upwind = self.lift_coeff[0:-1:10,2] + self.lift_coeff[0:-1:10,3]
            C_L_downwind = self.lift_coeff[0:-1:10,0] + self.lift_coeff[0:-1:10,1]
            C_M_upwind = self.pitch_coeff[0:-1:10,2] + self.pitch_coeff[0:-1:10,3]
            C_M_downwind = self.pitch_coeff[0:-1:10,0] + self.pitch_coeff[0:-1:10,1]

        # Create results dataframe
 
        static_coeff = pd.DataFrame({"pitch motion": self.pitch_motion[0:-1:10],
                                "C_D_upwind": C_D_upwind,
                                "C_D_downwind": C_D_downwind,
                                "C_L_upwind": C_L_upwind,
                                "C_L_downwind": C_L_downwind,
                                "C_M_upwind": C_M_upwind,
                                "C_M_downwind": C_M_downwind,
                                 })

        # Geometry/documentation DataFrame
        setUp = pd.DataFrame({
            "D": [section_height],
            "B": [section_width],
            "L in rig": [section_length_in_rig],
            "L on wall": [section_length_on_wall],
            "Upwind in rig": [upwind_in_rig]
        })

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
               
            
    def plot_drag(self,mode="total"):
        """ plots the drag coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """        
        
        if mode == "all": #individual load cells + total sum
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
                # Sum all load cells for each time step            
            for k in range(self.drag_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,k],label=("Load cell " + str(k+1)),alpha =0.5)
                # each load cell plottet separately
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()
        
        elif mode == "decks": #upwind and downwind deck + total sum
            plt.figure()
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,0]+self.drag_coeff[:,1],label=("Downwind deck"))
            plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,2]+self.drag_coeff[:,3],label=("Upwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()

        elif mode == "total": #only total sum
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")

        elif mode == "single": #single deck
            drag_coeff = self.drag_coeff[:, :2]  # Bruk kun aktive lastceller

            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,drag_coeff[:,0]+drag_coeff[:,1],label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()

        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
                 
    def plot_lift(self,mode="total"):
        """ plots the lift coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
                
        if mode == "all":
            print("Lift coeff shape:", self.lift_coeff.shape)

            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,k],label=("Load cell " + str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,0]+self.lift_coeff[:,1],label=("Downwind deck"))
            plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,2]+self.lift_coeff[:,3],label=("Upwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")

        elif mode == "single": #single deck
            lift_coeff = self.lift_coeff[:, :2]  # Bruk kun aktive lastceller

            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,lift_coeff[:,0]+lift_coeff[:,1],label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
        
    def plot_pitch(self,mode="total"):
        """ plots the pitch coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
                
        if mode == "all":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
                            
            for k in range(self.drag_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,k],label=("Load cell " + str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            #plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,0]+self.pitch_coeff[:,1],label=("Downwind deck"))
            plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,2]+self.pitch_coeff[:,3],label=("Upwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
        
        elif mode == "single": #single deck
            pitch_coeff = self.plot_pitch_coeff[:, :2]  # Bruk kun aktive lastceller

            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,pitch_coeff[:,0]+pitch_coeff[:,1],label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
    
    def plot_drag_mean_Line(self,mode="total"):
        """ plots the drag coefficient mean
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """        
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)      

        if mode == "all": #individual load cells + total sum
            plt.figure()

            cd_total = np.sum(self.drag_coeff,axis=1)
            cd_total_mean = np.array([np.mean(cd_total[alpha == val]) for val in unique_alphas])
            
            plt.plot(unique_alphas,cd_total_mean,label = "Total")
                      
            for k in range(self.drag_coeff.shape[1]):
                cd_k = self.drag_coeff[:,k]
                cd_k_mean = np.array([np.mean(cd_k[alpha == val]) for val in unique_alphas])
                plt.plot(unique_alphas,cd_k_mean,label=("Load cell " + str(k+1)),alpha =0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()
        
        elif mode == "decks": #upwind and downwind deck + total sum
            plt.figure()
            cd_total_mean = np.array([np.mean(np.sum(self.drag_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            cd_upwind_mean = np.array([np.mean(self.drag_coeff[:,0][alpha == val]) + np.mean(self.drag_coeff[:,1][alpha == val]) for val in unique_alphas])
            cd_downwind_mean = np.array([np.mean(self.drag_coeff[:,2][alpha == val]) + np.mean(self.drag_coeff[:,3][alpha == val]) for val in unique_alphas])

            #plt.plot(unique_alphas,cd_total_mean,label = "Total")    
            plt.plot(unique_alphas,cd_upwind_mean,label=("Downwind deck")) # Switch upwind and downwind deck. For Downstream files the load cells are switched.
            plt.plot(unique_alphas,cd_downwind_mean,label=("Upwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()

        elif mode == "total": #only total sum
            plt.figure()
            cd_total_mean = np.array([np.mean(np.sum(self.drag_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            plt.plot(unique_alphas,cd_total_mean,label = "Total")    
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
        
        elif mode == "single": #single deck
            drag_coeff = self.drag_coeff[:, :2]  # Bruk kun aktive lastceller

            cd_single_mean = np.array([np.mean(drag_coeff[:,0][alpha == val]) + np.mean(drag_coeff[:,1][alpha == val]) for val in unique_alphas])
            
            plt.figure()
            plt.plot(unique_alphas,cd_single_mean,label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()

        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
                 
    def plot_lift_mean_Line(self,mode="total"):
        """ plots the lift coefficient mean
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)   

        if mode == "all":
            print("Lift coeff shape:", self.lift_coeff.shape)

            plt.figure()
            cl_total_mean = np.array([np.mean(np.sum(self.lift_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            plt.plot(unique_alphas,cl_total_mean,label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                cl_k = self.lift_coeff[:,k]
                cl_k_mean = np.array([np.mean(cl_k[alpha == val]) for val in unique_alphas])
                plt.plot(unique_alphas,cl_k_mean,label=("Load cell " + str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            cl_total_mean = np.array([np.mean(np.sum(self.lift_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            cl_upwind_mean = np.array([np.mean(self.lift_coeff[:,0][alpha == val]) + np.mean(self.lift_coeff[:,1][alpha == val]) for val in unique_alphas])
            cl_downwind_mean = np.array([np.mean(self.lift_coeff[:,2][alpha == val]) + np.mean(self.lift_coeff[:,3][alpha == val]) for val in unique_alphas])

            #plt.plot(unique_alphas,cl_total_mean,label = "Total")
            plt.plot(unique_alphas,cl_upwind_mean,label=("Downwind deck"))
            plt.plot(unique_alphas,cl_downwind_mean,label=("Upwind deck"))
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            cl_total_mean = np.array([np.mean(np.sum(self.lift_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            plt.plot(unique_alphas,cl_total_mean)
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
        
        elif mode == "single": #single deck
            lift_coeff = self.lift_coeff[:, :2]  # Bruk kun aktive lastceller

            cl_single_mean = np.array([np.mean(lift_coeff[:,0][alpha == val]) + np.mean(lift_coeff[:,1][alpha == val]) for val in unique_alphas])
            
            plt.figure()
            plt.plot(unique_alphas,cl_single_mean,label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
        
    def plot_pitch_mean_Line(self,mode="total"):
        """ plots the pitch coefficient mean 
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
        alpha = np.round(self.pitch_motion*360/2/np.pi,1)
        unique_alphas = np.unique(alpha)  
                
        if mode == "all":
            plt.figure()
            cm_total_mean = np.array([np.mean(np.sum(self.pitch_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            plt.plot(unique_alphas,cm_total_mean,label = "Total")
                            
            for k in range(self.lift_coeff.shape[1]):
                cm_k = self.pitch_coeff[:,k]
                cm_k_mean = np.array([np.mean(cm_k[alpha == val]) for val in unique_alphas])
                plt.plot(unique_alphas,cm_k_mean,label=("Load cell " + str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            cm_total_mean = np.array([np.mean(np.sum(self.pitch_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            cm_upwind_mean = np.array([np.mean(self.pitch_coeff[:,0][alpha == val]) + np.mean(self.pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
            cm_downwind_mean = np.array([np.mean(self.pitch_coeff[:,2][alpha == val]) + np.mean(self.pitch_coeff[:,3][alpha == val]) for val in unique_alphas])

           

            #plt.plot(unique_alphas,cm_total_mean,label = "Total")
            plt.plot(unique_alphas,cm_upwind_mean,label=("Downwind deck"))
            plt.plot(unique_alphas,cm_downwind_mean,label=("Upwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            cm_total_mean = np.array([np.mean(np.sum(self.pitch_coeff,axis=1)[alpha == val]) for val in unique_alphas])
            plt.plot(unique_alphas,cm_total_mean)
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")

        elif mode == "single": #single deck
            pitch_coeff = self.pitch_coeff[:, :2]  # Bruk kun aktive lastceller

            cm_single_mean = np.array([np.mean(pitch_coeff[:,0][alpha == val]) + np.mean(pitch_coeff[:,1][alpha == val]) for val in unique_alphas])
            
            plt.figure()
            plt.plot(unique_alphas,cm_single_mean,label=("Single deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
    
    
         

def plot_compare_drag(static_coeff_single, static_coeff_up, static_coeff_down):
    """
    Plots drag coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_list : list of StaticCoeff objects
        The StaticCoeff objects to compare.
    """
    plt.figure()

    plt.plot(static_coeff_single.pitch_motion*360/2/np.pi,static_coeff_single.drag_coeff[:,0]+static_coeff_single.drag_coeff[:,1],label=("Single deck"))
    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.drag_coeff[:,0]+static_coeff_up.drag_coeff[:,1],label=("Upwind deck"))
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.drag_coeff[:,0]+static_coeff_down.drag_coeff[:,1],label=("Downwind deck"))

    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.drag_coeff[:,2]+static_coeff_up.drag_coeff[:,3],label=("Up-Down deck"), alpha=0.5)
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.drag_coeff[:,2]+static_coeff_down.drag_coeff[:,3],label=("Down-Up deck"), alpha=0.5)  
    
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_D(\alpha)$")
    plt.grid()
    plt.legend()
    plt.title("Comparison of drag coefficients")

def plot_compare_lift(static_coeff_single, static_coeff_up, static_coeff_down):
    """
    Plots lift coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_list : list of StaticCoeff objects
        The StaticCoeff objects to compare.
    
    """
    plt.figure()

    plt.plot(static_coeff_single.pitch_motion*360/2/np.pi,static_coeff_single.lift_coeff[:,0]+static_coeff_single.lift_coeff[:,1],label=("Single deck"))
    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.lift_coeff[:,0]+static_coeff_up.lift_coeff[:,1],label=("Upwind deck"))
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.lift_coeff[:,0]+static_coeff_down.lift_coeff[:,1],label=("Downwind deck"))

    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.lift_coeff[:,2]+static_coeff_up.lift_coeff[:,3],label=("Up-Down deck"), alpha=0.5)
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.lift_coeff[:,2]+static_coeff_down.lift_coeff[:,3],label=("Down-Up deck"), alpha=0.5)  
    
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_L(\alpha)$")
    plt.grid()
    plt.legend()
    plt.title("Comparison of lift coefficients")

def plot_compare_pitch(static_coeff_single, static_coeff_up, static_coeff_down):
    """
    Plots pitch coefficient from multiple StaticCoeff objects in the same figure.
    
    Parameters
    ----------
    static_coeff_list : list of StaticCoeff objects
        The StaticCoeff objects to compare.
    """
    plt.figure()

    plt.plot(static_coeff_single.pitch_motion*360/2/np.pi,static_coeff_single.pitch_coeff[:,0]+static_coeff_single.pitch_coeff[:,1],label=("Single deck"))
    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.pitch_coeff[:,0]+static_coeff_up.pitch_coeff[:,1],label=("Upwind deck"))
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.pitch_coeff[:,0]+static_coeff_down.pitch_coeff[:,1],label=("Downwind deck"))

    plt.plot(static_coeff_up.pitch_motion*360/2/np.pi,static_coeff_up.pitch_coeff[:,2]+static_coeff_up.pitch_coeff[:,3],label=("Up-Down deck"), alpha=0.5)
    plt.plot(static_coeff_down.pitch_motion*360/2/np.pi,static_coeff_down.pitch_coeff[:,2]+static_coeff_down.pitch_coeff[:,3],label=("Down-Up deck"), alpha=0.5)  
    
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$C_M(\alpha)$")
    plt.grid()
    plt.legend()
    plt.title("Comparison of pitch coefficients")
