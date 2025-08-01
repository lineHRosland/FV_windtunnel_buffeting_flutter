# -*- coding: utf-8 -*-
"""
Editited spring 2025

@author: Smetoch, Rosland
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:15:20 2022

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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter



__all__ = ["Experiment",]

class Experiment:
    """ 
    A class used to represent a forced vibration wind tunnel test
    
    Attributes
    ----------    
    name : str
        a formatted string that contains the name of the experiment
    time : float
        an array that contains the time vector
    temperature : float
        The temperature measured in the wind tunnel
     air_density : float
        The air density obtained using the presure and temperature
     wind_speed : float
        an (N time step,) array that contains the wind speed measured by pitot probe
     forces_global : float
        an (N time step, 24) array that contains measured forces in fixed global coordinates at the load cells 
     forces_global_center
        an (N time step, 24) array that contains measured forces in fixed global coordinates transformed to the center of rotation
     motion : float
        an (N time step, 3) array that contains the horizontal, vertical and pitching motion of the section model. 
 
    Methods
    -------
     fromWTT(experiment)
        creates and instance of the class from the hdf5 group experiment
     align_with(experiment0)
        aligns the current experiment with an experiments performed in still air
     filt_forces(order,cutoff_frequency)
        filter the forces with a Butterworth filter as defined by given parameters
     substract(experiment0)
        substracts an experiment from still-air from the current experiment to obtain wind forces
     harmonc_groups()
        identify the starts and stops of harmonic groups in a single harmonic forced vibration test.   
     plot_motion()
        plots the horizontal, vertical and pitching motion.
     plot_forces()
        plots the measured forces 
     Plot_wind_velocity
        plots the measured wind velocity 
     plot_experiment()
        plots the entire experiment    
    """
  
    def __init__(self,name = "", time=[], temperature=[], air_density=[], wind_speed=[], forces_global=[], forces_global_center=[], motion=[]):
        
        """
        parameters:
        -----------
        name : str
          name of the experiemnt
        time : float
          time vector 
        temperature : float
          measured air temperatur in the wind tunnel
        air_density : float
          calculated air density
        wind_speed : float
          measured wind speed
        forces_global : float
          measured forces transfromed to global fixed coordinate system
        forces_global_center : float
          forces transformed to the center of the pitching motion
         motion : the motion applied in the wind tunnel experiment
        """

        self.time = time
        self.name = name
        self.temperature = temperature
        self.air_density = air_density
        self.wind_speed = wind_speed
        self.forces_global = forces_global
        self.forces_global_center = forces_global_center
        self.motion = motion

        if self.motion_type() == 2 and np.max(np.abs(self.motion[:,2])) > 0.1: #0.01 radians = 5.7 degrees
            # 1. Finn første indeks med faktisk bevegelse
            motion_threshold = 0.01  # 0.01 radianer = 0.57 deg, juster etter behov
            motion_abs = np.abs(self.motion[:, 2])
            start_candidates = np.where(motion_abs > motion_threshold)[0]

            if len(start_candidates) > 0:
                approx_start_idx = start_candidates[0]
                
                # 2. Gå bakover og finn siste nullpunkt FØR dette
                zero_candidates = np.where(np.isclose(self.motion[:, 2], 0, atol=1e-6))[0]
                earlier_zeros = zero_candidates[zero_candidates < approx_start_idx]
                
                if len(earlier_zeros) > 0:
                    start_idx = earlier_zeros[-1]  # siste null før bevegelse
                else:
                    start_idx = approx_start_idx  # fallback
            else:
                start_idx = 0  # fallback hvis aldri over terskel
            
            #  Find all indices where u_theta is (approximately) zero
            zero_indices = np.where(np.isclose(self.motion[:, 2], 0, atol=1e-6))[0]

            # Find stop_idx as the 4th zero **after** start_idx
            zero_after_start = zero_indices[zero_indices > approx_start_idx]
            if len(zero_after_start) >= 4:
                stop_idx = zero_after_start[3]
            else:
                stop_idx = len(self.motion)  # fallback to end if not enough zero crossings

            # Apply slicing
            self.forces_global_center = self.forces_global_center[start_idx:stop_idx, :]
            self.forces_global = self.forces_global[start_idx:stop_idx, :]
            self.time = self.time[start_idx:stop_idx]
            self.wind_speed = self.wind_speed[start_idx:stop_idx]
            self.motion = self.motion[start_idx:stop_idx, :]



      
    
       
       
    @classmethod    
    def fromWTT(cls,experiment):
        """ obtains an instance of Experiment from a wind tunnel test
        parameters:
        -----------
        experiment : hdf5 group (a group of dataset stored in a *.hdf5 file)
        
        """
        dt = 1.0/experiment.attrs["sampling frequency"]
        n_samples = experiment["motion"][:].shape[0]
        time = np.linspace(0,(n_samples-1)*dt,n_samples)
        name = experiment.name
        temperature = experiment["temperature"][()]
        air_density = experiment["air_density"][()]
        wind_speed = experiment["wind_velocity"][:]
        forces_global = experiment["forces_global_coord"][:]
        forces_global_center = experiment["forces_global_coord_center"][:]
        motion = experiment["motion"][:]
        
        return cls(name, time, temperature, air_density, wind_speed, forces_global, forces_global_center, motion)
        
        return
        
        
    def __str__(self):
        return f'Wind tunnel experiment: {self.name}'
    
    def __repr__(self):
        return f'Wind tunnel experiment: {self.name}'
    
    def align_with(self,experiment0):
        """ alignes the current experiment with the reference experiment0
        parameters:
        ----------
        experiment0 : instance of the class Experiment 
        
        """
                
        motions1 = experiment0.motion
        motions2 = self.motion
        
        max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
        motion_type = np.argmax(max_hor_vert_pitch_motion)
        
        motion0 = motions1[:,motion_type]
        motion1 = motions2[:,motion_type]
        
        n_points_motion0 = motion0.shape[0]
        n_points_motion1 = motion1.shape[0]
        
        cross_correlation = spsp.correlate(motion0,motion1,mode='full', method='auto')
        
        cross_correlation_coefficient = cross_correlation/(np.std(motion0)*np.std(motion1))/(n_points_motion0*n_points_motion1)**0.5
        
        correlation_lags = spsp.correlation_lags(n_points_motion0,n_points_motion1,mode='full')
        delay = correlation_lags[np.argmax(cross_correlation_coefficient)]
        
        if delay<0:
            self.forces_global = self.forces_global[-delay:-1,:]
            self.forces_global_center = self.forces_global_center[-delay:-1,:]
            self.motion = self.motion[-delay:-1,:]
            self.wind_speed = self.wind_speed[-delay:-1]
            
        if delay>0:
            self.forces_global = np.vstack((np.ones((delay,24))*self.forces_global[0,:],self.forces_global))
            self.forces_global_center = np.vstack((np.ones((delay,24))*self.forces_global_center[0,:],self.forces_global_center))
            self.motion = np.vstack((np.ones((delay,3))*self.motion[0,:],self.motion))
            self.wind_speed = np.hstack((np.ones((delay))*self.wind_speed[0],self.wind_speed))
        
        n_points_motion0 = experiment0.motion.shape[0]
        n_points_motion1 = self.motion.shape[0]
            
        if n_points_motion0>n_points_motion1:
            n_samples_to_add = n_points_motion0 - n_points_motion1
            self.forces_global = np.vstack((self.forces_global,np.ones((n_samples_to_add,24))*self.forces_global[-1,:]))
            self.forces_global_center = np.vstack((self.forces_global_center,np.ones((n_samples_to_add,24))*self.forces_global_center[-1,:]))
            self.motion = np.vstack((self.motion,np.ones((n_samples_to_add,3))*self.motion[-1,:]))
            self.wind_speed = np.hstack((self.wind_speed,np.ones((n_samples_to_add))*self.wind_speed[-1]))
                        
        if n_points_motion0<n_points_motion1:
            self.forces_global = self.forces_global[0:n_points_motion0,:] 
            self.forces_global_center = self.forces_global_center[0:n_points_motion0,:]
            self.motion = self.motion[0:n_points_motion0,:]
            self.wind_speed = self.wind_speed[0:n_points_motion0]
            
        self.time = experiment0.time
    
    def filt_forces(self,order,cutoff_frequency):
        """ filter the measured forces using a Butterworth filter
        
        parameters:
        -----------
        order : int
          filter order
        cutoff_frequency : float
          filter cutoff frequency
          
        """
        
        sampling_frequency = 1/(self.time[1]-self.time[0])
        sos = spsp.butter(order,cutoff_frequency, fs=sampling_frequency, output="sos")
        self.forces_global = spsp.sosfiltfilt(sos, self.forces_global,axis=0)
        self.forces_global_center = spsp.sosfiltfilt(sos, self.forces_global_center,axis=0)
    
    def substract(self,experiment0):
        """substract the forces measured in experiment 0 from the current experiment
        
        parameters:
        ----------
        experiment0 : instance of the class Experimet 
        
        """
        self.forces_global = self.forces_global-experiment0.forces_global
        self.forces_global_center = self.forces_global_center-experiment0.forces_global_center

    def harmonic_groups(self,plot=False):
        """ identifies the start and end positions of harmonic groups in a forced vibration test
        
        parameters:
        -----------
        plot : boolean, optional
                
        """
        motions = self.motion

        #max_hor_vert_pitch_motion = [np.max(motions[:,0]), np.max(motions[:,1]), np.max(motions[:,2]) ]
        motion_type = self.motion_type()

        motion = self.motion[:,motion_type]

        motion = motion/np.max(motion)

        peak_indexes, _ = spsp.find_peaks(motion,height=0.9,distance=100)

        start_motion = peak_indexes[0] - 800
        end_motion = peak_indexes[-1] + 800
        mean_motion = np.mean(motion[start_motion:start_motion+100])

        motion[0:start_motion] = mean_motion
        motion[end_motion:] = mean_motion

        motion = motion - mean_motion



        filter_order = 6
        cutoff_frequency = 7
        sampling_frequency = 1/(self.time[1]- self.time[0])
       
        sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
        
        motion = spsp.sosfiltfilt(sos,motion)

        motion_normalized = motion/np.max(motion)

        peak_indexes, _ = spsp.find_peaks(motion_normalized, height=0.3)

        difference_in_peak_spacing = np.diff(peak_indexes,2)
        #print(difference_in_peak_spacing)
        difference_in_peak_spacing[np.abs(difference_in_peak_spacing)<40] = 0 #If the difference is small it is due to noise
        #print(difference_in_peak_spacing)
        difference_in_peak_spacing[difference_in_peak_spacing<0] = 0 # The spacing becomes nagative when the frequency change
        #print(difference_in_peak_spacing)
        start_group_indicator = np.hstack([300, 0, difference_in_peak_spacing]) # 
        stop_group_indicator = np.hstack([ 0, difference_in_peak_spacing, 300])

        start_groups = peak_indexes[start_group_indicator>70]
        stop_groups = peak_indexes[stop_group_indicator>70]            

        # Remove first and last points in group if the motion is not fully developed
        new_start_groups = np.zeros(len(start_groups),dtype=int)
        new_stop_groups = np.zeros(len(start_groups),dtype=int)
        for k in range(len(start_groups)):
            peaks_in_group, _ = spsp.find_peaks(motion_normalized[start_groups[k]:stop_groups[k]], height=0.9)
            new_start_groups[k] =  start_groups[k] + peaks_in_group[0]
            new_stop_groups[k] = start_groups[k] + peaks_in_group[-1]
            
        if plot!=False:
            plt.figure()
            plt.plot(motion_normalized)
            # plt.plot(peak_indexes, motion_normalized[peak_indexes], "x")
            # plt.plot(peak_indexes, motion_normalized[peak_indexes], "o")
            # plt.plot(peak_indexes, motion_normalized[peak_indexes], "o")
            
            #plt.plot(start_groups, motion_normalized[start_groups], "o")
            #plt.plot(stop_groups, motion_normalized[stop_groups], "o")
            
            

            plt.plot(new_start_groups, motion_normalized[new_start_groups], "o")
            plt.plot(new_stop_groups, motion_normalized[new_stop_groups], "o")
            plt.show()
        
        return start_groups, stop_groups
    
    def motion_type(self):

        sos_hp = spsp.butter(6,0.1, fs=1, output="sos", btype="high")

        filt_motions = spsp.sosfiltfilt(sos_hp,self.motion,axis=0)

        max_hor_vert_pitch_motion = [np.max(filt_motions[:,0])*0, np.max(filt_motions[:,1])*1000, np.max(filt_motions[:,2])*365/2/np.pi ]
        motion_type = np.argmax(max_hor_vert_pitch_motion)



        return motion_type
        
    
    def plot_motion(self,fig=[]):
        """ plots the motion applied in the wind tunnel test
        
        """
        
        if bool(fig) == False:
            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            for k in [2,3]:
                fig.add_subplot(3,1,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,15/2.54)
        
        axs[0].plot(self.time,self.motion[:,0])
        axs[0].set_title("Horizontal motion")
        axs[0].set_ylabel(r"$u_x $ [mm]")
        axs[0].grid(True)
        
        axs[1].plot(self.time,self.motion[:,1])
        axs[1].set_title("Vertical motion")
        axs[1].set_ylabel(r"$u_z $ [mm]")
        axs[1].grid(True)
        
        axs[2].plot(self.time,self.motion[:,2])
        axs[2].set_title("Pitching motion")
        axs[2].set_ylabel(r"$u_\theta$")
        axs[2].set_xlabel(r"$Time$ [s]")
        axs[2].grid(True)
        
        fig.tight_layout(rect=[0, 0, 1, 0.85])

        
        fig.show()
        
        
        
    def plot_forces(self,mode="all",fig=[]):
        """ plot the measured forces
        
        """

        
        if bool(fig) == False:
            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            for k in [2,3]:
                fig.add_subplot(3,1,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,15/2.54)
        
        if mode == "all":

            axs[0].plot(self.time,self.forces_global_center[:,0:24:6])
            
            axs[1].plot(self.time,self.forces_global_center[:,2:24:6])

            axs[2].plot(self.time,self.forces_global_center[:,4:24:6])
           
        elif mode == "decks":            
            
            axs[0].plot(self.time,np.sum(self.forces_global_center[:,0:12:6],axis=1),label = "Upwind deck")
            axs[0].plot(self.time,np.sum(self.forces_global_center[:,12:24:6],axis=1),label = "Downwind deck")
            
            axs[1].plot(self.time,np.sum(self.forces_global_center[:,2:12:6],axis=1),label = "Upwind deck")
            axs[1].plot(self.time,np.sum(self.forces_global_center[:,14:24:6],axis=1),label = "Downwind deck")
            
            axs[2].plot(self.time,np.sum(self.forces_global_center[:,4:12:6],axis=1),label = "Upwind deck")
            axs[2].plot(self.time,np.sum(self.forces_global_center[:,16:24:6],axis=1),label = "Downwind deck")

        elif mode == "total":

            axs[0].plot(self.time,np.sum(self.forces_global_center[:,0:24:6],axis=1),label = "Total")

            axs[1].plot(self.time,np.sum(self.forces_global_center[:,2:24:6],axis=1),label = "Total")

            axs[2].plot(self.time,np.sum(self.forces_global_center[:,4:24:6],axis=1),label = "Total")
            
        axs[0].set_title("Horizontal force")
        axs[0].grid(True)
        axs[0].set_ylabel(r"$F_x$")
        axs[0].legend()
    
        axs[1].set_title("Vertical force")
        axs[1].grid(True)
        axs[1].set_ylabel(r"$F_z$")
        axs[1].legend()
        
        axs[2].set_title("Pitching moment")
        axs[2].grid(True)
        axs[2].set_ylabel(r"$F_\theta$")
        axs[2].set_xlabel(r"$Time$ [s]")
        axs[2].legend()
        
        fig.tight_layout(rect=[0, 0, 1, 0.85])

    
    def plot_wind_velocity(self,fig=[]):
        """ plot the measured wind velocity
        
        """

        if bool(fig) == False:
            fig = plt.figure()
            ax = fig.add_subplot(4,2,1)
            for k in [1,3,4,5,6,7,8]:
                fig.add_subplot(4,2,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,5/2.54)
        
        axs[0].plot(self.time,self.wind_speed)
        axs[0].set_ylabel(r"$U(t)$ [m/s]")
        axs[0].set_xlabel(r"$Time$ [s]")
        axs[0].set_title("Wind speed")
        axs[0].grid(True)
        
        return fig
    
    def plot_experiment(self, mode="decks", fig=None):
        """plots the wind velocity, motions and forces
        
        """


        if bool(fig) == False:
            fig = plt.figure()
            gs = fig.add_gridspec(4, 2)
            axs = []
            axs.append(fig.add_subplot(gs[0, :]))   # wind speed over both columns
            axs.append(fig.add_subplot(gs[1, 0]))   # u_x
            axs.append(fig.add_subplot(gs[2, 0]))   # u_z
            axs.append(fig.add_subplot(gs[3, 0]))   # u_theta
            axs.append(fig.add_subplot(gs[1, 1]))   # F_x
            axs.append(fig.add_subplot(gs[2, 1]))   # F_z
            axs.append(fig.add_subplot(gs[3, 1]))   # F_theta
            
                        
        fig.set_size_inches(20/2.54,15/2.54)
        fig.tight_layout(rect=[0, 0, 1, 0.85])

        fig.align_ylabels()

        
        
       
        
        if mode == "all": #hver enkelt lastcelle plottes separat

            axs[0].plot( self.time,self.wind_speed)
            axs[0].set_title("Average wind speed")
            axs[0].set_ylabel(r"$U(t)$ [m/s]")
            axs[0].grid(True)
             
            axs[1].plot( self.time,self.motion[:,0])
            axs[1].set_title("Horisontal motion")
            axs[1].set_ylabel(r"$u_x $ [mm]")
            axs[1].grid(True)
            
            axs[2].plot( self.time,self.motion[:,1])
            axs[2].set_title("Vertical motion")
            axs[2].set_ylabel(r"$u_z $ [mm]")
            axs[2].grid(True)
            
            axs[3].plot( self.time,self.motion[:,2])
            axs[3].set_title("Pitching motion")
            axs[3].set_ylabel(r"$u_\theta$  [radianer]")
            axs[3].set_xlabel(r"$Time$ [s]")
            axs[3].grid(True)
            
            axs[4].plot( self.time,self.forces_global_center[:,0:24:6])
            axs[4].set_title("Horizontal force")
            axs[4].grid(True)
            axs[4].set_ylabel(r"$F_x$ [N]")
            axs[4].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            axs[5].plot( self.time,self.forces_global_center[:,2:24:6])
            axs[5].set_title("Vertical force")
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z [N]$")
            axs[5].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            axs[6].plot( self.time,self.forces_global_center[:,4:24:6])
            axs[6].set_title("Pitching moment")
            axs[6].grid(True)
            axs[6].set_ylabel(r"$F_\theta [Nm]$")
            axs[6].set_xlabel(r"$Time$ [s]")
            axs[6].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            fig.tight_layout(rect=[0, 0, 1, 0.85])

        
        elif mode == "decks": # krefter og momenter summeres per brodekke
            
            axs[0].plot( self.time,self.wind_speed, alpha = 0.6)
            axs[0].set_title("Wind speed",fontsize=15)
            axs[0].set_ylabel(r"$U(t)$ [m/s]", fontsize=15)
            axs[0].tick_params(axis='both', labelsize=15)
            axs[0].grid(True)
            
            axs[1].plot( self.time,self.motion[:,0], alpha = 0.6)
            axs[1].set_title("Horizontal motion",fontsize=15)
            axs[1].set_ylabel(r"$u_x $ [mm]", fontsize=15)
            axs[1].tick_params(axis='both', labelsize=15)
            axs[1].set_ylim([-0.001, 0.001])
            axs[1].grid(True)
            
            axs[2].plot( self.time,self.motion[:,1], alpha = 0.6)
            axs[2].set_title("Vertical motion",fontsize=15)
            axs[2].set_ylabel(r"$u_z $ [mm]", fontsize=15)
            axs[2].tick_params(axis='both', labelsize=15)
            axs[2].set_ylim([-0.001, 0.001])
            axs[2].grid(True)
            
            axs[3].plot( self.time,self.motion[:,2], alpha = 0.6)
            axs[3].set_title("Pitching motion",fontsize=15)
            axs[3].set_ylabel(r"$u_\theta [rad]$", fontsize=15)
            axs[3].set_xlabel(r"$Time$ [s]", fontsize=15)
            axs[3].tick_params(axis='both', labelsize=15)
            axs[3].grid(True)
            
            axs[4].plot( self.time,np.sum(self.forces_global_center[:,0:12:6],axis=1),label = "Upwind deck", alpha = 0.6) # lastcelle 1 og 2 f.eks
            axs[4].plot( self.time,np.sum(self.forces_global_center[:,12:24:6],axis=1),label = "Downwind deck", alpha= 0.6) # lastcelle 3 og 4 f.eks
            axs[4].set_title("Horizontal force",fontsize=15)
            axs[4].grid(True)
            axs[4].set_ylabel(r"$F_x [N]$", fontsize=15)
            axs[4].tick_params(axis='both', labelsize=15)
            # axs[4].legend()
           
            
            #axs[2,1].plot( self.time,self.forces_global_center[:,2:24:6])
            axs[5].plot( self.time,np.sum(self.forces_global_center[:,2:12:6],axis=1),label = "Upwind deck", alpha = 0.6)
            axs[5].plot( self.time,np.sum(self.forces_global_center[:,14:24:6],axis=1),label = "Downwind deck", alpha = 0.6)
            axs[5].set_title("Vertical force",fontsize=15)
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z [N]$", fontsize=15)
            axs[5].tick_params(axis='both', labelsize=15)
            # axs[5].legend()
           
            
            #axs[3,1].plot( self.time,self.forces_global_center[:,4:24:6])
            axs[6].plot( self.time,np.sum(self.forces_global_center[:,4:12:6],axis=1),label = "Upwind deck", alpha = 0.6)
            axs[6].plot( self.time,np.sum(self.forces_global_center[:,16:24:6],axis=1),label = "Downwind deck", alpha = 0.6)
            axs[6].set_title("Pitching moment",fontsize=15)
            axs[6].grid(True)
            axs[6].set_ylabel(r"$F_\theta$ [Nm]", fontsize=15)
            axs[6].set_xlabel(r"$Time$ [s]", fontsize=15)
            axs[6].tick_params(axis='both', labelsize=15)
            # axs[6].legend()

            for ax in axs:
                ax.tick_params(labelsize=13)
            fig.tight_layout(rect=[0.05, 0, 1, 0.90])

            
        elif mode == "total":
            
            axs[0].plot( self.time,self.wind_speed)
            axs[0].set_title("Wind speed")            
            axs[0].set_ylabel(r"$U(t)$ [m/s]", fontsize=12)
            axs[0].grid(True)
            
            axs[1].plot( self.time,self.motion[:,0])
            axs[1].set_title("Horizontal motion")
            axs[1].set_ylabel(r"$u_x $ [mm/s]", fontsize=12)
            # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # alltid vitenskapelig
            axs[1].grid(True)
            
            axs[2].plot( self.time,self.motion[:,1])
            axs[2].set_title("Vertical motion")
            axs[2].set_ylabel(r"$u_z $ [mm/s]", fontsize=12)
            axs[2].grid(True)
            
            axs[3].plot( self.time,self.motion[:,2])
            axs[3].set_title("Pitching motion")
            axs[3].set_ylabel(r"$u_\theta$ [rad/s]", fontsize=12)
            axs[3].set_xlabel(r"$Time$ [s]", fontsize=12)
            axs[3].grid(True)
            
            axs[4].plot( self.time,np.sum(self.forces_global_center[:,0:24:6], axis=1),label = "Total")
            axs[4].set_title("Horizontal force")
            axs[4].grid(True)
            axs[4].set_ylabel(r"$F_x$ [N/s]", fontsize=12)
            #axs[4].legend()
           
            
            axs[5].plot( self.time,np.sum(self.forces_global_center[:,2:24:6], axis=1),label = "Total")
            axs[5].set_title("Vertical force")
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z$ [N/s]", fontsize=12)
            #axs[5].legend()
           
            
            axs[6].plot( self.time,np.sum(self.forces_global_center[:,4:24:6], axis=1),label = "Total")
            axs[6].set_title("Pitching moment")
            axs[6].grid(True)
            axs[6].set_ylabel(r"$F_\theta$ [Nm/s]", fontsize=12)
            axs[6].set_xlabel(r"$Time$ [s]", fontsize=12)
            #axs[6].legend()
            
            for ax in axs:
                ax.tick_params(labelsize=12)
            fig.tight_layout(rect=[0.05, 0, 1, 0.90])

        

        
