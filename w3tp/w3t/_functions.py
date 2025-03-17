# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:24:00 2022

@author: oiseth
"""
import numpy as np
from scipy import signal as spsp




__all__ = ["group_motions", "align_experiments", ]
  

def group_motions(experiments):
    """ identifies wind tunnel tests with the same motion
    
    A list of experiment objects is can be passed into the function. 
    The function use the cross-correlations of all motions to identify wind 
    tunnel experiments where the same motion have been applied
    
    Arguments:
    ----------
    experiments  : a list of instances of the class Exeriment
    
    Returns:
    --------    
    a list that contains
    
    Example
    -------
    
    
    
    """
    upper_triangular_match_making_matrix = np.zeros((len(experiments),len(experiments))) 
            # Lager en N×N matrise (der N er antall eksperimenter)

    for k1 in range(len(experiments)): #  radindeks
        for k2 in range(k1+1,len(experiments)): # k2 starter alltid ett hakk etter k1, og dermed går man kun gjennom de parene som ligger i øvre trekant.

            # Henter ut bevegelsesdataene for to eksperimenter:
            motions1 = experiments[k1].motion
            motions2 = experiments[k2].motion

            max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
                # motions1[:, 0] = horisontal bevegelse
                # motions1[:, 1] = vertikal bevegelse
                # motions1[:, 2] = rotasjon bevegelse

            motion_type = np.argmax(max_hor_vert_pitch_motion)
            # finner indeksen til den største verdien i max_hor_vert_pitch_motion - finner den mest dominerende bevegelsen, f.eks rotasjon
            
            # Henter ut den mest dominerende bevegelsen for de to eksperimentene
            motion1 = motions1[:,motion_type]
            motion2 = motions2[:,motion_type]
                # motion_type baseres på motion 1, viktig å kunne sammenlige samme type bevegelse i begge eksperiementene
                # Hvis motion2 har en annen bevegelseskomponent som dominerer, er de uansett ikke ansett som like
            
            # Lagrer antall datapunkter for hver av bevegelsene:            
            n_points_motion1 = motion1.shape[0]
            n_points_motion2 = motion2.shape[0]
            
            # Beregner krysskorrelasjonen mellom bevegelsene:
            cross_correlation = spsp.correlate(motion1,motion2,mode='full', method='auto')

            #Normaliserer krysskorrelasjonen:
            cross_correlation_coefficient = cross_correlation/(np.std(motion1)*np.std(motion2))/(n_points_motion1*n_points_motion2)**0.5
            # Gir en verdi mellom -1 og 1, 1 = perfect match

            #correlation_lags = spsp.correlation_lags(n_points_motion1,n_points_motion2,mode='full')

            upper_triangular_match_making_matrix[k1,k2] = np.max(cross_correlation_coefficient) 
            # den  øvre trekanten av en kvadratisk matrise: f. eks. sammenligning mellom eksperiment 1 og 3 er det samme som eksperiment 3 og 1.

            # In case the motion is the same, but with different amplitude or mean value (Will not be detected by correlation)
            if np.abs((np.max(motion1)-np.max(motion2)))/np.max(motion1)>1/100:  # sjekker om forskjellen i amplitude er større enn 1%
                upper_triangular_match_making_matrix[k1,k2]  = 0  # Nullstiller korrelasjonsmatrisen hvis forskjellen er for stor
                # np.abs((np.max(motion1)-np.max(motion2)))/np.max(motion1): hvor stor forskjellen er i prosent 

            # In case the motion is the same, but with different mean value for other components
            # Er de andre komponentene (de som ikke er motion_type) også tilstrekkelig like?
            # forskjeller i maksimal og minimal verdi (amplitude) i de ikke-dominerende komponentene?
            for m in range(3):
                if m == motion_type:
                    continue # Hopper over hovedkomponenten
                else: #Sjekker om maksverdien i komponent m er veldig forskjellig mellom eksperimentene.
                    if np.abs(np.max(experiments[k1].motion[:,m])-np.max(experiments[k2].motion[:,m]))>1/100: 
                        upper_triangular_match_making_matrix[k1,k2]  = 0
                    if np.abs(np.min(experiments[k1].motion[:,m])-np.min(experiments[k2].motion[:,m]))>1/100:
                        upper_triangular_match_making_matrix[k1,k2]  = 0
                        # Det er en kontrollsjekk for å sikre at de andre komponentene er små nok, 
                        # og altså ikke «forstyrrer» det som skal være en «ren» bevegelse i én retning

        # vil være helt sikre på at de to eksperimentene er "rene" og at de andre bevegelseskomponentene
        #  (de ikke-dominerende DOF-ene) er like små i begge eksperimentene.
        # np.abs(np.max(np.abs(motion))):
         # 1. (abs) gjør alle verdiene positive (tar absoluttverdien av bevegelsen – altså amplituden i hvert øyeblikk) (blir et alternativ til å sjekke først max, og så min, som de gjør over)
         # 2. (max) finner den høyeste av disse amplitudene (dvs. maksimal bevegelsesutslag)
         # 3. (abs) har ingen effekt her, for max av absoluttverdier er alltid positiv uansett.

        # JEG SYNES DET VIRKER SOM AT OLE HARDKODET DETTE, SÅ FORBEDRET DET TIL DET OVER, OG GLEMTE Å SLETTE DENNE SJEKKEN HER.
            if motion_type == 0: #horisontal bevegelse dominerer
                if (np.abs(np.max(np.abs(experiments[k1].motion[:,1]))- np.max(np.abs(experiments[k2].motion[:,1]))) > 1/100) or (np.abs(np.max(np.abs(experiments[k1].motion[:,2]))- np.max(np.abs(experiments[k2].motion[:,2]))) > 1/100):
                    upper_triangular_match_making_matrix[k1,k2]  = 0
            if motion_type == 1: #vertikal bevegelse dominerer
                if (np.abs(np.max(np.abs(experiments[k1].motion[:,0]))- np.max(np.abs(experiments[k2].motion[:,0]))) > 1/100) or (np.abs(np.max(experiments[k1].motion[:,2])- np.max(experiments[k2].motion[:,2])) > 1/100):
                    upper_triangular_match_making_matrix[k1,k2]  = 0
            if motion_type == 2: #rotasjon bevegelse dominerer
                if (np.abs(np.max(experiments[k1].motion[:,0])- np.max(experiments[k2].motion[:,0])) > 1/100) or (np.abs(np.max(experiments[k1].motion[:,1])- np.max(experiments[k2].motion[:,1])) > 1/100): # sjekker om forskjellen i amplitude er større enn 1%
                    upper_triangular_match_making_matrix[k1,k2]  = 0 # Nullstiller korrelasjonsmatrisen hvis forskjellen er for stor

    # Fjerner korrelasjoner under 90%
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix<0.9] = 0
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix>=0.9] = 1

    match_making_matrix = upper_triangular_match_making_matrix + upper_triangular_match_making_matrix.T + np.eye(len(experiments))
    # Sikrer at matrisen er symmetrisk
        # np.eye(len(experiments)) setter diagonalene til 1 (et eksperiment samsvarer alltid med seg selv)
    
    tests_with_equal_motion = []
    for k1 in range(len(experiments)):
        equal_motion = np.array(np.where(match_making_matrix[k1,:]==1)) # Henter ut indeksene som har lik bevegelse
        if equal_motion.shape[1]>1: # Hvis det er flere enn ett eksperiment som har samme bevegelse
            tests_with_equal_motion.append(np.where(match_making_matrix[k1,:]==1)[0])
            for q in equal_motion:
                match_making_matrix[q,:] = match_making_matrix[q,:]*0
                # Sikrer at hvert eksperiment kun blir gruppert en gang
    return tests_with_equal_motion
    # liste over grupper (arrays) med like bevegelser

def group_motions_legacy(experiments):
    """ identifies wind tunnel tests with the same motion
    
    A list of experiment objects is can be passed into the function. 
    The function use the cross-correlations of all motions to identify wind 
    tunnel experiments where the same motion have been applied
    
    Arguments:
    ----------
    experiments  : a list of instances of the class Exeriment
    
    Returns:
    --------    
    a list that contains
    
    Example
    -------
    
    
    
    """
    upper_triangular_match_making_matrix = np.zeros((len(experiments),len(experiments))) 
    for k1 in range(len(experiments)):
        for k2 in range(k1+1,len(experiments)):

            motions1 = experiments[k1].motion
            motions2 = experiments[k2].motion

            max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
            motion_type = np.argmax(max_hor_vert_pitch_motion)

            motion1 = motions1[:,motion_type]
            motion2 = motions2[:,motion_type]
            

            n_points_motion1 = motion1.shape[0]
            n_points_motion2 = motion2.shape[0]
            
            cross_correlation = spsp.correlate(motion1,motion2,mode='full', method='auto')

            cross_correlation_coefficient = cross_correlation/(np.std(motion1)*np.std(motion2))/(n_points_motion1*n_points_motion2)**0.5

            #correlation_lags = spsp.correlation_lags(n_points_motion1,n_points_motion2,mode='full')

            upper_triangular_match_making_matrix[k1,k2] = np.max(cross_correlation_coefficient)
            
            # In case the motion is the same, but with different amplitude (Will not be detected by correlation)
            if np.abs((np.max(motion1)-np.max(motion2)))/np.max(motion1)>1/100:
                upper_triangular_match_making_matrix[k1,k2]  = 0
  
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix<0.9] = 0
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix>=0.9] = 1

    match_making_matrix = upper_triangular_match_making_matrix + upper_triangular_match_making_matrix.T + np.eye(len(experiments))

    tests_with_equal_motion = []
    for k1 in range(len(experiments)):
        equal_motion = np.array(np.where(match_making_matrix[k1,:]==1))
        if equal_motion.shape[1]>1:
            tests_with_equal_motion.append(np.where(match_making_matrix[k1,:]==1)[0])
            for q in equal_motion:
                match_making_matrix[q,:] = match_making_matrix[q,:]*0
    return tests_with_equal_motion

def align_experiments(experiment0,experiment1):

    motions1 = experiment0.motion
    motions2 = experiment1.motion
    
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
        experiment0.forces_global = np.vstack((np.ones((-delay,24))*experiment0.forces_global[0,:],experiment0.forces_global))
        experiment0.forces_global_center = np.vstack((np.ones((-delay,24))*experiment0.forces_global_center[0,:],experiment0.forces_global_center))
        experiment0.motion = np.vstack((np.ones((-delay,3))*experiment0.motion[0,:],experiment0.motion))
        experiment0.wind_speed = np.hstack((np.ones((-delay))*experiment0.wind_speed[0],experiment0.wind_speed))
        
    if delay>0:
        experiment1.forces_global = np.vstack((np.ones((delay,24))*experiment1.forces_global[0,:],experiment1.forces_global))
        experiment1.forces_global_center = np.vstack((np.ones((delay,24))*experiment1.forces_global_center[0,:],experiment1.forces_global_center))
        experiment1.motion = np.vstack((np.ones((delay,3))*experiment1.motion[0,:],experiment1.motion))
        experiment1.wind_speed = np.hstack((np.ones((delay))*experiment1.wind_speed[0],experiment1.wind_speed))
    
    n_points_motion0 = experiment0.motion.shape[0]
    n_points_motion1 = experiment1.motion.shape[0]
        
    if n_points_motion0>n_points_motion1:
        n_samples_to_add = n_points_motion0 - n_points_motion1
        experiment1.forces_global = np.vstack((experiment1.forces_global,np.ones((n_samples_to_add,24))*experiment1.forces_global[-1,:]))
        experiment1.forces_global_center = np.vstack((experiment1.forces_global_center,np.ones((n_samples_to_add,24))*experiment1.forces_global_center[-1,:]))
        experiment1.motion = np.vstack((experiment1.motion,np.ones((n_samples_to_add,3))*experiment1.motion[-1,:]))
        experiment1.wind_speed = np.hstack((experiment1.wind_speed,np.ones((n_samples_to_add))*experiment1.wind_speed[-1]))
       
        
        
    if n_points_motion0<n_points_motion1:
        n_samples_to_add = n_points_motion1 - n_points_motion0
        experiment0.forces_global = np.vstack((experiment0.forces_global,np.ones((n_samples_to_add,24))*experiment0.forces_global[-1,:]))
        experiment0.forces_global_center = np.vstack((experiment0.forces_global_center,np.ones((n_samples_to_add,24))*experiment0.forces_global_center[-1,:]))
        experiment0.motion = np.vstack((experiment0.motion,np.ones((n_samples_to_add,3))*experiment0.motion[-1,:]))
        experiment0.wind_speed = np.hstack((experiment0.wind_speed,np.ones((n_samples_to_add))*experiment0.wind_speed[-1]))
        
        
    n_samples = np.max([experiment0.motion.shape[0],experiment1.motion.shape[0]])
    dt = experiment0.time[1]-experiment0.time[0]
    common_time = np.linspace(0,(n_samples-1)*dt,n_samples)
    experiment0.time = common_time
    experiment1.time = common_time
    
    return experiment0, experiment1