# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:59:57 2023

@author: Justin
"""


from astropy.coordinates import SkyCoord, CartesianDifferential, Galactocentric
from galpy.potential import MWPotential2014, MiyamotoNagaiPotential
import astropy.units as u
from galpy.orbit import Orbit
import numpy as np


def gaia_to_galactocentric_phase_space(ra, dec, distance, pm_ra, pm_dec, radial_velocity):
    # Define Gaia phase space coordinates
    gaia_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc,
                           pm_ra_cosdec=pm_ra*u.mas/u.yr, pm_dec=pm_dec*u.mas/u.yr,
                           radial_velocity=radial_velocity*u.km/u.s,
                           frame='icrs')

    # Define the Galactocentric frame
    galactocentric_frame = Galactocentric(galcen_distance=8.3*u.kpc, z_sun=20.8*u.pc)

    # Convert Gaia coordinates to Galactocentric phase space coordinates
    galactocentric_coords = gaia_coords.transform_to(galactocentric_frame)

    # Extract Galactocentric phase space coordinates in [pc, pc, pc, km/s, km/s, km/s]
    galactocentric_position = galactocentric_coords.cartesian.xyz.value
    galactocentric_velocity = galactocentric_coords.velocity.d_xyz.value
    
    phase_coords = np.concatenate((galactocentric_position, galactocentric_velocity), axis=0)
    return phase_coords



def star_integrator(x0_nom, x0_surr, tph_nom, tph_surr, n_step=1000, potential=MWPotential2014, diff=True):
    '''
    if diff == Fasle:
        This function accepts the initial coordinates of the star and it's surrogates.
        And returns galactocentric coordinates of encounter of the star 
        also, encounter parameters.

    '''
    #initial nominal phase space coords and surrogates phase coords for star in idx
    #first element of X0 and tph_lma,ts is for orginal star
    X0 = np.vstack((x0_nom, x0_surr))
    #initial coordinates of the sun
    x0_sun = np.array([0,0,0,0,0,0])
    #insert tph of star into tph_surr at index 0 
    #now this variable has time range of integration for each surrogate
    tph_lma = np.insert(tph_surr, 0, tph_nom) 
    
    #create (0,2*tph,n_step) time steps for nominal star and surrogate star
    indices = np.linspace(0, 1, num=n_step)
    ts = 2*tph_lma[:, np.newaxis] * indices
    
    #make a for loop to integrate each surrogate and orginal star with init condition (X0, ts)
    enc_time = np.zeros(X0.shape[0]) #store the close encounter time of each surrogate and orginal star
    enc_dist = np.zeros(X0.shape[0]) #store the close encountere distance of each surrogate and orginal star
    enc_vel = np.zeros(X0.shape[0]) #store the close encountere velocity of each surrogate and orginal star
    enc_xyzv = np.zeros((X0.shape[0],6)) #coordinates of surrogates and orginal star in close encounter distance
    init_xyzv = np.zeros((X0.shape[0],6)) #initial coordinates of surrogates and orginal star

    enc_dxyzv = np.zeros((X0.shape[0],6)) #coordinates difference of surrogates and orginal star in close encounter distance
    init_dxyzv = np.zeros((X0.shape[0],6)) #initial difference coordinates of surrogates and orginal star

    for i in range(X0.shape[0]): #take each orginal/surrogate and integrate
        init_pos =  np.array([X0[i], x0_sun])
        orbits = Orbit(init_pos ,radec=True, ro=8*u.kpc, vo=217*u.km/u.s, zo=0.010*u.kpc, solarmotion=[-11.1, 25, 7.25]*u.km/u.s)
        orbits.integrate(ts[i]*u.yr, potential, method='odeint')
        
        
        X, Y, Z = orbits.x(ts[i]*u.yr)[0], orbits.y(ts[i]*u.yr)[0], orbits.z(ts[i]*u.yr)[0]
        Vx, Vy, Vz = orbits[0].vx(ts[i]*u.yr), orbits[0].vy(ts[i]*u.yr), orbits[0].vz(ts[i]*u.yr)
        #X0, Y0, Z0 = orbits.x(0*u.yr)[0], orbits.y(0*u.yr)[0], orbits.z(0*u.yr)[0]
        init_xyzv[i] = np.array([X[0], Y[0], Z[0], Vx[0], Vy[0], Vz[0]])
        
        Xsun, Ysun, Zsun  = orbits.x(ts[i]*u.yr)[1], orbits.y(ts[i]*u.yr)[1], orbits.z(ts[i]*u.yr)[1]
        Vxsun, Vysun, Vzsun = orbits.vx(ts[i]*u.yr)[1], orbits.vy(ts[i]*u.yr)[1], orbits.vz(ts[i]*u.yr)[1]
        
        dx = X - Xsun
        dy = Y - Ysun
        dz = Z - Zsun
        
        dvx = Vx - Vxsun
        dvy = Vy - Vysun
        dvz = Vz - Vzsun
        
        #minimum encounter distance for each surrogate star with sun
        
        #index of the minimum encounter distance is stored
        min_idx = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1).argmin() 
        #time of close encounter for each surrogate and it's orginal star
        enc_time[i] = ts[i][min_idx] 
        #coordinate difference with sun for each surrogate at close encounter
        if diff:
            init_dxyzv[i] = np.array([dx[0], dy[0], dz[0], dvx[0], dvy[0], dvz[0]])
            #distance of close encounter for each surrogate
            #enc_dist[i] = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1)[min_idx] #kpc
            #close encounter velocity of each surrogate and orginal star (km/s)
            #enc_vel[i] = np.linalg.norm(np.array([dvx,dvy,dvz]).T,axis=1)[min_idx] 
            #phase coordinates of each surrogate at time of close encounter
            enc_dxyzv[i] = np.array([dx[min_idx], dy[min_idx], dz[min_idx], dvx[min_idx], dvy[min_idx], dvz[min_idx]])

            #return init_dxyzv, enc_dxyzv, enc_time
        else:
            #distance of close encounter for each surrogate
            enc_dist[i] = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1)[min_idx] #kpc
            #close encounter velocity of each surrogate and orginal star (km/s)
            enc_vel[i] = np.linalg.norm(np.array([dvx,dvy,dvz]).T,axis=1)[min_idx] 
            #phase coordinates of each surrogate at time of close encounter
            enc_xyzv[i] = np.array([X[min_idx], Y[min_idx], Z[min_idx], Vx[min_idx], Vy[min_idx], Vz[min_idx]])

            #return init_xyzv, enc_xyzv, enc_time#, enc_dist, enc_vel 

    if diff:
        return init_dxyzv, enc_dxyzv, enc_time#, enc_dist, enc_vel 
    else:
        return init_xyzv, enc_xyzv, enc_time#, enc_dist, enc_vel 
'''

def star_encounter_integration(star_data, n_step=1000, potential = MWPotential2014, method='odeint'):
    
    
    x0_sun = np.array([0,0,0,0,0,0]) #initial position of sun in ICRS
    
    x0_nom = star_data[['ra','dec','parallax','pmra','pmdec','radial_velocity']].values #initial condition
    x0_nom[2] = 1/x0_nom[2] #replace the parallax with radial distance in kpc
   
    x0_surr = star_data['surrogates']
    x0_surr[:,2] = 1/x0_surr[:,2] #convert parallax of surrogates to radial distance in kpc
    
    tph_surr = star_data['tph_surr_lma']#.values.astype(float) #close encounter times of each surrogate star with sun   
    t_nom = np.linspace(0, 2*star_data['tph_nom_lma'], n_step)*u.yr #integration time range which is 0 to 2 times close encounter time
    
    
    indices = np.linspace(0, 1, num=n_step)
    t_surr = 2*tph_surr[:, np.newaxis] * indices #time range of integration for each surrogate star (0, 2*tph_surr)

    #orbit_nom = orbit_integrate(t_nom, x_nom, potential)
    
    init_xyz = np.zeros((len(t_surr),3))
    enc_time = np.zeros((len(t_surr)))
    enc_dist = np.zeros(len(t_surr))
    enc_Xt = np.zeros((len(t_surr),3))
    #enc_V = np.zeros((len(t_surr),4))
    
    for idx, ts in enumerate(t_surr):
       
        #This loop integrates the position of sun with surrogate star with their respective surrogate
        #time range centered on close encounter time with Sun.
        
       
        #initial position of surrogate star and the sun
        init_pos = np.array([x0_surr[idx], x0_sun])
      
        #do orbit integration for each surrogate star
        #at the same time do the orbit integration for sun
        orbits = Orbit(init_pos,radec=True, ro=8,vo=242,zo=0.002)
        orbits.integrate(ts*u.yr, potential,method)
        #time = ts*u.yr
        
        #initial coordinates of each surrogate star
        X0, Y0, Z0 = orbits.x(0*u.yr)[0], orbits.y(0*u.yr)[0], orbits.z(0*u.yr)[0]
        init_xyz[idx] = np.array([X0, Y0, Z0]) 
        #galactic coordinates of surrogates at the range of close encounter times ts
        X, Y, Z = orbits.x(ts*u.yr)[0], orbits.y(ts*u.yr)[0], orbits.z(ts*u.yr)[0]
        #galactic coordinates of sun at the range of close encounter times ts
        Xsun, Ysun, Zsun  = orbits.x(ts*u.yr)[1], orbits.y(ts*u.yr)[1], orbits.z(ts*u.yr)[1]
        
        dx = X - Xsun
        dy = Y - Ysun
        dz = Z - Zsun
        
        #minimum encounter distance for each surrogate star with sun

        min_idx = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1).argmin() #index of the minimum encounter distance is stored
        enc_time[idx] = ts[min_idx] #time of close encounter for each surrogate    
        #distance of close encounter for each surrogate
        enc_dist[idx] = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1)[min_idx] 
        #phase coordinates of each surrogate at time of close encounter
        enc_Xt[idx] = np.array([X[min_idx], Y[min_idx], Z[min_idx]])
        #enc_V[idx] = np.array([X[min_idx], Y[min_idx], Z[min_idx]])
        
    return enc_time, enc_dist, init_xyz, enc_Xt #enc_V #orbit_surrogate, orbit_sun


'''

def orbit_integrate(x0, tph_lma, potential, n_step=100, method='odeint', radec=True, diff=False):
    x0_sun = np.array([0,0,0,0,0,0])
    X0 = np.array([x0, x0_sun])
    ts = np.linspace(0, 2*tph_lma)
    orbits = Orbit(X0, radec=radec, ro=8*u.kpc, vo=217*u.km/u.s, zo=0.010*u.kpc, solarmotion=[-11.1, 25, 7.25]*u.km/u.s)
    orbits.integrate(ts*u.yr, potential,method)
    
    #make a for loop to integrate each surrogate and orginal star with init condition (X0, ts)
    #enc_time = np.zeros(n) #store the close encounter time of each surrogate and orginal star
    #enc_dist = np.zeros(X0.shape[0]) #store the close encountere distance of each surrogate and orginal star
    #enc_vel = np.zeros(X0.shape[0]) #store the close encountere velocity of each surrogate and orginal star
    #enc_xyzv = np.zeros((1,6)) #coordinates of surrogates and orginal star in close encounter distance
    #init_xyzv = np.zeros((1,6)) #initial coordinates of surrogates and orginal star
    
    
    X, Y, Z = orbits.x(ts*u.yr)[0], orbits.y(ts*u.yr)[0], orbits.z(ts*u.yr)[0]
    Vx, Vy, Vz = orbits[0].vx(ts*u.yr), orbits[0].vy(ts*u.yr), orbits[0].vz(ts*u.yr)
    #X0, Y0, Z0 = orbits.x(0*u.yr)[0], orbits.y(0*u.yr)[0], orbits.z(0*u.yr)[0]
    #init_xyzv[i] = np.array([X[0], Y[0], Z[0], Vx[0], Vy[0], Vz[0]])
    
    Xsun, Ysun, Zsun  = orbits.x(ts*u.yr)[1], orbits.y(ts*u.yr)[1], orbits.z(ts*u.yr)[1]
    Vxsun, Vysun, Vzsun = orbits.vx(ts*u.yr)[1], orbits.vy(ts*u.yr)[1], orbits.vz(ts*u.yr)[1]
    
    dx = X - Xsun
    dy = Y - Ysun
    dz = Z - Zsun
    
    dvx = Vx - Vxsun
    dvy = Vy - Vysun
    dvz = Vz - Vzsun
    
    #minimum encounter distance for each surrogate star with sun
    
    #index of the minimum encounter distance is stored
    min_idx = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1).argmin() 
    #time of close encounter for each surrogate and it's orginal star
    enc_time = ts[min_idx] 
    #distance of close encounter for each surrogate
    #enc_dist[i] = np.linalg.norm(np.array([dx,dy,dz]).T,axis=1)[min_idx] #kpc
    #close encounter velocity of each surrogate and orginal star (km/s)
    #enc_vel[i] = np.linalg.norm(np.array([dvx,dvy,dvz]).T,axis=1)[min_idx] 
    #phase coordinates of each surrogate at time of close encounter
    if diff:
        init_dxyzv = np.array([dx[0], dy[0], dz[0], dvx[0], dvy[0], dvz[0]])
        enc_dxyzv = np.array([dx[min_idx], dy[min_idx], dz[min_idx], dvx[min_idx], dvy[min_idx], dvz[min_idx]])
        
        return enc_time, init_dxyzv, enc_dxyzv
    
    else:   
        init_xyzv = np.array([X[0], Y[0], Z[0], Vx[0], Vy[0], Vz[0]])
        enc_xyzv = np.array([X[min_idx], Y[min_idx], Z[min_idx], Vx[min_idx], Vy[min_idx], Vz[min_idx]])
            
        return enc_time, init_xyzv, enc_xyzv #enc_dist, enc_vel 



def LMA_params(data, surrogates=False):
    '''pmra and pmdec are in mas/yr and
       all velocities are in km/s '''
    #data = pd.DataFrame(data)
  
    if surrogates:
    
        pts = data['surrogates']
        '''
        parallax, pmra, pmdec, vr = np.concatenate((pts[:,2],[1/data['parallax']])),np.concatenate((pts[:,3],[data['pmra']])),\
                                                   np.concatenate((pts[:,4],[data['pmdec']])),\
                                                np.concatenate((pts[:,5],[data['radial_velocity']]))'''
        r, pmra, pmdec, vr = pts[:,2],pts[:,3],pts[:,4],pts[:,5]
                        
        c1=0.97779e9 
        c2=4.74047
    
        vt = c2*((pmra**2 + pmdec**2)**0.5*r)
        vlin_ph = (vt**2 + vr**2)**0.5 #km/s
    
        tlin_ph = -c1*r*(vr/vlin_ph**2) #units of yr
        dlin_ph = 10**3*r*(vt/vlin_ph)/1000 #units of kpc
 
    else:
        
        parallax, pmra, pmdec, vr = data['parallax'],data['pmra'],data['pmdec'],data['radial_velocity']
                                                
                        
        c1=0.97779e9 
        c2=4.74047
    
        vt = c2*((pmra**2 + pmdec**2)**0.5/parallax)
        vlin_ph = (vt**2 + vr**2)**0.5 #km/s
    
        tlin_ph = -c1*(1/parallax)*(vr/vlin_ph**2) #units of yr
        dlin_ph = (1/parallax)*(vt/vlin_ph) #units of kpc
    
    return tlin_ph, dlin_ph, vlin_ph




def cartesian_dist(x1, x2):
    ''' 
    Take position vector x1 and x2 as input and returns the distance between 
    them.
    
    '''
    ds = np.linalg.norm(x1-x2)
    
    return ds








def vel_sph_cart(ra,dec,par,pmra,pmdec,rv):
    
    au = 1.495978701e11
    yr = 365.25*24*3600 
    km = 1000
    
    theta = np.deg2rad(90 - dec)
    phi = np.deg2rad(ra)
    r = 1000/par
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    '''T = np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
               [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
               [np.cos(theta), -np.sin(theta), 0]
              ], dtype=object)'''
    vx = (x/r)*rv + np.cos(theta)*np.cos(phi)*-au/(km*yr)*pmdec/par - np.sin(phi)*au/(km*yr)*pmra/par
    vy = (y/r)*rv + np.cos(theta)*np.sin(phi)*-au/(km*yr)*pmdec/par + np.cos(phi)*au/(km*yr)*pmra/par
    vz = (z/r)*np.cos(theta)*rv-np.sin(theta)*-au/(km*yr)*pmdec/par
    #v_cart = T@[rv, -au/(km*yr)*pmdec/par, au/(km*yr)*pmra/par]
    #vx = np.array([np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)])*np.array([rv, -au/(km*yr)*pmdec/par, au/(km*yr)*pmra/par])
    #vy = np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)])*np.array([rv, -au/(km*yr)*pmdec/par, au/(km*yr)*pmra/par])
    #vz = np.array([np.cos(theta), -np.sin(theta), 0])*np.array([rv, -au/(km*yr)*pmdec/par, au/(km*yr)*pmra/par])
    return x,y,z,vx,vy,vz