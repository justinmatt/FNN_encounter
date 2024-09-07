# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 01:27:54 2023

@author: ACER
"""

import numpy as np
import pandas as pd

def cov_mat(data):
    k00 = data['ra_error']**2 #unit of mas
    k01 = (data['ra_dec_corr']*data['ra_error']*data['dec_error'])
    k02 = (data['ra_parallax_corr']*data['ra_error']*data['parallax_error'])
    k03 = (data['ra_pmra_corr']*data['ra_error']*data['pmra_error'])
    k04 = (data['ra_pmdec_corr']*data['ra_error']*data['pmdec_error'])
    k05 = 0

    k11 = (data['dec_error']**2) #unit of mas
    k12 = (data['dec_parallax_corr']*data['dec_error']*data['parallax_error'])
    k13 = (data['dec_pmra_corr']*data['dec_error']*data['pmra_error'])
    k14 = (data['dec_pmdec_corr']*data['dec_error']*data['pmdec_error'])
    k15 = 0

    k22 = (data['parallax_error']**2) #unit of mas
    k23 = (data['parallax_pmra_corr']*data['parallax_error']*data['pmra_error'])
    k24 = (data['parallax_pmdec_corr']*data['parallax_error']*data['pmdec_error'])
    k25 = 0

    k33 = (data['pmra_error']**2) #unit of mas/yr
    k34 = (data['pmra_pmdec_corr']*data['pmra_error']*data['pmdec_error'])
    k35 = 0

    k44 = (data['pmdec_error']**2) #unit of mas/yr
    k45 = 0

    k55 = (data['radial_velocity_error']**2) #km/s
    
    
    cov_mat = np.array([[k00, k01, k02, k03, k04, k05],
                        [k01, k11, k12, k13, k14, k15],
                        [k02, k12, k22, k23, k24, k25],
                        [k03, k13, k23, k33, k34, k35],
                        [k04, k14, k24, k34, k44, k45],
                        [k05, k15, k25, k35, k45, k55]]) 
    
    return cov_mat



def gen_surrogates(data,sampling_size=800):
    dec = data['dec']
    ra = data['ra']
    parallax = data['parallax'] #mas
    pmra = data['pmra'] #mas/yr
    pmdec = data['pmdec'] #mas/yr
    RV = data['radial_velocity'] #km/s
    #covmat = data['cov_matrix']
    covmat = cov_mat(data)
    X0 = [0,0,0,0,0,0] #to generate surrogates in orgin of phase space coordinates
    #gaussian with mean zero
    surrogates_00 = np.random.multivariate_normal(X0,covmat,size=sampling_size,check_valid='ignore') 
    surrogates_00[:,0] = surrogates_00[:,0]/3600000
    surrogates_00[:,1] = surrogates_00[:,1]/3600000
   
    X = [ra, dec, parallax, pmra, pmdec, RV] #phase space coordinates of the star
    surrogates = surrogates_00 + X #add the nominal phase space coordinate values to every surrogate star
    surrogates[:,2] = 1/surrogates[:,2] #store parallax values as radial distance in kpc 
    #surrogates[:,0] = surrogates[:,0]/3600000#convert mas back to degrees
    #surrogates[:,1] = surrogates[:,1]/3600000 #convert mas back to degrees
    return surrogates