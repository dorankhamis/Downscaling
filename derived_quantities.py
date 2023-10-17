import numpy as np
from partition_day_night import *

def calc_PET(TA, PA, SWIN, LWIN, WS, RH, DTR=None, docorr=True, daynight=False):
    ''' potential evapotranspiration [kg m−2 day−1]
    TA: air temperature [K]
    PA: air pressure [Pa]
    SWIN: shortwave radiation incoming [W m-2]
    LWIN: longwave radiation incoming [W m-2]
    WS: 10m wind speed [m s-1]
    RH: relative humidity [%]
    '''
    
    lai = const.lai[month-1]
    
    if daynight:
        ## timings
        tmax_offset = 0.15
        time_up, time_down = solar.sun_times(dayno, year, lat, lon)
        time_max = solar.time_max_temperature(time_up, time_down, tmax_offset)
        day_hours = time_down - time_up
        night_hours = const.daylen_hour - day_hours

        ## partition met vars into day, night
        # Get day/night air temperature
        tair = get_daynight_tair(TA, DTR, time_up, time_down, time_max)
        
        # Get day/night humidity, assuming relative humidity is constant...
        # different to our approach!
        qair = get_daynight_qair(RH, TA, PA, TA_dn[0], TA_dn[1])        
        
        # Get day/night SW down. All of the SW is in the day and none at night (ignoring twilight)
        swdown = (SWIN * const.daylen_hour / day_hours, 0)

        # Get day/night LW down. Uses the assumed sinusoid as in JULES
        lwdown = get_daynight_lwdown(LWIN, DTR, TA, time_down, time_up, time_max)
        
        ## different constants for day, night
        # Canopy resistance        
        rsc_day = const.rsc_day[month-1]
        rs_day = canres_day(lai, rsc_day, const.rss)
        rs_night = canres_night(lai, const.rsc_night, const.rss)
        rs = [rs_day, rs_night]

        # Time period length
        Ds = [const.hourlen_sec*day_hours, const.hourlen_sec*night_hours]
    
    
    else:
        tair = [TA,]
        qair = [get_qair_from_relhum(RH, TA, PA),]
        #qair = [q_a,]
        swdown = [SWIN,]
        lwdown = [LWIN,]

        # canopy resistance        
        rsc = const.rsc_day[month-1]
        rs = [canres_day(lai, rsc, const.rss),]

        # Time period length
        Ds = [const.daylen_sec,]
        
    
    
    ## required variables
    t_d = const.daylen_sec # [s day-1] timescale in seconds (daily) 
    lamb = const.l # [J kg-1] latent heat of evaporation
    c_p = const.cp # [J kg-1 K-1] specific heat capacity of air
    gam = const.gamma # [K-1] psychrometric constant    
    r_s = 70.0 # [s m-1] stomatal resistance
    rho_a = calc_rho_a(TA, PA) # air density [kg m-3]
    r_a = calc_r_a(WS) # aerodynamic resistance [s m-1] (of reference crop)
    A = calc_A(SWIN, LWIN, TA, albe=const.albedo_c, emis=const.emiss) # available energy [W m-2] (replacing T_surf with TA)
    # saturated values
    e_s = calc_e_s(TA) # saturated vapour pressure [Pa]
    q_s = calc_q(PA, e_s) # saturated specific humidity [kg kg-1]
    Delta = calc_Delta(TA, PA, q_s, e_s) # derivative of saturated specific humidity with respect to temperature
    # actual values
    e = calc_e(RH, PA, e_s) # (partial) vapour pressure [Pa]
    q_a = calc_q(PA, e) # specific humidity [kg kg-1]    
    
    ## join to calculate PET
    prefct = (t_d / lamb)
    radiative_term = Delta * A
    if docorr:
        corr = 4. * const.emiss * const.sigma * TA**3
        denom = Delta + gam * (1. + r_s/r_a) * (1. + corr*r_a/(c_p*rho_a))
        aerodynamic_term = (c_p * rho_a / r_a) * (q_s - q_a) * (1. + corr*r_a/(c_p*rho_a))        
    else:
        denom = Delta + gam * (1. + r_s / r_a)
        aerodynamic_term = (c_p * rho_a / r_a) * (q_s - q_a)

    pet = prefct * (radiative_term + aerodynamic_term) / denom
    
    # Calculate Penman-Monteith potential evaporation
    # PE =  Del*A + cp*rhoair*(qsat-qair)*(1+b*ra/(rhoair*cp))/ra
    #      ----------------------------------
    #          Del + gamma*(1+rs/ra)*(1+b*ra/(rhoair*cp))
    # assume (1+b*ra/(rhoair*cp) ~= 1
    # PE = ((D*A) + (cpra*qdef/ra)) / \
    #   (D + (const.gamma*(1.0+(rs/ra))))
    # or with docorr
    # PE = ((D*A) + (cpra*qdef*(1.0+(corr*ra/cpra))/ra)) / \
    #   (D + (const.gamma*(1.0+(corr*ra/cpra))*(1.0+(rss/ra))))
    
    
    bits = dict(
        rs = r_s,
        rho_a = rho_a,
        ra = r_a,
        A = A,
        e_s = e_s,
        q_s = q_s,
        Delta = Delta,
        e = e,
        q_a = q_a,
        pet  = pet        
    )
    return pet, bits
    
def calc_e_s(TA):
    ''' saturated vapour pressure [Pa]
    TA: air temperature [K]    
    '''
    p_sp = const.Ps # [Pa] steam point pressure
    T_sp = const.Ts # [K] steam point temperature
    a = const.a_coeffs # empirical coeffs (Richards, 1971)
    exponent = 0
    for i in range(len(a)):
        exponent += a[i] * (1 - T_sp / TA)**(i+1)
    e_s = p_sp * np.exp(exponent)
    return e_s
    
def calc_e(RH, PA, e_s):
    ''' saturated vapour pressure [Pa]
    RH: relative humidity [%]
    PA: air pressure [Pa]
    e_s: saturated vapour pressure [Pa]
    
    This uses RH/100 == w / w_s (ratio of mixing ratio to saturated mixing ratio)
    and w / w_s = e (PA - e_s) / (e_s (PA - e))    
    '''
    e = RH/100. * PA / (PA/e_s + RH/100. - 1)
    return e
    
def calc_q(PA, e):
    ''' specific humidity [kg kg-1]
    PA: air pressure [Pa]
    e: vapour pressure [Pa]
    '''
    eps = const.epsilon # mass ratio of water and dry air
    q = eps * e / (PA - (1 - eps) * e)
    return q

def calc_Delta(TA, PA, q_s, e_s):
    ''' derivative of saturated specific humidity with respect to temperature
    TA: air temperature [K]
    PA: air pressure [Pa]
    q_s: saturated specific humidity [kg kg-1]
    e_s: saturated vapour pressure [Pa]
    '''
    eps = const.epsilon # mass ratio of water and dry air
    p_sp = const.Ps # [Pa] steam point pressure
    T_sp = const.Ts # [K] steam point temperature
    a = const.a_coeffs # empirical coeffs (Richards, 1971)
    summ = 0
    for i in range(len(a)):
        summ += (i+1) * a[i] * (1 - T_sp / TA)**(i)
    Delta = (T_sp / TA**2) * PA * q_s / (PA - (1 - eps) * e_s)
    Delta *= summ
    return Delta

def calc_A(SWIN, LWIN, T_surf, albe=0.25, emis=0.95):
    ''' available energy [W m-2]
    SWIN: shortwave radiation incoming [W m-2]
    LWIN: longwave radiation incoming [W m-2]
    T_surf: temperature of surface (in CHESS just replaced by air temp) [K]
    albe: albedo of surface
    emis = 0.92 # emissivity of surface
    '''
    sb_sigma = const.sigma # [W m−2 K−4] Stefan-Boltzmann constant
    A = (1 - albe) * SWIN + emis * (LWIN - sb_sigma*T_surf**4)
    return A

def calc_rho_a(TA, PA):
    ''' air density [kg m-3]
    TA: air temperature [K]
    PA: air pressure [Pa]
    '''
    r = const.r # [J kg-1 K-1] gas constant of air    
    rho_a = PA / (r * TA)
    return rho_a
    
def calc_r_a(WS):
    ''' aerodynamic resistance [s m-1]
    of a reference crop with canopy height 0.12m
    WS: 10m wind speed [m s-1]
    '''
    r_a = 278. / WS
    return r_a


TA = 5.1 + 273.15
PA = 1010 * 100
SWIN = 600
LWIN = 300
WS = 3.1
RH = 87.0
DTR = 8.5

TA = np.array([TA])
PA = np.array([PA])
SWIN = np.array([SWIN])
LWIN = np.array([LWIN])
WS = np.array([WS])
RH = np.array([RH])
DTR = np.array([DTR])

PE_cor, pe_bits_cor = calc_PET(TA, PA, SWIN, LWIN, WS, RH, docorr=True)
PE_raw, pe_bits_raw = calc_PET(TA, PA, SWIN, LWIN, WS, RH, docorr=False)

import alternate_quantities as am
import hydro_pe.src.pet.make_pet as em
am_PE_cor, am_bits_cor = am.get_pet(TA, RH, PA, SWIN, LWIN, WS, docorr=True)
am_PE_raw, am_bits_raw = am.get_pet(TA, RH, PA, SWIN, LWIN, WS, docorr=False)

em_PE_cor, em_bits_cor = em.hydro_pe(TA, PA, SWIN, LWIN, WS, RH, docorr=True)
em_PE_raw, em_bits_raw = em.hydro_pe(TA, PA, SWIN, LWIN, WS, RH, docorr=False)
em_PE_cor_dn, em_bits_cor_dn = em.hydro_pe(TA, PA, SWIN, LWIN, WS, RH, docorr=True, daynight=True)

