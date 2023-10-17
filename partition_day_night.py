import numpy as np
from types import SimpleNamespace
import calendar as _cal
import sys as _sys


const = SimpleNamespace(
    # specific heat of air
    cp = 1010.0,

    # gas constant of dry air (J kg-1 K-1)
    r = 287.05,

    # latent heat (J kg-1)
    l = 2.5e6,

    # gamma: psychometric constant for specific humidity calculations (K-1)
    gamma = 0.0004,

    # MORECS stomatal resistance for grass (m s-1)
    rsc_day = [80.,80.,60.,50.,40.,60.,60.,70.,70.,70.,80.,80.],
    rsc_night = 2500.,

    # MORECS wet bare soil surface resistance m s-1
    rss = 100.0,

    # MORECS grass LAI
    lai = [2.,2.,3.,4.,5.,5.,5.,5.,4.,3.,2.5,2.0],

    # albedos
    # Dry bare soil (high-medium-low AWC) albedo MORECS
    albedo_s_dry = [0.1, 0.2, 0.3],
    # Wet bare soil (high-medium-low-low AWC) albedo MORECS
    albedo_s_wet = [0.05, 0.1, 0.15],
    # Full crop albedo MORECS grass
    albedo_c = 0.25,

    # MORECS emissivity
    emiss = 0.95,

    # Use MORECS canopy height
    canht = 0.15,

    # ratio mol weight water vapour/dry air
    epsilon = 0.622,

    # steam point temperature
    Ts = 373.15,

    # steam point pressure
    Ps = 101325.0,

    # Stefan-Boltzmann constant (W m-2 K-4)
    sigma = 5.670374419e-8,

    # coeffs for fit of saturated specific humidity (need citation)
    a_coeffs = [13.3185, -1.9760, -0.6445, -0.1299],

    # ground heat storage (W hr m-2) From MORECS v2 (Hydrol Mem 45)
    P = [-137., -75.,  30.,  167.,  236.,  252.,
          213.,  69., -85., -206., -256., -206.],

    # MORECS interception enhancement factor
    enhance = [1.0, 1.0, 1.2, 1.4, 1.6, 2.0, 2.0, 2.0, 1.8, 1.4, 1.2, 1.0],

    # time constants
    daylen_sec = 86400.0,
    daylen_hour = 24.0,
    hourlen_sec = 3600.0,
    
    # Longitude of perihelion
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    lonperi = 102.94719 * np.pi / 180.
)

def days_in_year(year, calendar='gregorian'):
    if calendar == 'gregorian':
        if _cal.isleap(year):
            return 366
        else:
            return 365
    elif calendar == '360_day':
        return 360
    else:
        sys.exit("Error: unknown calendar")

def sun_times(daynumber, year, lat, lon):

    delta = solar_declination(daynumber, year)

    # convert lat and lon to radians
    phi = lat * np.pi / 180.0
    lambdal = lon * np.pi /180.0

    tanp_tand = np.tan(phi) * np.tan(delta)

    # Perpetual day if tantan<-1.0, so sun is apparent from midnight to midnight
    # Perpetual night if tantan>1.0, so sun 'rises' and 'sets' at exactly
    # midday, resulting in a day length of zero
    omega_s = np.where(tanp_tand<-1.0, 0.0, 
                       np.where(tanp_tand>1, np.pi,
                                np.arccos(-1.0*tanp_tand)))

    time_up = (1.0 - (omega_s + lambdal)/np.pi) * consts.daylen_hour / 2.0
    time_down = (1.0 + (omega_s - lambdal)/np.pi) * consts.daylen_hour / 2.0

    return time_up, time_down


def time_max_temperature(time_up, time_down, tmax_offset):
    # From HCTN96 (Williams and Clark, 2014), but ultimately from IMOGEN code
    return 0.5 * (time_up+time_down) + tmax_offset * (time_down - time_up)


def solar_declination(daynumber, year):
    # From metprod. Equivalent to MORECS
    dy_in_yr = utils.days_in_year(year)

    return 0.409 * np.sin((2.0*np.pi*daynumber/dy_in_yr) - 1.39)


def get_daynight_lwdown(lwdown, dtr, tair, time_down, time_up, time_max):

    lwdown_day = lwdown_subdaily(lwdown, time_up, time_down, time_max,
                                 tair, dtr)
    lwdown_night = lwdown_subdaily(lwdown, time_down-const.daylen_hour,
                                   time_up, time_max, tair, dtr)

    return lwdown_day, lwdown_night


def lwdown_subdaily(lwdown_mean, t1, t2, time_max, tair, dtr):
    D = const.daylen_hour
    lwdown = lwdown_mean * (1.0 + ((np.sin(2.0*np.pi*(t2 - time_max)/D) -
                                    np.sin(2.0*np.pi*(t1 - time_max)/D)) *
                            dtr * D / (tair*np.pi*(t2-t1))))
    return lwdown

def tref(tair):
    # Tr = 1 - Ts/Ta
    tref = 1.0-(const.Ts/tair)
    return tref
    
def get_qair_from_relhum(relhum, tair, psurf):
    # Get reference temperature (K)
    tr = tref(tair)
    # Get qsat as function of tref (kg kg-1)
    qs = qsat(tr, psurf)
    return (relhum/100.0) * qs * psurf / (psurf + (qs * ((relhum/100.0)-1.0)))
    
def get_daynight_qair(relhum, tair_mean, psurf, tair_day, tair_night):
    ## assuming RH constant!
    #relhum = get_relhum_from_qair(qmean, tair_mean, psurf)
    qair_day = get_qair_from_relhum(relhum, tair_day, psurf)
    qair_night = get_qair_from_relhum(relhum, tair_night, psurf)
    return qair_day, qair_night

def get_daynight_tair(tair_mean, dtr, time_up, time_down, time_max):
    tair_day = tair_subdaily(tair_mean, time_up, time_down, time_max, dtr)
    tair_night = tair_subdaily(tair_mean, time_down-const.daylen_hour,
                               time_up, time_max, dtr)
    return tair_day, tair_night

def tair_subdaily(tair_mean, t1, t2, time_max, dtr):
    D = const.daylen_hour
    return tair_mean + (dtr/2.0) * (D / (2.0*np.pi*(t2 - t1))) * \
        (np.sin(2.0*np.pi*(t2 - time_max)/D) -
         np.sin(2.0*np.pi*(t1 - time_max)/D))
         
def ground_heat_flux_day(Rnet):
    Gd = 0.2 * Rnet
    return Gd

def ground_heat_flux_night(Gd, t1, t2, P):
    Gn = (P - ((t2-t1)*Gd))/(24-(t2-t1))
    return Gn        
 

def raero(wind10, h):
    z0 = 0.1 * h
    ra = 6.25 * np.log(10./z0) * np.log(6./z0) / wind10
    return ra


def canres_day(lai, rsc, rss):
    f = 0.7
    A = f**lai
    rs = 1./(((1.0-A)/rsc)+(A/rss))
    return rs


def canres_night(lai, rsc, rss):
    rs = 1.0 / ((lai/2500.)+(1.0/rss))
    return rs
