import numpy as np
import pandas as pd
import argparse
from types import SimpleNamespace
    
class SolarPosition():
    ## from https://gml.noaa.gov/grad/solcalc/calcdetails.html
    def __init__(self, timestamp, timezone=0):
        # time in hours past midnight, e.g. 2.3/24
        self.time = timestamp.hour + timestamp.minute/60 + timestamp.second/60/60
        self.timezone = timezone
        base_date = pd.to_datetime('01/01/1900', format='%d/%m/%Y')
        # add 2 from indexing/counting shift?
        self.jul_day = (2 + (pd.to_datetime(timestamp.date()) - base_date).total_seconds()/60/60/24 + 
            2415018.5 + self.time/24 - self.timezone/24)
        self.jul_century = (self.jul_day - 2451545)/36525
        
        # calculate location independent quantities
        self.geom_mean_anom_sun = 357.52911 + self.jul_century*(35999.05029 - 0.0001537*self.jul_century)    
        self.sun_eq_cent = (np.sin(np.deg2rad(self.geom_mean_anom_sun)) * 
            (1.914602 - self.jul_century*(0.004817 + 0.000014*self.jul_century)) + 
            np.sin(np.deg2rad(2*self.geom_mean_anom_sun)) * 
            (0.019993 - 0.000101*self.jul_century) + 
            np.sin(np.deg2rad(3*self.geom_mean_anom_sun)) * 0.000289
        )
        self.geom_mean_long_sun = np.mod(
            280.46646 + self.jul_century*(36000.76983 + self.jul_century*0.0003032), 360)
        self.sun_true_long = self.geom_mean_long_sun + self.sun_eq_cent
        self.sun_app_long = (self.sun_true_long - 0.00569 - 
            0.00478*np.sin(np.deg2rad(125.04 - 1934.136*self.jul_century)))

        self.mean_obliq_eliptic = (23 + (26 + (21.448 - self.jul_century*(46.815 + 
            self.jul_century*(0.00059 - self.jul_century*0.001813)))/60)/60)
        self.obliq_corr = (self.mean_obliq_eliptic + 
            0.00256*np.cos(np.deg2rad(125.04 - 1934.136*self.jul_century)))
        self.sun_declin = np.rad2deg(np.arcsin((
            np.sin(np.deg2rad(self.obliq_corr)) * 
            np.sin(np.deg2rad(self.sun_app_long))).clip(-1,1))
        ) 

        self.ecc_earth_orbit = (0.016708634 - 
            self.jul_century*(0.000042037 + 0.0000001267*self.jul_century))
        self.var_y = np.tan(np.deg2rad(self.obliq_corr/2)) * np.tan(np.deg2rad(self.obliq_corr/2))
        self.eq_of_time = (4*np.rad2deg(self.var_y * np.sin(2*np.deg2rad(self.geom_mean_long_sun)) - 
            2*self.ecc_earth_orbit * np.sin(np.deg2rad(self.geom_mean_anom_sun)) + 
            4*self.ecc_earth_orbit * self.var_y * 
                np.sin(np.deg2rad(self.geom_mean_anom_sun))*np.cos(2*np.deg2rad(self.geom_mean_long_sun)) - 
            0.5*self.var_y*self.var_y * np.sin(4*np.deg2rad(self.geom_mean_long_sun)) - 
            1.25*self.ecc_earth_orbit*self.ecc_earth_orbit * np.sin(2*np.deg2rad(self.geom_mean_anom_sun)))
        )
    
    def calc_solar_angles(self, lats, lons, arr=False):
        true_solar_time = np.mod((self.time/24) * 1440 + self.eq_of_time + 
            4*lons - 60*self.timezone, 1440)
        hour_angle = lats.copy()
        if arr:
            hour_angle.fill(0)
            mask = (true_solar_time/4) < 0
            hour_angle[mask] = true_solar_time[mask]/4 + 180
            hour_angle[~mask] = true_solar_time[~mask]/4 - 180
        else:
            hour_angle.values.fill(0)
            mask = (true_solar_time.values/4) < 0        
            hour_angle.values[mask] = true_solar_time.values[mask]/4 + 180
            hour_angle.values[~mask] = true_solar_time.values[~mask]/4 - 180
        
        # solar zenith angle in degrees from vertical
        rlat = np.deg2rad(lats)
        rsundeclin = np.deg2rad(self.sun_declin)        
        self.solar_zenith_angle = np.rad2deg(
            np.arccos((np.sin(rlat) * np.sin(rsundeclin) + 
                    np.cos(rlat) * np.cos(rsundeclin) * 
                        np.cos(np.deg2rad(hour_angle))).clip(-1,1))
        )
        
        # solar azimuth angle in degrees clockwise from north
        self.solar_azimuth_angle = lats.copy()
        rzenith = np.deg2rad(self.solar_zenith_angle)
        if arr:
            self.solar_azimuth_angle.fill(0)
            mask = hour_angle>0
            self.solar_azimuth_angle[mask] = np.mod(
                np.rad2deg(np.arccos((
                    ((np.sin(rlat[mask]) * np.cos(rzenith[mask])) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat[mask]) * np.sin(rzenith[mask]))).clip(-1,1))) + 180, 360
            )
            self.solar_azimuth_angle[~mask] = np.mod(
                540 - np.rad2deg(np.arccos((
                    ((np.sin(rlat[~mask]) * np.cos(rzenith[~mask])) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat[~mask]) * np.sin(rzenith[~mask]))).clip(-1,1))), 360
            )
            # fix nans at 360
            mask = np.isnan(self.solar_azimuth_angle)
            self.solar_azimuth_angle[mask] = 0.            
        else:
            self.solar_azimuth_angle.values.fill(0)
            mask = hour_angle.values>0
            self.solar_azimuth_angle.values[mask] = np.mod(
                np.rad2deg(np.arccos((
                    ((np.sin(rlat.values[mask]) * np.cos(rzenith.values[mask])) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat.values[mask]) * np.sin(rzenith.values[mask]))).clip(-1,1))) + 180, 360
            )
            self.solar_azimuth_angle.values[~mask] = np.mod(
                540 - np.rad2deg(np.arccos((
                    ((np.sin(rlat.values[~mask]) * np.cos(rzenith.values[~mask])) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat.values[~mask]) * np.sin(rzenith.values[~mask]))).clip(-1,1))), 360
            )
            # fix nans at 360
            mask = self.solar_azimuth_angle.isnull()
            self.solar_azimuth_angle.values[mask] = 0.
        
        # solar elevation in degrees from horizontal
        solar_elevation = 90 - self.solar_zenith_angle
        atmos_refraction = lats.copy()
        rsolelev = np.deg2rad(solar_elevation)
        if arr:
            atmos_refraction.fill(0)
            mask = np.bitwise_and(solar_elevation > 5, solar_elevation <= 85)
            atmos_refraction[mask] = (
                58.1 / np.tan(rsolelev[mask]) - 
                0.07 / np.power(np.tan(rsolelev[mask]), 3) +
                0.000086 / np.power(np.tan(rsolelev[mask]), 5)
            )
            mask2 = np.bitwise_and(solar_elevation <= 5, solar_elevation > -0.575)
            atmos_refraction[mask2] = (
                1735 + solar_elevation[mask2]*(-518.2 + 
                    solar_elevation[mask2]*(103.4 + 
                        solar_elevation[mask2]*(-12.79 + 
                            solar_elevation[mask2]*0.711)))
            )
            mask3 = np.bitwise_or(mask, mask2)
            atmos_refraction[~mask3] = -20.772 / np.tan(rsolelev[~mask3])
        else:
            atmos_refraction.values.fill(0)
            mask = np.bitwise_and(solar_elevation > 5, solar_elevation <= 85)
            atmos_refraction.values[mask] = (
                58.1 / np.tan(rsolelev.values[mask]) - 
                0.07 / np.power(np.tan(rsolelev.values[mask]), 3) +
                0.000086 / np.power(np.tan(rsolelev.values[mask]), 5)
            )
            mask2 = np.bitwise_and(solar_elevation <= 5, solar_elevation > -0.575)
            atmos_refraction.values[mask2] = (
                1735 + solar_elevation.values[mask2]*(-518.2 + 
                    solar_elevation.values[mask2]*(103.4 + 
                        solar_elevation.values[mask2]*(-12.79 + 
                            solar_elevation.values[mask2]*0.711)))
            )
            mask3 = np.bitwise_or(mask, mask2)
            atmos_refraction.values[~mask3] = -20.772 / np.tan(rsolelev.values[~mask3])
        atmos_refraction /= 3600
        self.solar_elevation = solar_elevation + atmos_refraction

    def calc_solar_angles_return(self, lat, lon):
        true_solar_time = np.mod((self.time/24) * 1440 + self.eq_of_time + 
            4*lon - 60*self.timezone, 1440)        
        if (true_solar_time/4 < 0):
            hour_angle = true_solar_time/4 + 180
        else:
            hour_angle = true_solar_time/4 - 180
        
        # solar zenith angle in degrees from vertical
        rlat = np.deg2rad(lat)
        rsundeclin = np.deg2rad(self.sun_declin)
        rhourangle = np.deg2rad(hour_angle)
        solar_zenith_angle = np.rad2deg(
            np.arccos((np.sin(rlat) * np.sin(rsundeclin) + 
                    np.cos(rlat) * np.cos(rsundeclin) * 
                        np.cos(rhourangle)).clip(-1,1))
        )
        
        # solar azimuth angle in degrees clockwise from north
        rzenith = np.deg2rad(solar_zenith_angle)
        if hour_angle>0:
            solar_azimuth_angle = np.mod(
                np.rad2deg(np.arccos((
                    ((np.sin(rlat) * np.cos(rzenith)) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat) * np.sin(rzenith))).clip(-1,1))) + 180, 360
            )
        else:
            solar_azimuth_angle = np.mod(
                540 - np.rad2deg(np.arccos((
                    ((np.sin(rlat) * np.cos(rzenith)) - 
                        np.sin(rsundeclin)) / 
                    (np.cos(rlat) * np.sin(rzenith))).clip(-1,1))), 360
            )
        # fix nans at 360
        if np.isnan(solar_azimuth_angle):
            solar_azimuth_angle = 0
        
        # solar elevation in degrees from horizontal
        solar_elevation = 90 - solar_zenith_angle
        rsolelev = np.deg2rad(solar_elevation)
        if (solar_elevation > 5 and solar_elevation <= 85):        
            atmos_refraction = (
                58.1 / np.tan(rsolelev) - 
                0.07 / np.power(np.tan(rsolelev), 3) +
                0.000086 / np.power(np.tan(rsolelev), 5)
            )
        elif (solar_elevation <= 5 and solar_elevation > -0.575):
            atmos_refraction = (
                1735 + solar_elevation*(-518.2 + 
                    solar_elevation*(103.4 + 
                        solar_elevation*(-12.79 + 
                            solar_elevation*0.711)))
            )
        else:        
            atmos_refraction = -20.772 / np.tan(rsolelev)
        atmos_refraction /= 3600
        solar_elevation = solar_elevation + atmos_refraction
        return SimpleNamespace(
            solar_zenith_angle = float(solar_zenith_angle),
            solar_azimuth_angle = float(solar_azimuth_angle),
            solar_elevation = float(solar_elevation)
        )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date_time", help = "in format dd/mm/yyyy HH:MM:SS")
    parser.add_argument("timezone", help = "in hours relative to UTC/GMT")
    parser.add_argument("latitude", help = "in decimal degrees north")
    parser.add_argument("longitude", help = "in decimal degrees east")
    args = parser.parse_args()
    
    sp = SolarPosition(pd.to_datetime(args.date_time), timezone=float(args.timezone))
    out = sp.calc_solar_angles_return(float(args.latitude), float(args.longitude))
    print(out)
