import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import datetime
from skimage.draw import line
from pathlib import Path
#from joblib import Parallel, delayed, parallel_backend
#from dask.distributed import Client, LocalCluster

from solar_position import SolarPosition

class TerrainShading():
    def __init__(self):
        pass

    def correct_azi_for_local_north(self, oi, oj, azi_raw, lats, lons, ys, xs):
        # find local north
        if oi == lats.shape[0]-1: use_oi = oi
        else: use_oi = oi + 1
        
        if oj==0:
            lat_choices = lats[use_oi, oj:(oj+2)]
            lon_choices = lons[use_oi, oj:(oj+2)]
            lon_inds = [oj, oj+1]
        elif oj == lons.shape[1]-1:
            lat_choices = lats[use_oi, (oj-1):(oj+1)]
            lon_choices = lons[use_oi, (oj-1):(oj+1)]
            lon_inds = [oj-1, oj]             
        else:
            lat_choices = lats[use_oi, (oj-1):(oj+2)]
            lon_choices = lons[use_oi, (oj-1):(oj+2)]
            lon_inds = [oj-1, oj, oj+1]
        
        home_lon = lons[use_oi, oj]
        lon_weights = np.abs(lon_choices - home_lon)
        inds = np.argsort(lon_weights)        
        node_weights = 1 - (lon_weights[inds[:2]] / np.sum(lon_weights[inds[:2]]))
        
        loc_north_x = xs[lon_inds[inds[0]]] * node_weights[0] + \
            xs[lon_inds[inds[1]]] * node_weights[1]

        home_x = xs[oj]
        home_y = ys[use_oi-1]
        loc_north_y = ys[use_oi]
        loc_north_theta = np.rad2deg(np.arctan(abs(home_x - loc_north_x) / abs(home_y - loc_north_y)))
        # but which direction is this angle?    
        if 2 in inds[:2]:
            # north is to the "right"
            return float(azi_raw + loc_north_theta)
        elif 0 in inds[:2]:
            # north is to the "left"
            return float(azi_raw - loc_north_theta)
        else:
            print('broken!')
            return azi_raw

    def calc_end_point(self, azi, start_point, map_size):
        # i, j direction vector
        vec_dir = np.array([np.cos(np.deg2rad(azi)), np.sin(np.deg2rad(azi))])
        mults = np.array([1,1])
        for ii in range(2):
            if vec_dir[ii]<0:
                mults[ii] = start_point[ii]/(-vec_dir[ii])
            else:
                mults[ii] = (map_size[ii] - start_point[ii]) / vec_dir[ii]
        mult = np.min(mults.astype(int))
        return start_point + (mult * vec_dir).astype(np.int32)    

    def get_line_pixel_stats(self, start_point, end_point, mask, elevation):        
        pix_i, pix_j = line(start_point[0], start_point[1], end_point[0], end_point[1])
        pix_i = pix_i.clip(0, mask.shape[0]-1)
        pix_j = pix_j.clip(0, mask.shape[1]-1)
        mask[pix_i, pix_j] = 255
        dists = np.sqrt(np.sum(np.square(start_point - 
            np.hstack([pix_i[...,None], pix_j[...,None]])), axis=1))
        elevs = elevation[pix_i, pix_j]
        sort_inds = np.argsort(dists)
        dists = dists[sort_inds] * 1000  # dist km to m
        elevs = elevs[sort_inds] # elev in m
        rel_elevs = elevs - elevation[start_point[0], start_point[1]]
        return dists, rel_elevs, mask

    def check_shaded(self, dists, rel_elevs, solar_elev):
        if solar_elev<0:
            return 1
        max_elev_change = np.nanmax(rel_elevs)
        tan_sol_elev = np.tan(np.deg2rad(solar_elev))
        shaded = 0
        for k in range(1,len(dists)):
            if np.isnan(rel_elevs[k]):
                continue
            eff_solar_height = tan_sol_elev * dists[k]
            if eff_solar_height > max_elev_change:
                # as distance is ordered
                break
            if eff_solar_height > rel_elevs[k]:
                # not shaded
                continue
            else:
                shaded = 1
                break
        return shaded

    def inner_calc(self, oi, oj, solar_azimuthal_angle, solar_elevation,
                    lats, lons, ys, xs, zeromask, elevation):
        
        if solar_elevation<0:
            return 1, np.array([0,0]), np.array([0,0])
        
        start_point = np.array([oi, oj])
                
        # correct azimuthal angle from local north
        azi = self.correct_azi_for_local_north(
            oi, oj, solar_azimuthal_angle,
            lats, lons, ys, xs
        )
                
        # find end point of line
        end_point = self.calc_end_point(azi, start_point, zeromask.shape)
        end_point[0] = end_point[0].clip(0, zeromask.shape[0]-1)
        end_point[1] = end_point[1].clip(0, zeromask.shape[1]-1)
        
        # draw line, find pixels passed through, calc distance and relative elev
        dists, rel_elevs, _ = self.get_line_pixel_stats(
            start_point, end_point, zeromask.copy(), elevation)
        
        # is pixel shaded?                
        return self.check_shaded(dists, rel_elevs, solar_elevation), start_point, end_point

    def calculate_terrain_shading(self, x_inds, y_inds, sp, grid, elevation):
        zeromask = np.zeros(elevation.shape, dtype=np.uint8)
        ojs, ois = np.meshgrid(x_inds, y_inds)
        shade_mask = np.zeros(ojs.shape, dtype=np.uint8)
        start_end_points = []
        for i in range(ojs.shape[0]):
            for j in range(ojs.shape[1]):
                oi = ois[i,j]
                oj = ojs[i,j]
                shaded, st_pnt, en_pnt = self.inner_calc(
                    oi, oj, sp.solar_azimuth_angle.values[oi, oj],
                    sp.solar_elevation.values[oi, oj], 
                    grid.lat.values, grid.lon.values,
                    grid.y.values, grid.x.values,
                    zeromask, elevation.values
                )
                shade_mask[i,j] = shaded
                start_end_points.append((st_pnt, en_pnt))
        return shade_mask, start_end_points

    def calc_svf(slope, aspect, nbr_Hphi, nbr_phi):
        # sky view factor for scaling diffuse short wave radiation
        return (np.cos(np.deg2rad(slope)) * np.sin(nbr_Hphi)**2 +
            np.sin(np.deg2rad(nbr_phi) - np.deg2rad(aspect)) *
            (nbr_Hphi - np.sin(nbr_Hphi)*np.cos(nbr_Hphi)))

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        #parser.add_argument("day", help = "day to run, 0-indexed")
        parser.add_argument("chunk_num", help = "id of run to find spatial chunk, 0-indexed")
        parser.add_argument("of_n_chunks", help = "total number of chunks")
        args = parser.parse_args()
        #day = int(args.day)
        chunk_num = int(args.chunk_num)
        of_n_chunks = int(args.of_n_chunks)
    except:
        print('Not running as script')
        #day = 275
        chunk_num = 0
        of_n_chunks = 8000
    
    av_over_hour = True
    new = True
    
    home_data_dir = '/home/users/doran/data_dump/'
    hj_base = '/gws/nopw/j04/hydro_jules/'
    hj_ancil_fldr = hj_base + '/data/uk/ancillaries/'        
    outdir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/terrain_shading/'
    Path(outdir).mkdir(exist_ok=True, parents=True)
    
    #grid = xr.open_dataset(hj_ancil_fldr+'/chess_lat_lon.nc')
    grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
    grid = grid.load() # force load data

    #elevation = xr.open_dataset(hj_ancil_fldr+'/uk_ihdtm_topography+topoindex_1km.nc')
    elevation = xr.open_dataset(home_data_dir + '/height_map/topography_bng_1km.nc')
    elevation = elevation.drop(['topi','stdtopi','fdepth','area','stdev','slope','aspect']).load()    
    elevation = elevation.elev

    ## which subset of pixels are we running?
    #ys, xs = np.where(grid.landfrac)
    ys, xs = np.where(grid.landfrac==0)
    
    fname = 'sea_shading_mask' # 'shading_mask'
                
    # take only a chunk of the y,x indices
    chunk_size = len(ys) // of_n_chunks
    leftover = len(ys) - chunk_size*of_n_chunks
    chunk_sizes = np.repeat(chunk_size, of_n_chunks)
    chunk_sizes[-leftover:] += 1
    
    csum = np.hstack([0, np.cumsum(chunk_sizes)])    
    ys = ys[csum[chunk_num]:csum[chunk_num+1]]
    xs = xs[csum[chunk_num]:csum[chunk_num+1]]                
    
    zeromask = np.zeros(elevation.shape, dtype=np.uint8)
    minute_resolution = 10
    for day in range(366):
        out_fname = outdir+f'{fname}_day_{day}_chunk{chunk_num}.npy'
        merged_fname = outdir+f'{fname}_day_{day}_merged_to_{of_n_chunks-1}.npy'
        if not (Path(out_fname).exists() or Path(merged_fname).exists()):
            print(f'Starting day {day}')
            daystamp = pd.to_datetime('2016-01-01') + datetime.timedelta(days=day)
            shading = np.zeros((24, len(ys)), dtype=np.float32)
            ts = TerrainShading()
            if not av_over_hour:
                for hr in range(24):
                    print(f'Doing hour {hr}')            
                    timestamp = daystamp + datetime.timedelta(hours=hr)
                    sp = SolarPosition(timestamp, timezone=0) # utc
                    for i in range(len(ys)):
                        if i%25000 == 0:
                            print(i)
                        sp_angles = sp.calc_solar_angles_return(
                            grid.lat.values[ys[i],xs[i]],
                            grid.lon.values[ys[i],xs[i]]
                        )
                        shad, st, en = ts.inner_calc(
                            ys[i], xs[i], sp_angles.solar_azimuth_angle, 
                            sp_angles.solar_elevation,
                            grid.lat.values, grid.lon.values,
                            grid.y.values, grid.x.values,
                            zeromask, elevation.values
                        )
                        shading[hr,i] = shad
            else:
                for hr in range(24):
                    print(f'Doing hour {hr}')                
                    for mn in range(0, 60, minute_resolution):
                        # accumulate timestamp over the last hour                        
                        timestamp = daystamp + datetime.timedelta(hours=hr) - datetime.timedelta(minutes=mn)
                        sp = SolarPosition(timestamp, timezone=0) # utc
                        for i in range(len(ys)):
                            sp_angles = sp.calc_solar_angles_return(
                                grid.lat.values[ys[i],xs[i]],
                                grid.lon.values[ys[i],xs[i]]
                            )
                            shad, st, en = ts.inner_calc(
                                ys[i], xs[i], sp_angles.solar_azimuth_angle, 
                                sp_angles.solar_elevation,
                                grid.lat.values, grid.lon.values,
                                grid.y.values, grid.x.values,
                                zeromask, elevation.values
                            )
                            shading[hr,i] += shad # accumulate minute shading
                shading /= (60 // minute_resolution) # average the minute shading over the hour
            np.save(out_fname, shading.astype(np.float32))
            print(f'Done day {day}')
        else:
            print(f'Skipping  day {day} chunk {chunk_num}')
            continue        
        
    ## these files should be joined column-wise in chunk order
    ## to recreate the same index ordering as np.where(landfrac)
    if False:
        of_n_chunks = 8000
        outdir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/terrain_shading/'
        day = 0
        successfully_merged = []
        fnames = 'sea_shading_mask'
        
        shading_cur = np.load(outdir + f'/{fnames}_day_{day}_chunk0.npy')
        successfully_merged.append(0)
        for ch in range(1, of_n_chunks):
            try:
                shading_chunk = np.load(outdir + f'/{fnames}_day_{day}_chunk{ch}.npy')
                shading_cur = np.hstack([shading_cur, shading_chunk])
                successfully_merged.append(ch)
            except:
            # reached gap in contiguous chunks
                break
        
        # # deal with wrong order of hours (subtracting rather than adding (fixed in new runs)
        # shading_cur = shading_cur[1:,:][::-1,:]
        # # add fully shaded zero-hour at correct side of the day!
        # shading_cur = np.vstack([np.ones((1,shading_cur.shape[1]), dtype=np.float32), shading_cur])
        # # roll day back
        # if day==0: true_day = 366
        # else: true_day = day - 1
        
        # save current merged array
        np.save(outdir+f'{fnames}_day_{true_day}_merged_to_{successfully_merged[-1]}.npy', shading_cur)
        
        if False:
            # visualise the shade map!
            import xarray as xr
            grid = xr.open_dataset('/home/users/doran/data_dump/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
            grid = grid.load() # force load data
            
            shademap = grid.landfrac.copy()
            shademap.values[:,:] = 0
            shademap.values[np.where(grid.landfrac.values > 0)] = 1 - shading_cur[9,:] 
            shademap.plot(); plt.show()
              
        # remove files that have been successfully merged... but only after we have got everything! 
        if successfully_merged[-1] == of_n_chunks-1:
            for ch in successfully_merged:
                os.remove(outdir + f'/{fnames}_day_{day}_chunk{ch}.npy')
            
            
                
    if False:
        ## Calculate sky view factor for each pixel
        ##############################
        home_data_dir = '/home/users/doran/data_dump/'
        hj_base = '/gws/nopw/j04/hydro_jules/'
        hj_ancil_fldr = hj_base + '/data/uk/ancillaries/'        
                
        grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
        grid = grid.load() # force load data
        
        height_grid = xr.open_dataset(home_data_dir + '/height_map/topography_bng_1km.nc')
        height_grid = height_grid.load()    
        
        ## sky view factor
        #ys, xs = np.where(grid.landfrac > 0)
        ys, xs = np.where(grid.landfrac == 0)
        
        fname = 'sea_sky_view_factor' # 'sky_view_factor'
        
        height_grid.elev.values[np.isnan(height_grid.elev.values)] = 0
        height_grid.aspect.values[np.isnan(height_grid.aspect.values)] =\
            np.random.uniform(low=0, high=360, size=np.where(np.isnan(height_grid.aspect.values))[0].shape[0])
        height_grid.slope.values[np.isnan(height_grid.slope.values)] = 0
        
        svfs = np.ones(len(ys), dtype=np.float32)
        
        # aspect is clockwise from north
        # but arrays are north down, east right
        # nbr_direction = np.array([[225, 180   , 135],
                                  # [270, np.nan,  90],
                                  # [315, 0     ,  45]], dtype=np.float32)
        nbr_phi = [0   , 45    , 90  , 135   , 180 , 225   , 270 , 315   ]
        nbr_x = [1000, 1414.2, 1000, 1414.2, 1000, 1414.2, 1000, 1414.2]
        for i in range(len(ys)):            
            asp = height_grid.aspect.values[ys[i], xs[i]]
            slo = height_grid.slope.values[ys[i], xs[i]]
            ele = height_grid.elev.values[ys[i], xs[i]]
                        
            # clockwise from north
            nbr_h = []
            if ys[i]>0:
                nbr_h.append(height_grid.elev.values[ys[i]-1, xs[i]])
            else:
                nbr_h.append(0)
                
            if ys[i]>0 and xs[i]<(height_grid.elev.shape[-1]-1):
                nbr_h.append(height_grid.elev.values[ys[i]-1, xs[i]+1])
            else:
                nbr_h.append(0)
                
            if xs[i]<(height_grid.elev.shape[-1]-1):
                nbr_h.append(height_grid.elev.values[ys[i], xs[i]+1])
            else:
                nbr_h.append(0)

            if ys[i]<(height_grid.elev.shape[0]-1) and xs[i]<(height_grid.elev.shape[-1]-1):
                nbr_h.append(height_grid.elev.values[ys[i]+1, xs[i]+1])
            else:
                nbr_h.append(0)
                            
            if ys[i]<(height_grid.elev.shape[0]-1):
                nbr_h.append(height_grid.elev.values[ys[i]+1, xs[i]])
            else:
                nbr_h.append(0)

            if ys[i]<(height_grid.elev.shape[0]-1) and xs[i]>0:
                nbr_h.append(height_grid.elev.values[ys[i]+1, xs[i]-1])
            else:
                nbr_h.append(0)
                
            if xs[i]>0:
                nbr_h.append(height_grid.elev.values[ys[i], xs[i]-1])
            else:
                nbr_h.append(0)
            
            if ys[i]>0 and xs[i]>0:
                nbr_h.append(height_grid.elev.values[ys[i]-1, xs[i]-1])
            else:
                nbr_h.append(0)
            
            nbr_Hphi = [0.5*np.pi - np.arctan(max(0, (nbr_h[j] - ele))/nbr_x[j]) for j in range(len(nbr_x))]
            
            svfs[i] = np.sum([calc_svf(slo, asp, nbr_Hphi[j], nbr_phi[j]) for j in range(len(nbr_x))]) / 8
        
        np.save(f'/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/{fname}.npy', svfs)
                
        
        # skyviewmap = grid.landfrac.copy()
        # skyviewmap.values[:,:] = 0
        # skyviewmap.values[np.where(grid.landfrac.values == 0)] = svfs
        # skyviewmap.plot(); plt.show()


        
    if False:           
        ####################################
        # how to re-map onto the BNG grid        
        shading = np.load(outdir+f'shading_mask_day_{day}.npy')
        grid = xr.open_dataset(hj_ancil_fldr+'/chess_lat_lon.nc')
        elevation = xr.open_dataset(hj_ancil_fldr+'/uk_ihdtm_topography+topoindex_1km.nc')

        hr = 6
        shade_map1 = (grid.lat.values * 0).astype(np.uint8)
        shade_map1[grid.landfrac.values==1] = shading[hr,:]

        hr = 7
        shade_map = (grid.lat.values * 0).astype(np.uint8)
        shade_map[grid.landfrac.values==1] = shading[hr,:]

        hr = 8
        shade_map2 = (grid.lat.values * 0).astype(np.uint8)
        shade_map2[grid.landfrac.values==1] = shading[hr,:]

        # hr = 13
        # shade_map2 = (grid.lat.values * 0).astype(np.uint8)
        # shade_map2[grid.landfrac.values==1] = shading[hr,:]

        hr = 17
        shade_map3 = (grid.lat.values * 0).astype(np.uint8)
        shade_map3[grid.landfrac.values==1] = shading[hr,:]

        fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
        ax[0].imshow(shade_map[::-1,:])
        ax[1].imshow(elevation.elev.values[::-1,:])
        ax[2].imshow(shade_map3[::-1,:])
        plt.show()


        fig, ax = plt.subplots(1,4, sharex=True, sharey=True)
        ax[0].imshow(shade_map1[::-1,:])
        ax[1].imshow(shade_map[::-1,:])
        ax[2].imshow(shade_map2[::-1,:])
        ax[3].imshow(elevation[::-1,:])        
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(shade_map[::-1,:])
        ax.plot(st[1],st[0], 'xr')
        ax.plot(en[1],en[0], 'or')
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(shade_map[:,:])
        ax.plot(st[1],st[0], 'xr')
        ax.plot(en[1],en[0], 'or')
        plt.show()

    if False:
        
        ######################################
        ## tests on a small grid
        hj_base = '/gws/nopw/j04/hydro_jules/'
        hj_ancil_fldr = hj_base + '/data/uk/ancillaries/'
        # create artificial terrain map to test north/south indexing problem
        size = (50,50)

        grid = xr.open_dataset(hj_ancil_fldr+'/chess_lat_lon.nc')
        grid = grid.isel(y=range(size[0]), x=range(size[1]))
        grid.load()
        
        grid.landfrac.values[:,:] = 1
        ys, xs = np.where(grid.landfrac)
        grid['elev'] = grid.lat.copy()
        grid.elev.values[24:26,24:26] = 4000
        grid.elev.values[4, 25] = 2500 # southern-most point, latitude increased in pos i direction
        zeromask = np.zeros(size, dtype=np.uint8)
        ts = TerrainShading()
        
        day = 0
        min_res = 20
        shading = np.zeros((24, len(ys)), dtype=np.float32)
        daystamp = pd.to_datetime('2016-01-01') + datetime.timedelta(days=day)
        wrong_tstamps_array = np.empty((24, 60//min_res), dtype=object)
        proper_tstamps_array = np.empty((24, 60//min_res), dtype=object)
        for hr in range(24):
            print(f'Doing hour {hr}')
            for jj, mn in enumerate(range(0, 60, min_res)):
                # accumulate timestamp over the last hour
                timestamp = daystamp + datetime.timedelta() - datetime.timedelta(hours=hr, minutes=mn)
                proper_tstamp = daystamp + datetime.timedelta(hours=hr) - datetime.timedelta(minutes=mn)                
                wrong_tstamps_array[hr, jj] = timestamp
                proper_tstamps_array[hr, jj] = proper_tstamp
                sp = SolarPosition(timestamp, timezone=0) # utc
                for i in range(len(ys)):
                    sp_angles = sp.calc_solar_angles_return(
                        grid.lat.values[ys[i],xs[i]],
                        grid.lon.values[ys[i],xs[i]]
                    )
                    shad, st, en = ts.inner_calc(
                        ys[i], xs[i], sp_angles.solar_azimuth_angle, 
                        sp_angles.solar_elevation,
                        grid.lat.values, grid.lon.values,
                        grid.y.values, grid.x.values,
                        zeromask, grid.elev.values
                    )
                    shading[hr,i] += shad # accumulate minute shading
        shading /= float(60 // min_res)



        shademap = grid.landfrac.copy()
        shademap.values[:,:] = 0
        shademap.values[np.where(grid.landfrac.values > 0)] = 1 - shading[8,:] 
        shademap.plot()
        plt.show()






        hr = 12
        timestamp = pd.to_datetime('2016-09-01') + datetime.timedelta(hours=hr)
        sp = SolarPosition(timestamp, timezone=0) # utc
        shad_mask = zeromask.copy()
        st_en_map = np.zeros(size, dtype=object)        

        for oi in range(size[0]):
            for oj in range(size[1]):
                sp_angles = sp.calc_solar_angles_return(lats[oi,oj], lons[oi,oj])
                
                shading, st_pnt, en_pnt = ts.inner_calc(
                    oi, oj,
                    sp_angles.solar_azimuth_angle,
                    sp_angles.solar_elevation,
                    lats, lons, ys, xs,
                    zeromask, elev
                )
                shad_mask[oi, oj] = shading
                st_en_map[oi, oj] = (st_pnt, en_pnt)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(shad_mask[:,:])
        ax[1].imshow(elev[:,:])
        ax[0].plot(st_en_map[22,22][0][1], st_en_map[22,22][0][0], 'xr')
        ax[0].plot(st_en_map[22,22][1][1], st_en_map[22,22][1][0], 'or')
        plt.show()

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(shad_mask[::-1,:])
        ax[1].imshow(elev[::-1,:])
        plt.show()
