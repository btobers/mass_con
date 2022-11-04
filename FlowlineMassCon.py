import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.path as path
import geopandas as gpd
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
"""
FlowlineMassConModel.py

author: Brandon S. Tober
date: 01SEPT2022
updated: 03NOV2022

Mass conservation approach to deriving glacier thickness alonng flowlines

inputs:
verts_x: csv file containing x-coordinates along each flowline
verts_y: csv file containing y-coordinates along each flowline
vx_ds: x-component surface velocity geotiff file (same projection as cx & cy)
vy_ds: y-component surface velocity geotiff file (same projection as cx & cy)
dem_ds: digital elevation model geotiff file (same projection as cx & cy)
h_in: geopackage/csv with thickness measurements
mb: mass balance gradient (mm w.e./m)
ela: equilibrium line altitude (m)
dhdt: surface elevation change rate (m/yr)

outputs:
pcloud: file of size (#Steps x #Flowlines - 1, 3) containing x-coordinates, y-coordinates, thicknesses
"""

# make sure vx and vy rasters are same size and extent
def check_rasters(r1, r2):
    if r1.transform != r2.transform:
        exit(1)
    if r1.shape != r2.shape:
        exit(1)

    return


# sample raster at all x,y locations using two 2d x,y input arrays
def sample_2d_raster(xy, r_ds):
    # instantiate output array
    vals = np.full((xy.shape[:2]), np.nan)

    # sample raster
    for _i in range(vals.shape[1]):
        vals[:,_i] = np.asarray([x[0] for x in r_ds.sample(xy[:,_i,:])])
    
    return vals


# go through x and y flowline vertices and get centroid locations
def get_centroids(verts_x, verts_y):
    # first make sure verts_x and verts_y are same size
    r, c = verts_x.shape
    if (r,c) != verts_y.shape:
        exit(1)
    # instantiate cx,cy,dx,dy
    cx = np.zeros((r,c-1))
    cy = np.zeros((r,c-1))
    dx = np.zeros((r,c-1))
    dy = np.zeros((r,c-1))

    # go along flowline pairs and get centroid location as well as deltax and deltay
    for _i in range(c - 1):
        for _j in range(r):
            cx[_j, _i] = (verts_x[_j,_i] + verts_x[_j,_i+1])/2
            cy[_j, _i] = (verts_y[_j,_i] + verts_y[_j,_i+1])/2
            dx[_j, _i] = verts_x[_j,_i+1] - verts_x[_j,_i]
            dy[_j, _i] = verts_y[_j,_i+1] - verts_y[_j,_i]

    return cx, cy, dx, dy


# function to get the surface mass balance at a given elevation given a mass balance gradient (mm/m/yr) and an equilibrium line altitue (ELA; (m))
def get_smb(elev, mb, ela):
    # mass balance at a given grid cell will be the difference in elevation from the ela multiplies by the gradient
    # output smb should be negative below ela, positive above
    smb = (elev - ela) * mb

    return smb      # mm w.e.


# get thickness input between each set of flowlines
# from each set of neighboring flowlines, create a polygon and see which thickness measurements fall within, then get average
def get_thickness_input(verts_x, verts_y, thick_gdf):
    # get thickness gdf x,y coords stacked together
    thick_coords = thick_gdf[["x","y"]].to_numpy()
    r, c = verts_x.shape
    # instantiate thickness input array that will hold average of all thickness measurements between each set of flowlines
    h_in = np.repeat(np.nan, c - 1)
    # instantiate average input coordinate for each flowline pair
    coords_in = np.full((c -1, 2), np.nan)

    # do the magic
    for _i in range(c - 1):
            # take first set of along flowline verts, then concatenate flipped second set to make a closed polygon from
            tmpx = np.concatenate((verts_x[:,_i], np.flipud(verts_x[:,_i+1])))
            tmpy = np.concatenate((verts_y[:,_i], np.flipud(verts_y[:,_i+1])))
            # horizontally stack x and y coords
            coords = np.column_stack((tmpx, tmpy))
            poly = path.Path(coords)
            # get thickness points that fill within this poly
            idxs = poly.contains_points(thick_coords)
            if np.sum(idxs) > 0:
                # get averaged thickness
                h_in[_i] = np.nanmean(thick_gdf["h"].iloc[idxs])
                # save average coord
                coords_in[_i, 0] = np.nanmean(thick_gdf["x"].iloc[idxs])
                coords_in[_i, 1] = np.nanmean(thick_gdf["y"].iloc[idxs])

    return h_in, coords_in


# for each flowline pair, find the index of the first upglacier centroid for which we'll define the input flux
def get_starts(cx, cy, dx, dy, coords_in):
    ### ATTENTION ###
    # we'll use prior knowledge that our glacier flows in a negative x-direction from our thickness measuremnts. 
    # so we'll of the index correspoinding to the first downglacier centroid from the averaged location of our upglacier thickness meaurements.
    start_pos = np.zeros(cx.shape[1])
    for _i in range(cx.shape[1]):
        ### change the following line based on glacier application ###
        ### for a glacier flowing in the positive y-direction, the following line should be: ###
        # idx_arr = np.where(cy[:, _i] >= coords_in[_i, 1])[0] ###
        idx_arr = np.where(cx[:, _i] <= coords_in[_i, 0])[0]
        if len(idx_arr) == 0:
            continue
        else:
            start_pos[_i] = idx_arr[0]

    return start_pos.astype(int)


def conserve_mass(dx, dy, vx, vy, smb, dhdt, h_in, start_pos):
    """
    we determine the ice thickness along flowlines by conservation of mass

    the ice flux through a segment (dx, dy) is (v_surface * h), where v_surface is the surface velocity normal to the flux gate segment and h is the ice thickness.
    the unit normal to (dx, dy) is sqrt(dx^2 + dy^2).
    the normal surface velocity through a segment (dx, dy) is ((vy*dx - vx*dy) / sqrt(dx^2 + dy^2))
    therefore the ice flux through a segment (dx, dy) simplifies to (h*(vy*dx-vx*dy))

    here, we'll refer to (vy*dx-vx*dy) as our area flux
    """
    # step along flowlines and get vx vy for each centroid
    area_flux = np.full_like(dx, np.nan)
    h = np.full_like(dx, np.nan)
    flux_in = []

    # convert smb from mm w.e. to m ice
    smb = smb / 1000
    # convert to meters ice equiv.
    smb = smb * 1000 / 917

    # iterate through each flowline, get vx and vy arrays
    for _i in range(dx.shape[1]):
        # get area flux through each centroid, determined by (vy*dx-vx*dy)
        area_flux[:, _i] = np.abs((vx[:, _i] * dy[:, _i]) - (vy[:, _i] * dx[:, _i]))

        # determine input ice flux, as (h*(vy*dx-vx*dy)) - this is the quantity that will be conserved along each flowline
        flux_in.append(h_in[_i] * area_flux[start_pos[_i], _i])
        h[start_pos[_i], _i] = h_in[_i]

        # iterate over all centroids and get thickness - we'll use two for loops, one for going upstream and one for going downstream. there's probably a more efficient way to do this
        # go upstream first
        for _j in range(start_pos[_i] - 1, -1, -1):
            lastf = h[_j+1,_i] * area_flux[_j+1, _i]
            h[_j, _i] = (lastf / area_flux[_j, _i]) + smb[_j,_i] - dhdt

        # now downstream
        for _j in range(start_pos[_i] + 1, dx.shape[0]):
            lastf = h[_j-1,_i] * area_flux[_j-1, _i]
            h[_j, _i] = (lastf / area_flux[_j, _i]) + smb[_j,_i] - dhdt      

    print("total input ice flux = {:.3f} cubic km. per year".format(np.nansum(flux_in)*1e-9))

    return h


def main():
    ##############################################################################################
    ##### user inputs - note all input files should be projected to same coordinate system   #####
    ##############################################################################################
    # mass balance gradient (mm w.e./m)
    mb = 4
    # equilibrium line altitude (m)
    ela = 1550
    # surface elevation change rate (m/yr)
    dhdt = -.5
    # data file path
    dat_path = '../massCon/ruth/data/'
    # x and y vertex coordinate arrays output by GenFlowlines.m
    verts_x = 'verts_x.csv'
    verts_y = 'verts_y.csv'
    # x and y component velocity rasters
    vx_ds = 'ALA_G0120_0000_vx_clip.tif'
    vy_ds = 'ALA_G0120_0000_vy_clip.tif'
    # digital elevation model
    dem_ds = 'ifsar_ruth.tif'
    # thickness measurements
    rdata = 'amp_picks.gpkg'
    # output file name
    oname = 'tmp.csv'
    # plot results
    plot = True
    ##############################################################################################

    # x and y vertex coordinates
    verts_x = pd.read_csv(dat_path + verts_x,header=None).to_numpy()
    verts_y = pd.read_csv(dat_path + verts_y,header=None).to_numpy()

    # remove first flowline - seems to be some issues on output of this one, perhaps too close to gorge edge
    # verts_x = verts_x[:,:-1]
    # verts_y = verts_y[:,:-1]

    # get centroids
    cx, cy, dx, dy = get_centroids(verts_x,verts_y)
    # stack x,y centroid pairs
    cxcy = np.dstack((cx, cy))

    # x and y component surface velocities
    vx_ds = rio.open(dat_path + vx_ds, "r")
    # vx_ds = rio.open(dat_path + "Millan_vx_clip.tif", "r")    
    vy_ds = rio.open(dat_path + vy_ds, "r")
    # vy_ds = rio.open(dat_path + "Millan_vy_clip.tif", "r")
    # check for raster size mismatch
    check_rasters(vx_ds, vy_ds)

    dem_ds = rio.open(dat_path + dem_ds, "r")

    # read in thickness measurements - this is currently set up to read a geopackage, but could easily by swapped for a csv file using:
    # rdata = pd.read_csv('file.csv'), so long as the csv has x, y, h columns
    rdata = gpd.read_file(dat_path + rdata)
    rdata = rdata.rename(columns ={'srf_bottom_thick':'h'})
    rdata = rdata[~rdata.h.isna()]
    # projet radar data to same coordinate sys as velocity data
    rdata = rdata.to_crs(epsg=3413)
    # rdata = rdata.to_crs(epsg=32607)
    rdata["x"] = rdata.centroid.x
    rdata["y"] = rdata.centroid.y
    rdata = rdata.sort_values(["y"], ascending=True)

    # for each set of centroids, get average input thickness
    h_in, coords_in = get_thickness_input(verts_x, verts_y, rdata)

    # trim all centroid points upglacer from thickness measurments
    start_pos = get_starts(cx, cy, dx, dy, coords_in)

    # sample vx, vy, and elev
    vx = sample_2d_raster(cxcy, vx_ds)
    vy = sample_2d_raster(cxcy, vy_ds)
    elev = sample_2d_raster(cxcy, dem_ds)

    # get surface mass balance
    smb = get_smb(elev, mb, ela)

    # conserve max and get along-flowline thicknesses
    h = conserve_mass(dx, dy, vx, vy, smb, dhdt, h_in, start_pos)

    if plot:
        fig, axs = plt.subplots(1,3,sharey=True, figsize=(15,5))

        ax = axs[0]
        ax.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax.scatter(cx,cy,c=elev,cmap='gist_earth')
        fig.colorbar(c, ax=ax,label='elevation (m)')
        ax.set_xlabel('x-distance (m)')
        ax.set_ylabel('y-distance (m)')

        ax = axs[1]
        v = max(np.abs(np.nanmin(smb)), np.nanmax(smb))
        ax.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax.scatter(cx,cy,c=1e-3*smb,vmin=-1e-3*v,vmax=1e-3*v,cmap='RdBu')
        fig.colorbar(c,ax=ax,label='annual mass balance (m w.e.)')
        ax.set_xlabel('x-distance (m)')

        ax = axs[2]
        ax.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax.scatter(cx,cy,c=h,cmap='viridis_r',vmin=200,vmax=1000)
        c = ax.scatter(rdata.x, rdata.y, c=rdata.h, cmap='viridis_r', vmin=0,vmax=1000,zorder=100)
        fig.colorbar(c, ax=ax,label='ice thickness (m)')
        ax.set_xlabel('x-distance (m)')

        fig.suptitle(f"mass balance gradient = {mb} mm w.e./m/yr\nela = {ela} m\ndh/dt = {dhdt} m/yr")
        plt.show()
        fig.savefig(dat_path + f'out/mb_{mb}_ela_{ela}_dhdt_{dhdt}.png', dpi=300)

    # trim unreasonable thicknesses - we'll set anything greater than 950 m thick to nan, as our deepest amp thickness meaurements are ~920 m
    # h_centroids[h_centroids > 940] = np.nan

    # export output xyz points
    cx = np.ravel(cx, order="F")
    cy = np.ravel(cy, order="F")
    h = np.ravel(h, order="F")

    out = np.column_stack((cx,cy,h))
    out_df = pd.DataFrame(data=out, columns=["x","y","h"])
    out_df.to_csv(dat_path + 'out/' + oname)
    print(f"point cloud exported to:\t{dat_path + 'out/' + oname}")

# execute if run as a script
if __name__ == "__main__":
    main()