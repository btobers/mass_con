import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.path as path
import geopandas as gpd
import sys, os, argparse, configparser
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Calibri"
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8

'''
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
'''

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))


# make sure vx and vy rasters are same size and extent
def check_rasters(r1, r2):
    if r1.transform != r2.transform:
        exit(1)
    if r1.shape != r2.shape:
        exit(1)

    return


# sample raster at all x,y locations using two 2d x,y input arrays
def sample_2d_raster(x, y, r_ds, mean=False):
    # if mean - we will get mean of pixels in between flowline vertices
    if mean:
        vals = np.full((x.shape[0],x.shape[1]-1), np.nan)
        for _i in range(vals.shape[1]):
            for _j in range(vals.shape[0]):
                pts = getEquidistantPoints((x[_j,_i], y[_j,_i]), (x[_j,_i+1], y[_j,_i+1]), 10)
                vals[_j,_i] = np.nanmean(np.asarray([x[0] for x in r_ds.sample(pts)]))
    
    else:
        vals = np.full(x.shape, np.nan)
        # sample raster
        for _i in range(vals.shape[1]):
            vals[:,_i] = np.asarray([x[0] for x in r_ds.sample(np.column_stack((x[:,_i], y[:,_i])))])

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
    thick_coords = thick_gdf[['x','y']].to_numpy()
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
                h_in[_i] = np.nanmean(thick_gdf['h'].iloc[idxs])
                # save average coord
                coords_in[_i, 0] = np.nanmean(thick_gdf['x'].iloc[idxs])
                coords_in[_i, 1] = np.nanmean(thick_gdf['y'].iloc[idxs])

    return h_in, coords_in


# for each flowline pair, get the centroid coordinate pair closest to the average thickness measurment location for that flowband - this will be the location where we define the known thickness
def get_starts(cx, cy, coords_in):
    start_pos = np.zeros(cx.shape[1])
    for _i in range(cx.shape[1]):
        dist = ((cx[:,_i] - coords_in[_i,0])**2 + (cy[:,_i] - coords_in[_i,1])**2)**0.5
        start_pos[_i] = np.argmin(dist)

    return start_pos.astype(int)


def conserve_mass(dx, dy, vx, vy, smb, dhdt, h_in, start_pos):
    '''
    we determine the ice thickness along flowlines by conservation of mass

    the ice flux through a segment (dx, dy) is (v_surface * h), where v_surface is the surface velocity normal to the flux gate segment and h is the ice thickness.
    the unit normal to (dx, dy) is sqrt(dx^2 + dy^2).
    the normal surface velocity through a segment (dx, dy) is ((vy*dx - vx*dy) / sqrt(dx^2 + dy^2))
    therefore the ice flux through a segment (dx, dy) simplifies to (h*(vy*dx-vx*dy))

    here, we'll refer to (vy*dx-vx*dy) as our area flux
    '''
    # step along flowlines and get vx vy for each centroid
    area_flux = np.full_like(dx, np.nan)
    h = np.full_like(dx, np.nan)
    flux_in = []

    # convert smb from mm w.e. to m ice equivalent 
    smb = smb / 1000            # mm to m
    smb = smb * 1000 / 917      # m water to m ice

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
            # account for zero incoming thickness
            if h[_j-1,_i] <= 0:
                h[_j, _i] = np.nan
                continue
            lastf = h[_j-1,_i] * area_flux[_j-1, _i] 
            thish = (lastf / area_flux[_j, _i]) + smb[_j,_i] - dhdt
            if thish < 0:
                thish = np.nan
            h[_j, _i] = thish  

    return h


def main():
    # Set up CLI
    parser = argparse.ArgumentParser(
    description='''Program conserving mass and calculating ice thickness along glacier flowlines\nNot all arguments are set up for command line input. Edif input files in the configuration file''',
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('conf', help='path to configuration file (.ini)', type=str)
    parser.add_argument('-mb', dest = 'mb', help='mass balance gradient (mm w.e./m)', type=float, nargs='?')
    parser.add_argument('-ela', dest = 'ela', help='equilibrium line altitude (m)', type=float, nargs='?')
    parser.add_argument('-dhdt', dest = 'dhdt', help='surface elevation change rate (m/yr)', type=float, nargs='?')
    parser.add_argument('-out_name', dest = 'out_name', help='output point cloud file name', type=str, nargs='?', default='pcloud.csv')
    parser.add_argument('-plot', help='Flag: Plot results', default=False, action='store_true')
    args = parser.parse_args()

    # parse config file
    config = configparser.ConfigParser()
    try:
        config.read(args.conf)
    except Exception as err:
        print("Unable to parse config file.")
        print(err)
        sys.exit(1)

    dat_path = config['path']['dat_path']
    verts_x = config['path']['verts_x']
    verts_y = config['path']['verts_y']
    vx_ds = config['path']['vx']
    vy_ds = config['path']['vy']
    dem_ds = config['path']['dem']
    rdata = config['path']['rdata']
    out_name = config['path']['out_name']
    mb = float(config['param']['mb'])
    ela = float(config['param']['ela'])
    dhdt = float(config['param']['dhdt'])
    plot = config['param'].getboolean('plot')

    if args.mb:
        mb = args.mb
    if args.ela:
        ela = args.ela
    if args.dhdt:
        dhdt = args.dhdt
    if args.out_name:
        out_name = args.out_name
        # make sure endswith csv
        if not out_name.endswith('.csv'):
            out_name = out_name.split('.')[0] + '.csv'
    if args.plot:
        plot = args.plot

    # x and y vertex coordinates
    verts_x = pd.read_csv(dat_path + verts_x,header=None).to_numpy()
    verts_y = pd.read_csv(dat_path + verts_y,header=None).to_numpy()

    # remove first flowline - seems to be some issues on output of this one, perhaps too close to gorge edge
    # verts_x = verts_x[:,:-1]
    # verts_y = verts_y[:,:-1]

    # get centroids
    cx, cy, dx, dy = get_centroids(verts_x, verts_y)

    # x and y component surface velocities
    vx_ds = rio.open(dat_path + vx_ds, 'r')
    # vx_ds = rio.open(dat_path + 'Millan_vx_clip.tif', 'r')    
    vy_ds = rio.open(dat_path + vy_ds, 'r')
    # vy_ds = rio.open(dat_path + 'Millan_vy_clip.tif', 'r')
    # check for raster size mismatch
    check_rasters(vx_ds, vy_ds)

    dem_ds = rio.open(dat_path + dem_ds, 'r')

    # read in thickness measurements - this is currently set up to read a geopackage, but could easily by swapped for a csv file using:
    # rdata = pd.read_csv('file.csv'), so long as the csv has x, y, h columns
    rdata = gpd.read_file(dat_path + rdata)
    rdata = rdata.rename(columns ={'srf_bottom_thick':'h'})
    rdata = rdata[~rdata.h.isna()]
    # projet radar data to same coordinate sys as velocity data
    rdata = rdata.to_crs(epsg=3413)
    # rdata = rdata.to_crs(epsg=32607)
    rdata['x'] = rdata.centroid.x
    rdata['y'] = rdata.centroid.y
    rdata = rdata.sort_values(['y'], ascending=True)

    # for each set of centroids, get average input thickness
    h_in, coords_in = get_thickness_input(verts_x, verts_y, rdata)

    # trim all centroid points upglacer from thickness measurments
    start_pos = get_starts(cx, cy, coords_in)

    # sample vx, vy, and elev - we'll take average raster value in between flowline vertices
    vx = sample_2d_raster(verts_x, verts_y, vx_ds, mean=True)
    vy = sample_2d_raster(verts_x, verts_y, vy_ds, mean=True)
    elev = sample_2d_raster(verts_x, verts_y, dem_ds, mean=True)

    # get surface mass balance
    smb = get_smb(elev, mb, ela)

    # conserve max and get along-flowline thicknesses
    h = conserve_mass(dx, dy, vx, vy, smb, dhdt, h_in, start_pos)
    # trim unreasonable thicknesses - we'll set anything greater than 950 m thick to nan, as our deepest amp thickness meaurements are ~920 m
    # h[h > 940] = np.nan

    # outpath
    path = dat_path + '../out/' + out_name
    path = os.path.normpath(path)

    if plot:
        fig = plt.figure(figsize=(6.5,9))
        pad = '1%'
        size = '2%'
        s=5
        gs = fig.add_gridspec(nrows=5, ncols=1, left=0.125, right=0.85, wspace=0, hspace=0.05)

        ax1 = fig.add_subplot(gs[1,0])
        ax1.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax1.scatter(cx, cy, c=elev, cmap='gist_earth', s=s)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size=size, pad=pad)
        fig.colorbar(c, cax=cax, orientation='vertical', label='Elevation (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.xaxis.set_ticks_position('both')
        ax1.set_xticklabels([])
        ax1.set_aspect('equal')

        ax0 = fig.add_subplot(gs[0,0])
        left, right = ax1.get_xlim()
        bottom, top = ax1.get_ylim()
        xx = np.arange(left, right, step=500)
        yy = np.arange(bottom, top, step=500)
        XX,YY = np.meshgrid(xx,yy)
    
        grid_coords = [(x,y) for x, y in np.column_stack((np.ravel(XX),np.ravel(YY)))]
        vx = np.asarray([x[0] for x in vx_ds.sample(grid_coords)])
        vy = np.asarray([x[0] for x in vy_ds.sample(grid_coords)])
        q = ax0.quiver(np.ravel(XX), np.ravel(YY), vx, vy, color='r',width=.003)
        ax0.add_patch(patches.FancyBboxPatch((left+100,bottom+100), 4250, 1500, fc="w", ec='gray', alpha=.8, boxstyle='round'))
        qk = ax0.quiverkey(q,  X=.05, Y=.08, U=100, label=r'$100\ \frac{m}{yr}$', labelpos='E', labelsep=.05, fontproperties={'size':8}, coordinates = 'axes')
        ax0.set_xlim(left,right)
        ax0.set_ylim(bottom,top)
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size=size, pad=pad)
        cax.axis('off')
        ax0.set_ylabel('Northing (m)')
        ax0.set_xlabel('Easting (m)')   
        ax0.xaxis.tick_top() 
        ax0.xaxis.set_label_position('top') 
        ax0.xaxis.set_ticks_position('both')

        ax2 = fig.add_subplot(gs[2,0])
        v = max(np.abs(np.nanmin(smb)), np.nanmax(smb))
        ax2.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax2.scatter(cx, cy, c=1e-3*smb, vmin=-1e-3*v, vmax=1e-3*v, cmap='RdBu', s=s)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size=size, pad=pad)
        fig.colorbar(c, cax=cax, orientation='vertical', label='Annual mass\nbalance (m w.e.)')
        ax2.set_ylabel('Northing (m)')
        ax2.set_xticklabels([])
        ax2.xaxis.set_ticks_position('both')
        ax2.set_aspect('equal')

        ax3 = fig.add_subplot(gs[3,0])
        ax3.plot(verts_x[:,:],verts_y[:,:],'tab:grey',lw=.5)
        c = ax3.scatter(cx, cy, c=h, cmap='viridis_r', vmin=200, vmax=1000, s=s)
        c = ax3.scatter(rdata.x, rdata.y, c=rdata.h, cmap='viridis_r', s=s, vmin=0, vmax=1000, zorder=100)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size=size, pad=pad)
        fig.colorbar(c, cax=cax, orientation='vertical', label='Ice thickness (m)')
        # ax3.set_xlabel('Easting (m)')
        ax3.set_xticklabels([])
        ax3.set_ylabel('Northing (m)')
        ax3.set_aspect('equal')
        ax3.xaxis.set_ticks_position('both')

        line = 2
        ax4 = fig.add_subplot(gs[4,0])
        dist = np.zeros(cx.shape[0])
        dist[1:] = np.cumsum(np.sqrt(np.diff(cx[:,line]) ** 2.0 + np.diff(cy[:,line]) ** 2.0))*1e-3
        l1, = ax4.plot(dist, elev[:,line], c='k')
        l2, = ax4.plot(dist, elev[:,line] - h[:,line], c='tab:gray')
        bed = elev[:,line] - h[:,line]
        ax4.fill_between(dist, bed, elev[:,line], color='tab:blue', alpha=0.2)
        d0 = (np.sqrt((cx[0,line] - coords_in[line,0]) ** 2.0 + (cy[0,line] - coords_in[line,1]) ** 2.0))*1e-3
        i0 = (np.abs(dist - d0)).argmin()
        l3 = ax4.vlines(x=d0, color='tab:gray', ls='--', zorder=-1000, ymin=bed[i0], ymax=elev[i0,line])
        ax4.set_ylabel('Elevation (m)')
        ax4.set_yticks([-500, 0, 500, 1000, 1500])
        ax4.set_xlabel('Distance down gorge (km)')
        ax4.invert_xaxis()
        ax4.legend(handles=[l1,l3,l2], labels=['IFSAR surface','Known thickness', 'Modeled bed'])
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size=size, pad=pad)
        cax.axis('off')

        # add hillshade to top four plots
        ls = colors.LightSource(azdeg=315, altdeg=15)
        z_tmp = dem_ds.read(1, window=rio.windows.from_bounds(left, bottom, right, top, dem_ds.transform))
        Z_hillshade = ls.hillshade(z_tmp,vert_exag=1000, dx=right - left,dy=top - bottom)
        ax0.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        ax1.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        ax2.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        ax3.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        fig.suptitle(f'Mass balance gradient = {mb} mm w.e./m\nELA = {ela} m\ndh/dt = {dhdt} m/yr', fontsize=10)
        fig.tight_layout()
        plt.show()
        fig.savefig(path[:-4] + '.png', dpi=300)

    # export output xyz points
    cx = np.ravel(cx, order='F')
    cy = np.ravel(cy, order='F')
    h = np.ravel(h, order='F')

    out = np.column_stack((cx,cy,h))
    out_df = pd.DataFrame(data=out, columns=['x','y','h'])
    out_df.to_csv(path)
    print('point cloud exported to:\t' + str(path))

# execute if run as a script
if __name__ == '__main__':
    main()
