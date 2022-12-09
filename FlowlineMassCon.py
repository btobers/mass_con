import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from shapely import geometry
import sys, os, argparse, configparser
import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Calibri"
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8

'''
FlowlineMassConModel.py

author: Brandon S. Tober
date: 01SEPT2022
updated: 08DEC2022

Mass conservation approach to deriving glacier thickness along flowlines

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
    # if mean - we will get mean of pixels in between transverse vertices
    if mean:
        vals = np.full((x.shape[0],x.shape[1]-1), np.nan)
        for _i in range(vals.shape[1]):
            for _j in range(vals.shape[0]):
                pts = getEquidistantPoints((x[_j,_i], y[_j,_i]), (x[_j,_i+1], y[_j,_i+1]), 10)
                smpls = np.asarray([x[0] for x in r_ds.sample(pts)])
                smpls[smpls == r_ds.nodata] = np.nan
                vals[_j,_i] = np.nanmean(smpls)
    
    else:
        vals = np.full(x.shape, np.nan)
        # sample raster
        for _i in range(vals.shape[1]):
            vals[:,_i] = np.asarray([x[0] for x in r_ds.sample(np.column_stack((x[:,_i], y[:,_i])))])

    return vals


# go through x and y flowline vertices and get centroid locations, as well as cell area
def get_centroids(verts_x, verts_y):
    # first make sure verts_x and verts_y are same size
    r, c = verts_x.shape
    if (r,c) != verts_y.shape:
        exit(1)
    # instantiate cx,cy,dx,dy,area
    cx = np.zeros((r-1,c-1))
    cy = np.zeros((r-1,c-1))
    area = np.zeros((r-1,c-1))
    dx = np.zeros((r,c-1))
    dy = np.zeros((r,c-1))

    # go along flowline pairs and get centroid location as well as deltax and deltay between consecutive vertices (used to get cell boundary length)
    for _i in range(c - 1):
        for _j in range(r):
            if _j < r - 1:
                coords = (  (verts_x[_j,_i],verts_y[_j,_i]),
                            (verts_x[_j,_i+1],verts_y[_j,_i+1]),
                            (verts_x[_j+1,_i+1],verts_y[_j+1,_i+1]),
                            (verts_x[_j+1,_i],verts_y[_j+1,_i]))
                poly = geometry.Polygon(coords)
                area[_j,_i] = poly.area
                centroid = poly.centroid
                cx[_j,_i], cy[_j,_i] = centroid.x, centroid.y
            dx[_j, _i] = verts_x[_j,_i+1] - verts_x[_j,_i]
            dy[_j, _i] = verts_y[_j,_i+1] - verts_y[_j,_i]

    return cx, cy, dx, dy, area


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


def conserve_mass(dx, dy, area, vx, vy, smb, dhdt, h_in, start_pos, gamma):
    '''
    we determine the ice thickness along flowlines by conservation of mass
    following McNabb et al., 2012, we can express the upstream ice thickness as \frac{q_{out} + \int_S (\dot{b}_{sfc} + \frac{\partial h}{\partial t})dS }{\gamma W_{R} v_{sfc}},
    where q_{out} is the downstream ice flux at boundary R,  \dot{b}_{sfc} is the surface mass balance, \frac{\partial h}{\partial t} is the surface elevation change rate, 
    \gamma is the factor relating observed surface velocity to the depth-averaged velocity, W_{R} is the length of downstream boundary R, and v_{sfc} is the normal surface velocity.
    the downstream ice thickness is then: \frac{q_{in} - \int_S (\dot{b}_{sfc} + \frac{\partial h}{\partial t})dS }{\gamma W_{P} v_{sfc}},
    where q_{in} is the upstream ice flux at boundary P, and W_{P} is the length of upstream boundary P.


    our observable here is the surface velocity at each cell and the length of each upstream and downstream boundary.
    for a segment of ice (dx, dy), the unit normal to (dx, dy) is sqrt(dx^2 + dy^2)
    the normal surface velocity v_{sfc} through segment (dx, dy) then is: ((vy*dx - vx*dy) / sqrt(dx^2 + dy^2)), 
    which simplifies to (vy*dx - vx*dy)
    dx and dy here represent the x and y distance between consecutive flowband vertices, where sqrt(dx^2 + dy^2) is the length of upstream or downstream boundaries P and R
    in our ice thickness equations from McNabb et al., 2012, (vy*dx - vx*dy)*\gamma represents our denominator (\gamma * W * v_{sfc}), which we'll refer to as q_ovr_h (our ice flux divided by thickness)
    '''
    # step along flowlines and get vx vy for each centroid
    q_ovr_h = np.full_like(dx, np.nan)
    h = np.full((dx.shape[0] - 1, dx.shape[1]), np.nan)         # thickness output array will be one less row than input dx, or dy shape (since we derive thickness at center of each cell)
    flux_in = []                                                # instantiate list to hold input ice flux for each flowband - based on measured thickness at a certain location

    # convert smb from mm w.e. to m ice equivalent 
    smb = smb / 1000            # mm to m
    smb = smb * 1000 / 917      # m water to m ice

    # iterate through each flowline, get vx and vy arrays
    for _i in range(dx.shape[1]):
        # get q_ovr_h (q/h) through each centroid, determined by (vy*dx-vx*dy)*gamma
        q_ovr_h[:, _i] = np.abs((vx[:, _i] * dy[:, _i]) - (vy[:, _i] * dx[:, _i]))*gamma

        # determine input ice flux and store measured average thickness at start_pos cell
        flux_in.append(h_in[_i] * q_ovr_h[start_pos[_i] + 1, _i])
        h[start_pos[_i], _i] = h_in[_i]

        # iterate over all cells and get thickness - we'll use two for loops, one for going upstream and one for going downstream.
        # upstream
        for _j in range(start_pos[_i] - 1, -1, -1):
            q_jp2 = h[_j + 1, _i] * q_ovr_h[_j + 2, _i]
            h_j = (q_jp2  - (smb[_j + 1, _i] - dhdt)*(area[_j + 1, _i])) / q_ovr_h[_j + 1, _i]
            # constrain positive thickness
            if h_j < 0:
                h_j = np.nan 
            h[_j, _i] = h_j  

        # downstream
        for _j in range(start_pos[_i] + 1, h.shape[0]):
            # account for zero incoming thickness
            if h[_j - 1, _i] <= 0:
                h[_j, _i] = np.nan
                continue
            q_j = h[_j - 1 , _i] * q_ovr_h[_j, _i]
            h_j = (q_j - (smb[_j, _i] - dhdt)*(area[_j, _i])) / q_ovr_h[_j + 1, _i]
            # constrain positive thickness
            if h_j < 0:
                h_j = np.nan
            h[_j, _i] = h_j  

    # print(f'Total input ice flux: {np.sum(flux_in)*1e-9} km^3/year')

    return h


def main():
    # Set up CLI
    parser = argparse.ArgumentParser(
    description='''Program conserving mass and calculating ice thickness along glacier flowlines\nNot all arguments are set up for command line input. Edif input files in the configuration file\n\n
                    Example call: $python FlowlineMassCon.py config.ini -mb 4 -ela 1000 -dhdt -0.5 -gamma 0.8 -plot''',
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('conf', help='path to configuration file (.ini)', type=str)
    parser.add_argument('-mb', dest = 'mb', help='mass balance gradient (mm w.e./m/y)', type=float, nargs='?')
    parser.add_argument('-ela', dest = 'ela', help='equilibrium line altitude (m)', type=float, nargs='?')
    parser.add_argument('-dhdt', dest = 'dhdt', help='surface elevation change rate (m/yr)', type=float, nargs='?')
    parser.add_argument('-gamma', dest = 'gamma', help='factor relating surface velocity to depth-averaged velocity', type=float, nargs='?')
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
    gamma = float(config['param']['gamma'])
    mb = float(config['param']['mb'])
    ela = float(config['param']['ela'])
    dhdt = float(config['param']['dhdt'])
    plot = config['param'].getboolean('plot')

    if args.gamma is not None:
        gamma = args.gamma
    if args.mb is not None:
        mb = args.mb
    if args.ela is not None:
        ela = args.ela
    if args.dhdt is not None:
        dhdt = args.dhdt
    if args.out_name is not None:
        out_name = args.out_name
        # make sure endswith csv
        if not out_name.endswith('.csv'):
            out_name = out_name.split('.')[0] + '.csv'
    if args.plot:
        plot = args.plot    

    print(f'Mass Balance Gradient:\t\t{mb} mm w.e./m/y\nEquilibrium Line Altutde:\t{ela} m\nSurface Elevation Change Rate:\t{dhdt} m/y\nGamma:\t\t\t\t{gamma}')

    # x and y vertex coordinates
    verts_x = pd.read_csv(dat_path + verts_x,header=None).to_numpy()
    verts_y = pd.read_csv(dat_path + verts_y,header=None).to_numpy()

    # remove first flowline - seems to be some issues on output of this one, perhaps too close to gorge edge
    # verts_x = verts_x[:,:-1]
    # verts_y = verts_y[:,:-1]

    # get centroids
    cx, cy, dx, dy, area = get_centroids(verts_x, verts_y)

    # x and y component surface velocities
    vx_ds = rio.open(dat_path + vx_ds, 'r')
    vy_ds = rio.open(dat_path + vy_ds, 'r')

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
    vx = sample_2d_raster(verts_x, verts_y, vx_ds, mean=True)           # take the mean x-component velocity between each set of transverse vertices
    vy = sample_2d_raster(verts_x, verts_y, vy_ds, mean=True)           # take the mean y-component velocity between each set of transverse vertices
    elev = sample_2d_raster(cx, cy, dem_ds)                             # take dem elevation value at cell's center

    # verts_x[-90:,-1] = np.nan
    # verts_y[-90:,-1] = np.nan
    # cx[-90:,-1] = np.nan
    # cy[-90:,-1]=np.nan

    # get surface mass balance
    smb = get_smb(elev, mb, ela)

    # conserve max and get along-flowline thicknesses
    h = conserve_mass(dx, dy, area, vx, vy, smb, dhdt, h_in, start_pos, gamma)

    # trim unreasonable thicknesses - we'll set anything greater than 950 m thick to nan, as our deepest amp thickness meaurements are ~920 m
    # h[h > 940] = np.nan


    if plot:
        pass
        # fig = plt.figure(figsize=(5.2,8.25))
        # pad = '1%'
        # size = '2%'
        # s=5
        # gs = fig.add_gridspec(nrows=5, ncols=1, left=0.125, right=0.850, wspace=0, hspace=0.1)

        # ax1 = fig.add_subplot(gs[1,0])
        # ax1.plot(verts_x[:,:],verts_y[:,:],'k',lw=.15)
        # c = ax1.scatter(cx, cy, c=elev, cmap='gist_earth', s=s)
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes("right", size=size, pad=pad)
        # fig.colorbar(c, cax=cax, orientation='vertical', label='Elevation (m)')
        # ax1.set_ylabel('Northing (km)')
        # ax1.xaxis.set_ticks_position('both')
        # ax1.set_xticklabels([])
        # ax1.set_aspect('equal')
        # ax1.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: f'{int(x * 1e-3)}'))

        # ax0 = fig.add_subplot(gs[0,0])
        # left, right = ax1.get_xlim()
        # bottom, top = ax1.get_ylim()
        # xx = np.arange(left, right, step=500)
        # yy = np.arange(bottom, top, step=500)
        # XX,YY = np.meshgrid(xx,yy)

        # grid_coords = [(x,y) for x, y in np.column_stack((np.ravel(XX),np.ravel(YY)))]
        # vx = np.asarray([x[0] for x in vx_ds.sample(grid_coords)])
        # vy = np.asarray([x[0] for x in vy_ds.sample(grid_coords)])
        # q = ax0.quiver(np.ravel(XX), np.ravel(YY), vx, vy)
        # r = patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
        #                          visible=False)
        # ax0.legend([r], ['          '], framealpha=0.5, loc='upper left').set_zorder(1)
        # qk = ax0.quiverkey(q,  X=.07, Y=.8625, U=100, label=r'$100\ \frac{m}{yr}$', labelpos='E', labelsep=.05, fontproperties={'size':8}, coordinates = 'axes', zorder=1e5)
        # ax0.set_xlim(left,right)
        # ax0.set_ylim(bottom,top)
        # divider = make_axes_locatable(ax0)
        # cax = divider.append_axes("right", size=size, pad=pad)
        # cax.axis('off')
        # ax0.set_ylabel('Northing (km)')
        # ax0.set_xlabel('Easting (km)')   
        # ax0.xaxis.tick_top() 
        # ax0.xaxis.set_label_position('top') 
        # ax0.xaxis.set_ticks_position('both')
        # ax0.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: f'{int(x * 1e-3)}'))
        # ax0.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: f'{int(x * 1e-3)}'))

        # ax2 = fig.add_subplot(gs[2,0])
        # v = max(np.abs(np.nanmin(smb)), np.nanmax(smb))
        # ax2.plot(verts_x[:,:],verts_y[:,:],'k',lw=.15)
        # c = ax2.scatter(cx, cy, c=1e-3*smb, vmin=-3, vmax=3, cmap='RdBu', s=s)
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("right", size=size, pad=pad)
        # fig.colorbar(c, cax=cax, orientation='vertical', label='Modeled annual\nmass balance (m w.e.)')
        # ax2.set_ylabel('Northing (km)')
        # ax2.set_xticklabels([])
        # ax2.xaxis.set_ticks_position('both')
        # ax2.set_aspect('equal')
        # ax2.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: f'{int(x * 1e-3)}'))

        # ax3 = fig.add_subplot(gs[3,0])
        # ax3.plot(verts_x[:,:],verts_y[:,:],'k',lw=.15)
        # c = ax3.scatter(cx, cy, c=h, cmap='YlGnBu', vmin=200, vmax=1000, s=s)
        # c = ax3.scatter(rdata.x, rdata.y, c=rdata.h, cmap='YlGnBu', s=s, vmin=0, vmax=1000, zorder=100)
        # divider = make_axes_locatable(ax3)
        # cax = divider.append_axes("right", size=size, pad=pad)
        # fig.colorbar(c, cax=cax, orientation='vertical', label='Ice thickness (m)')
        # # ax3.set_xlabel('Easting (m)')
        # ax3.set_xticklabels([])
        # ax3.set_ylabel('Northing (km)')
        # ax3.set_aspect('equal')
        # ax3.xaxis.set_ticks_position('both')
        # ax3.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: f'{int(x * 1e-3)}'))

        # line = 2
        # ax4 = fig.add_subplot(gs[4,0])
        # dist = np.zeros(cx.shape[0])
        # dist[1:] = np.cumsum(np.sqrt(np.diff(cx[:,line]) ** 2.0 + np.diff(cy[:,line]) ** 2.0))*1e-3
        # diff = np.abs(elev[:,line] - ela)
        # l1, = ax4.plot(dist, elev[:,line], c='k',zorder=1e7)
        # l2, = ax4.plot(dist, elev[:,line] - h[:,line], c='tab:gray',zorder=1e7)
        # bed = elev[:,line] - h[:,line]
        # ax4.fill_between(dist, bed, elev[:,line], color='tab:blue', alpha=0.5,zorder=1e4)
        # ax4.fill_between(dist, bed, elev[:,line], color='white', alpha=0.5,zorder=1e4)
        # d0 = (np.sqrt((cx[0,line] - coords_in[line,0]) ** 2.0 + (cy[0,line] - coords_in[line,1]) ** 2.0))*1e-3
        # i0 = (np.abs(dist - d0)).argmin()
        # l3 = ax4.vlines(x=d0, color='tab:gray', ls='--', lw=1, zorder=-1000, ymin=bed[i0], ymax=elev[i0,line])
        # l3.set_zorder(1e6)
        # idx = dist[diff.argmin()]       
        # l4 = ax4.axvline(x=idx, color='tab:brown', ls='--', lw=1, zorder=1e5)
        # ax4.set_ylabel('Elevation (m)')
        # ax4.set_xlabel('Distance from seed points (km)')
        # # ax4.set_xlim([dist.min()-1.5,dist.max()+1.5])
        # ax4.set_ylim([-500, round(elev[:,line].max(),-3)])
        # ax4.invert_xaxis()
        # # ax4.grid(zorder=1)
        # ax4.legend(handles=[l3,l4,l1,l2], labels=['Known Thickness', 'Equilibrium Line', 'IFSAR surface', 'Modeled bed'], framealpha=1, loc='upper left', ncol=2).set_zorder(1e8)
        # divider = make_axes_locatable(ax4)
        # cax = divider.append_axes("right", size=size, pad=pad)
        # cax.axis('off')

        # # add hillshade to top four plots
        # ls = colors.LightSource(azdeg=315, altdeg=15)
        # z_tmp = dem_ds.read(1, window=rio.windows.from_bounds(left, bottom, right, top, dem_ds.transform))
        # Z_hillshade = ls.hillshade(z_tmp,vert_exag=1000, dx=right - left,dy=top - bottom)
        # ax0.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        # ax1.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        # ax2.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        # ax3.imshow(Z_hillshade,extent=(left, right, bottom, top),cmap=plt.cm.gray)
        # fig.suptitle(f'Mass balance gradient = {mb} mm w.e./m\nELA = {ela} m\ndh/dt = {dhdt} m/yr', fontsize=10)
        # fig.tight_layout()
        # plt.subplots_adjust(wspace=0, hspace=0)
        # if idx >0:
        #     plt.show()
        #     # fig.savefig(path[:-4] + '.jpg', dpi=200)

    # export output xyz points
    cx = np.ravel(cx, order='F')
    cy = np.ravel(cy, order='F')
    h = np.ravel(h, order='F')

    out = np.column_stack((cx,cy,h))
    out_df = pd.DataFrame(data=out, columns=['x','y','h'])
    out_df.to_csv(out_name)
    print('point cloud exported to:\t' + str(out_name))

# execute if run as a script
if __name__ == '__main__':
    main()