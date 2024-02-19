import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from shapely import geometry
from scipy.interpolate import griddata
import sys, os, argparse, configparser, json
import matplotlib.path as path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''
FlowlineMassConModel.py

author: Brandon S. Tober
date: 01SEPT2022
updated: 09FEB2024

Mass conservation approach to deriving glacier thickness along flowlines following methods of McNabb et al., 2012

inputs:
verts_x: csv file containing x-coordinates along each flowline
verts_y: csv file containing y-coordinates along each flowline
vx_ds: x-component surface velocity geotiff file (same projection as verts_x & verts_y)
vy_ds: y-component surface velocity geotiff file (same projection as verts_x & verts_y)
dem_ds: digital elevation model geotiff file (same projection as verts_x & verts_y)
rdata: geopackage/csv with thickness measurements
mb: mass balance gradient (mm w.e./m)
ela: equilibrium line altitude (m)
dhdt: surface elevation change rate (m/yr)
gamma: factor relating observed surface velocity to depth-averaged glacier velocity
gridres: spatial resolution of output gridded data

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


# grid xyz data and export geotif
def grid(x, y, z, res=100, outline_csv=None, epsg=None, outpath=None, debug=False):
    if epsg and outpath:

        # ravel coords
        x_rav = np.ravel(x)
        y_rav = np.ravel(y)
        z_rav = np.ravel(z)
        if x.shape!=y.shape!=z.shape:
            raise('grid error: input arrays are not of the same shape')

        if outline_csv and os.path.exists(outline_csv):
            # if outline csv read in coordinates - this should be x,y coordinate pairs
            outline_df = pd.read_csv(outline_csv)
            outlinex = outline_df.iloc[:,0]
            outliney = outline_df.iloc[:,1]
            x = np.append(x,outlinex)
            y = np.append(y,outliney)
            z = np.append(z,np.zeros(len(outlinex)))
        else:
            outline_csv = None

        # create meshgrid for output points
        minx = np.nanmin(x_rav)
        maxx = np.nanmax(x_rav)
        miny = np.nanmin(y_rav)
        maxy = np.nanmax(y_rav)
        x_out = np.arange(minx,maxx,res)
        y_out = np.arange(maxy,miny,-1*res)
        X_out,Y_out = np.meshgrid(x_out,y_out)
        # grid thickness
        z_out = griddata(points=np.column_stack((x_rav,y_rav)), values=z_rav, xi=(X_out,Y_out), method='linear')
        # ravel output and set values outside outer flowbanks to nan
        x_rav = np.ravel(X_out)
        y_rav = np.ravel(Y_out)
        z_rav = np.ravel(z_out)
        # mask out pixesl outside ROI - use csv outline if present, otherwise use outer flowbands
        if outline_csv:
            xy_mask = np.column_stack((outlinex,outliney))
        else:
            # create mask from flowband column 0,-1
            xy_mask = np.column_stack((np.append(x[:,0],x[:,-1][::-1]), np.append(y[:,0],y[:,-1][::-1])))
            xy_mask = np.vstack((xy_mask,np.column_stack((x[0,0],y[0,0]))))
        mask = path.Path(xy_mask)
        # get xy points inside mask
        insidepts = mask.contains_points(np.column_stack((x_rav, y_rav)))
        # set points outside to nan
        z_rav[~insidepts] = np.nan
        # reshape to 2d
        z_out = z_rav.reshape(z_out.shape)
        if debug:
            fig,ax = plt.subplots(1)
            cm=ax.imshow(z_out,extent=[minx,maxx,miny,maxy])
            ax.plot(xy_mask[:,0],xy_mask[:,1],'r', label = 'Outline mask')
            for c in range(x.shape[1]):
                ax.plot(x[:,c],y[:,c],'k', ls='none', marker='x', lw=1, ms=2,label=r'h$_{i,j}$')
            h, l = ax.get_legend_handles_labels()
            h = [h[0],h[x.shape[1]]]
            l = [l[0],l[x.shape[1]]]
            ax.legend(h, l,borderaxespad=0,fancybox=False)
            ax.set_xlabel('easting')
            ax.set_ylabel('northing')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='2%', pad='3%')
            cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
            cbar.set_label(label='Ice Thickness (m)', labelpad=5.5)
            fig.tight_layout()
            plt.show()  
        # export geotif
        driver = "GTiff"
        dim = z_out.shape
        count = 1
        height = dim[0]
        width = dim[1]
        crs = rio.crs.CRS.from_epsg(epsg)  # Use the correct EPSG code
        dtype = z_out.dtype
        transform = rio.transform.from_origin(minx, maxy, res, res)
        if not outpath.endswith('.tif'):
            outpath += '.tif'
        with rio.open(outpath, 'w',
                        driver=driver,
                        height=height,
                        width=width,
                        count=count,
                        dtype=dtype,
                        crs=crs,
                        transform=transform) as dst:
            dst.write_band(1, z_out)
        
        return 


# sample raster at all x,y locations using two 2d x,y input arrays
def sample_2d_raster(x, y, r_ds, mean=False):
    # if mean - we will get mean of pixels in between transverse vertices
    if mean:
        vals = np.full((x.shape[0],x.shape[1]-1), np.nan)
        for _i in range(vals.shape[1]):
            for _n in range(vals.shape[0]):
                if np.isnan(np.array((x[_n,_i], y[_n,_i], x[_n,_i+1], y[_n,_i+1]))).any():
                    continue
                else:
                    pts = getEquidistantPoints((x[_n,_i], y[_n,_i]), (x[_n,_i+1], y[_n,_i+1]), 10)
                    smpls = np.asarray([x[0] for x in r_ds.sample(pts)])
                    smpls[smpls == r_ds.nodata] = np.nan
                    vals[_n,_i] = np.nanmean(smpls)
    
    else:
        vals = np.full(x.shape, np.nan)
        # sample raster
        for _i in range(vals.shape[1]):
            # get non nan idx
            idx = np.isnan(x[:,_i])
            idx += np.isnan(y[:,_i])
            vals[~idx,_i] = np.asarray([x[0] for x in r_ds.sample(np.column_stack((x[~idx,_i], y[~idx,_i])))])

    return vals


# go through x and y flowline vertices and get midpoint locations between vertices, cell centers, cell areas
def build_mesh(verts_x, verts_y):
    # first make sure verts_x and verts_y are same size
    r, c = verts_x.shape
    if (r,c) != verts_y.shape:
        exit(1)

    # instantiate cx,cy,mx,my,dx,dy,area
    cx = np.zeros((r-1,c-1))
    cy = np.zeros((r-1,c-1))
    area = np.zeros((r-1,c-1))
    mx = np.zeros((r,c-1))
    my = np.zeros((r,c-1))
    dx = np.zeros((r,c-1))
    dy = np.zeros((r,c-1))

    # go along flowline pairs and get midpoint between each pair of vertices, deltax and deltay between consecutive vertices (used to get cell boundary length), cell area, and cell center
    for _i in range(c - 1):
        for _n in range(r):
            if _n < r - 1:
                # create lists of x,y cell vertex pairs
                x = [   verts_x[_n,_i],
                        verts_x[_n,_i+1],
                        verts_x[_n+1,_i+1],
                        verts_x[_n+1,_i]    ]
                y = [   verts_y[_n,_i],
                        verts_y[_n,_i+1],
                        verts_y[_n+1,_i+1],
                        verts_y[_n+1,_i]    ]
                # get cell area
                area[_n,_i] = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
                # get cell centroid
                cx[_n,_i] = sum(x)/4
                cy[_n,_i] = sum(y)/4

            # get midpoint x and y coordinates between vertices, also dx and dy
            mx[_n, _i] = (verts_x[_n,_i] + verts_x[_n,_i+1])/2
            my[_n, _i] = (verts_y[_n,_i] + verts_y[_n,_i+1])/2
            dx[_n, _i] = verts_x[_n,_i+1] - verts_x[_n,_i]
            dy[_n, _i] = verts_y[_n,_i+1] - verts_y[_n,_i]

    return cx, cy, mx, my, dx, dy, area


# get average input thickness between each set of flowlines
# from each set of neighboring flowlines, create a polygon and see which thickness measurements fall within, then get average, find closest midpoint along cell up/downstream boundaries to assign avg thickness
def get_input_thickness(verts_x, verts_y, mx, my, thick_gdf):
    # get thickness gdf x,y coords stacked together
    thick_coords = thick_gdf[['x','y']].to_numpy()
    r, c = mx.shape
    # instantiate thickness input array that will hold average of all thickness measurements between each set of flowlines
    h_avg = np.repeat(np.nan, c)
    # instantiate average input coordinate for each flowline pair
    coords_avg = np.full((c, 2), np.nan)
    # instantiate start_pos array, which will hold the index of the closest midpoint to the average thickness measurement for each flowband - this will be used to determine up v downstream in conserve_mass()
    start_pos = np.zeros(c, dtype=int)

    # loop through consecutive flowline vertices and make polygon, get average thickness obs and assign value to closest flowline vertice midpoint
    for _i in range(c):
        # take first set of along flowline verts, then concatenate flipped second set to make a closed polygon from
        tmpx = np.concatenate((verts_x[:,_i], np.flipud(verts_x[:,_i+1])))
        tmpy = np.concatenate((verts_y[:,_i], np.flipud(verts_y[:,_i+1])))
        # horizontally stack x and y coords
        coords = np.column_stack((tmpx, tmpy))
        poly = path.Path(coords)
        # get thickness points that fall within this poly
        idxs = poly.contains_points(thick_coords)

        # if at least one thickness obs within flowband, get average and store closest midpoint index
        if np.sum(idxs) > 0:
            # get averaged thickness
            h_avg[_i] = np.nanmean(thick_gdf['h'].iloc[idxs])
            # store average coord
            coords_avg[_i, 0] = np.nanmean(thick_gdf['x'].iloc[idxs])
            coords_avg[_i, 1] = np.nanmean(thick_gdf['y'].iloc[idxs])

            # calculate distance from average thickness measurement coord to all midpoints along flowband
            dist = ((mx[:,_i] - coords_avg[_i,0])**2 + (my[:,_i] - coords_avg[_i,1])**2)**0.5
            # save index of closest midpoint as start_pos for flowband
            start_pos[_i] = np.argmin(dist)

    return h_avg, start_pos


# create dictionary to compare modeled and observed thickness for all cells
# from each set of neighboring flowlines, create a polygon and see which thickness measurements fall within, then get average
def thickness_comp_stats(verts_x, verts_y, mx, my, hmod, thick_gdf):
    # get thickness gdf x,y coords stacked together
    thick_coords = thick_gdf[['x','y']].to_numpy()
    r, c = verts_x.shape
    # nested for loop - go
    # instantiate dictionary to hold modeled and observed thicknesses
    h_comp_dict = {}
    h_comp_dict['rdr_xmean'] = []
    h_comp_dict['rdr_ymean'] = []
    h_comp_dict['rdr_hmean'] = []
    h_comp_dict['rdr_hmedian'] = []
    h_comp_dict['rdr_hstd'] = []
    h_comp_dict['rdr_hiqr'] = []
    h_comp_dict['rdr_npts'] = []
    h_comp_dict['mod_h'] = []
    h_comp_dict['mod_dist'] = []

    # go along flowline pairs
    for _i in range(c - 1):
        for _n in range(r - 1):
            # make poly from 4 vertices
            tmpx = [verts_x[_n,_i], verts_x[_n+1,_i],verts_x[_n+1,_i+1],verts_x[_n,_i+1],verts_x[_n,_i,]]
            tmpy = [verts_y[_n,_i], verts_y[_n+1,_i],verts_y[_n+1,_i+1],verts_y[_n,_i+1],verts_y[_n,_i,]]
            # stack x and y coords
            coords = np.column_stack((tmpx, tmpy))
            # create polygon
            poly = path.Path(coords)
            # get thickness points that fall within this poly
            idxs = poly.contains_points(thick_coords)
            # if at least one thickness obs within poly, add results to h_comp_dict
            if np.sum(idxs) > 0:
                # get averaged thickness
                h_comp_dict['rdr_hmean'].append(np.nanmean(thick_gdf['h'].iloc[idxs]))
                h_comp_dict['rdr_hmedian'].append(np.nanmedian(thick_gdf['h'].iloc[idxs]))
                h_comp_dict['rdr_hstd'].append(np.nanstd(thick_gdf['h'].iloc[idxs]))
                h_comp_dict['rdr_hiqr'].append(np.nanquantile(thick_gdf['h'].iloc[idxs],.75) - np.nanquantile(thick_gdf['h'].iloc[idxs],.25))
                h_comp_dict['rdr_npts'].append(float(np.sum(idxs)))
                # store average coord
                rdr_xmean = np.nanmean(thick_gdf['x'].iloc[idxs])
                rdr_ymean = np.nanmean(thick_gdf['y'].iloc[idxs])
                h_comp_dict['rdr_xmean'].append(rdr_xmean)
                h_comp_dict['rdr_ymean'].append(rdr_ymean)

                # calculate distance from average thickness measurement coord to all midpoints along flowband
                dist = ((mx[:,_i] - rdr_xmean)**2 + (my[:,_i] - rdr_ymean)**2)**0.5
                # add results to h_comp_dict
                h_comp_dict['mod_h'].append(hmod[np.argmin(dist),_i])
                h_comp_dict['mod_dist'].append(dist[np.argmin(dist)])

    return h_comp_dict


# function to get the surface mass balance at a given elevation given a mass balance gradient (mm/m/yr) and an equilibrium line altitue (ELA; (m))
def get_smb(elev, mb, ela):
    # mass balance at a given grid cell will be the difference in elevation from the ela multiplies by the gradient
    # output smb should be negative below ela, positive above
    smb = (elev - ela) * mb

    # convert smb from mm w.e. to m ice equivalent 
    smb = smb / 1000            # mm to m
    smb = smb * 1000 / 917      # m water to m ice

    return smb      # m of ice


def conserve_mass(dx, dy, area, vx, vy, smb, dhdt, h_avg, start_pos, gamma, direction):
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

    h = np.full((dx.shape[0], dx.shape[1]), np.nan)         # thickness output array
    flux_in = []                                            # instantiate list to hold input ice flux for each flowband - based on measured thickness at a certain location
    # instantiate flux array
    q = np.full_like(dx, np.nan)
    q_ovr_h = np.full_like(dx, np.nan)

    # iterate through each flowline, get vx and vy arrays
    for _i in range(dx.shape[1]):

        # get q_ovr_h (q/h) through each centroid, determined by (vy*dx-vx*dy)*gamma
        q_ovr_h[:, _i] = np.abs((vx[:, _i] * dy[:, _i]) - (vy[:, _i] * dx[:, _i]))*gamma

        # determine input ice flux and store measured average thickness at start_pos cell
        flux_in.append(h_avg[_i] * q_ovr_h[start_pos[_i], _i])
        h[start_pos[_i], _i] = h_avg[_i]

        # iterate over all cells and get thickness - we'll use two for loops, one for going upstream and one for going downstream.
        # upstream
        if direction != "down":
            for _n in range(start_pos[_i] - 1, -1, -1):
                q[_n+1, _i] = h[_n+1, _i] * q_ovr_h[_n+1, _i]
                h[_n, _i] = (q[_n+1, _i] - (smb[_n, _i] - dhdt)*(area[_n, _i])) / q_ovr_h[_n, _i]

                # constrain positive thickness
                if h[_n, _i] < 0:
                    h[_n, _i] = np.nan
                    break

        # downstream
        if direction != "up":
            for _n in range(start_pos[_i], h.shape[0] - 1):
                # get ice flux at upstream cell boundary - gamma*l_n*h_n*v_n
                q[_n,_i] = h[_n, _i] * q_ovr_h[_n, _i]
                # solve for thickness at downstream cell boundary
                h[_n+1, _i] = (q[_n,_i] + (smb[_n, _i] - dhdt)*(area[_n, _i])) / q_ovr_h[_n+1, _i]

                # constrain positive thickness
                if h[_n+1, _i] < 0:
                    h[_n+1, _i] = np.nan
                    break

    print(f'Total input ice flux: {np.sum(flux_in)*1e-9} km^3/year')

    return h


def main():
    # Set up CLI
    parser = argparse.ArgumentParser(
    description='''Program conserving mass and calculating ice thickness along glacier flowlines\nNot all arguments are set up for command line input. Edif input files in the configuration file\n\n
                    Example call: /Users/btober/Drive/work/ruth/mass_con/data''',
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('conf', help='path to configuration file (.ini)', type=str)
    parser.add_argument('-mb', dest = 'mb', help='mass balance gradient (mm w.e./m/y)', type=float, nargs='?')
    parser.add_argument('-ela', dest = 'ela', help='equilibrium line altitude (m)', type=float, nargs='?')
    parser.add_argument('-dhdt', dest = 'dhdt', help='surface elevation change rate (m/yr)', type=float, nargs='?')
    parser.add_argument('-gamma', dest = 'gamma', help='factor relating surface velocity to depth-averaged velocity', type=float, nargs='?')
    parser.add_argument('-out_name', dest = 'out_name', help='output file name with no extension (same name will be used for point cloud, stats dictionary, and gridded results but with different suffixes)', type=str, nargs='?')
    parser.add_argument('-direction', dest = 'direction', help='direction to apply mass conservation [up, down, updown], default = down', type=str, default='down', nargs='?')
    parser.add_argument('-outlinemask', dest = 'outlinemask', help='mask to use in gridding as glacier outline (csv file)', type=str, nargs='?')
    parser.add_argument('-grid', dest = 'grid', help='grid model thickness?', action='store_true')
    parser.add_argument('-debug', dest = 'debug', help='debug', action='store_true')
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
    rdata_gate = config['path']['rdata_gate']
    rdata_all = config['path']['rdata_all']
    if rdata_all == 'None':
        rdata_all = None
    out_name = config['path']['out_name']
    gamma = float(config['param']['gamma'])
    mb = float(config['param']['mb'])
    ela = float(config['param']['ela'])
    dhdt = float(config['param']['dhdt'])
    grid_ = config['param'].getboolean('grid')
    gridres = float(config['param']['gridres'])
    outlinemask = config['path']['outline']
    
    if args.gamma is not None:
        gamma = args.gamma
    if args.mb is not None:
        mb = args.mb
    if args.ela is not None:
        ela = args.ela
    if args.dhdt is not None:
        dhdt = args.dhdt
    if args.outlinemask is not None:
        outlinemask = args.outlinemask
    if args.out_name is not None:
        out_name = args.out_name
        if out_name == 'None':
            out_name = None
    if args.grid:
        grid_  = True
    direction = args.direction
    debug = args.debug

    print(f'Mass Balance Gradient:\t\t{mb} mm w.e./m/y\nEquilibrium Line Altutde:\t{ela} m\nSurface Elevation Change Rate:\t{dhdt} m/y\nGamma:\t\t\t\t{gamma}')

    # x and y vertex coordinates
    verts_x = pd.read_csv(dat_path + verts_x,header=None).to_numpy()
    verts_y = pd.read_csv(dat_path + verts_y,header=None).to_numpy()

    # x and y component surface velocities
    vx_ds = rio.open(dat_path + vx_ds, 'r')
    vy_ds = rio.open(dat_path + vy_ds, 'r')

    # check for raster size mismatch
    check_rasters(vx_ds, vy_ds)

    # load surface elevation raster
    dem_ds = rio.open(dat_path + dem_ds, 'r')

    # ensure three rasters are same crs
    if not len(set([vx_ds.crs.to_epsg(), vy_ds.crs.to_epsg(),dem_ds.crs.to_epsg()]))==1:
        print('Input raster files are not of the same coordinate system')
    else:
        epsg = vx_ds.crs.to_epsg()

    # read in thickness measurements to constrain flux along flowlines - this is currently set up to read a geopackage, but could easily by swapped for a csv file using:
    # rdata = pd.read_csv('file.csv'), so long as the csv has x, y, h columns
    rdata = gpd.read_file(dat_path + rdata_gate)
    rdata = rdata[~rdata.h.isna()]
    # projet radar data to same coordinate sys as velocity data
    rdata = rdata.to_crs(epsg=epsg)
    rdata['x'] = rdata.centroid.x
    rdata['y'] = rdata.centroid.y
    rdata = rdata.sort_values(['y'], ascending=True)

    # same thing, but read in all thickness obs to compare with modeled thicknesses - this is currently set up to read a geopackage, but could easily by swapped for a csv file using:
    if rdata_all:
        # rdata_all = pd.read_csv('file.csv'), so long as the csv has x, y, h columns
        rdata_all = gpd.read_file(dat_path + rdata_all)
        rdata_all = rdata[~rdata.h.isna()]
        # projet radar data to same coordinate sys as velocity data
        rdata_all = rdata_all.to_crs(epsg=epsg)
        rdata_all['x'] = rdata_all.centroid.x
        rdata_all['y'] = rdata_all.centroid.y
    else:
        rdata_all = rdata

    # get cell centroids and area, vertex pair midpoints, vertex pair dx and dy
    cx, cy, mx, my, dx, dy, area = build_mesh(verts_x, verts_y)
    
    if debug:
        fig,ax = plt.subplots(1)
        for c in range(verts_x.shape[1]):
            ax.plot(verts_x[:,c],verts_y[:,c],'k-', marker='x', lw=1, ms=5,label=r'v$_{i,j}$')
        for c in range(cx.shape[1]):
            ax.plot(cx[:,c],cy[:,c],'r.', ms=5, label=r'c$_{i}$')
        h, l = ax.get_legend_handles_labels()
        h = [h[0],h[verts_x.shape[1]]]
        l = [l[0],l[verts_x.shape[1]]]
        ax.legend(h, l,borderaxespad=0,fancybox=False)
        ax.set_xlabel('easting')
        ax.set_ylabel('northing')
        fig.tight_layout()
        plt.show()

    # for each set of flowband, get average input thickness  and start position - closest midpoint within flowband to average thickness measurement
    h_avg, start_pos = get_input_thickness(verts_x, verts_y, mx, my, rdata)
 
    # sample vx, vy, and elev - we'll take average raster value in between flowline vertices
    vx = sample_2d_raster(verts_x, verts_y, vx_ds, mean=True)           # take the mean x-component velocity between each set of transverse vertices
    vy = sample_2d_raster(verts_x, verts_y, vy_ds, mean=True)           # take the mean y-component velocity between each set of transverse vertices
    elev_mp = sample_2d_raster(mx, my, dem_ds)                          # take the elevtion at modpoint between each set of transverse vertices
    elev = sample_2d_raster(cx, cy, dem_ds)                             # take dem elevation value at cell's center

    # get surface mass balance
    smb = get_smb(elev, mb, ela)

    # conserve max and get along-flowline thicknesses
    h = conserve_mass(dx, dy, area, vx, vy, smb, dhdt, h_avg, start_pos, gamma, direction)

    # generate comparison dictionary with modeled v. observed thicknesses
    h_comp_dict = thickness_comp_stats(verts_x, verts_y, mx, my, h, rdata_all)

    if out_name:
        # export output xyz points
        mx_ = np.ravel(mx, order='F')
        my_ = np.ravel(my, order='F')
        h_ = np.ravel(h, order='F')
        elev_mp = np.ravel(elev_mp, order='F')
        # subtrace thickness from surface elevation to get bed elevation
        z_ = elev_mp - h_                                

        out = np.column_stack((mx_,my_,h_,z_))
        out_df = pd.DataFrame(data=out, columns=['x','y','h','z'])
        out_df.to_csv(out_name + '.csv')
        print('point cloud exported to:\t' + str(out_name + '.csv'))


        with open(f'{out_name}_h_model_v_obs.json', 'w') as fp:
            json.dump(h_comp_dict, fp)
            print(f'model v. observed thickness stats exported to:\t{out_name}_h_model_v_obs.json')

            # # example of reading model comparison dict
            # with open(f'{out_name}_h_model_v_obs.json', 'r') as fp:
            #     stats_dict = json.load(fp)
            #     ks = list(stats_dict.keys())

        
        # generate gridded output
        if grid_:
            grid(mx, my, h, res=gridres, epsg=epsg, outline_csv=outlinemask, outpath = f'{out_name}_h_{epsg}_{gridres}m_grid.tif', debug=debug)
            print(f'gridded model thickness exported to:\t{out_name}_h_{epsg}_{gridres}m_grid.tif')


# execute if run as a script
if __name__ == '__main__':
    main()