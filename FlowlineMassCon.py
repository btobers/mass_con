import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from shapely import geometry
from scipy.interpolate import griddata
import sys, os, argparse, configparser, json
import matplotlib.path as path


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
def grid(x, y, z, res=100, epsg=3413, outpath=None):
    # create meshgrid for output points
    minx = np.nanmin(x)
    maxx = np.nanmax(x)
    miny = np.nanmin(y)
    maxy = np.nanmax(y)
    x_out = np.arange(minx,maxx,100)
    y_out = np.arange(maxy,miny,-100)
    X_out,Y_out = np.meshgrid(x_out,y_out)
    # grid thickness and std
    z_out = griddata(points=np.column_stack((x,y)), values=z, xi=(X_out,Y_out), method='linear')
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
        # get thickness points that fill within this poly
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
def thickness_comp_stats(verts_x, verts_y, mx, my, thick_gdf):
    print('THIS FUNCTION IS UNDER CONSTRUCTION!')
    return
    # get thickness gdf x,y coords stacked together
    thick_coords = thick_gdf[['x','y']].to_numpy()
    r, c = mx.shape
    # instantiate dictionary to hold modeled and observed thicknesses
    h_comp_dict = {}
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
        # get thickness points that fill within this poly
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
    parser.add_argument('-out_name', dest = 'out_name', help='output point cloud file name', type=str, nargs='?')
    parser.add_argument('-direction', dest = 'direction', help='direction to apply mass conservation [up, down, updown], default = down', type=str, default='down', nargs='?')
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
    v_ds = config['path']['v']
    dem_ds = config['path']['dem']
    rdata = config['path']['rdata']
    out_name = config['path']['out_name']
    gamma = float(config['param']['gamma'])
    mb = float(config['param']['mb'])
    ela = float(config['param']['ela'])
    dhdt = float(config['param']['dhdt'])
    
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
        if (out_name != 'None') and (not out_name.endswith('.csv')):
                out_name = out_name.split('.')[0] + '.csv'
    if out_name == 'None':
        out_name = None
    if args.plot:
        plot = args.plot    
    direction = args.direction

    print(f'Mass Balance Gradient:\t\t{mb} mm w.e./m/y\nEquilibrium Line Altutde:\t{ela} m\nSurface Elevation Change Rate:\t{dhdt} m/y\nGamma:\t\t\t\t{gamma}')

    # x and y vertex coordinates
    verts_x = pd.read_csv(dat_path + verts_x,header=None).to_numpy()
    verts_y = pd.read_csv(dat_path + verts_y,header=None).to_numpy()

    # trim points outside gorge
    # verts_x = verts_x[:-185,:]
    # verts_y = verts_y[:-185,:]

    # remove first flowline - seems to be some issues on output of this one, perhaps too close to gorge edge
    # verts_x = verts_x[:,:-1]
    # verts_y = verts_y[:,:-1]

    # x and y component surface velocities
    vx_ds = rio.open(dat_path + vx_ds, 'r')
    vy_ds = rio.open(dat_path + vy_ds, 'r')
    v_ds = rio.open(dat_path + v_ds, 'r')

    # check for raster size mismatch
    check_rasters(vx_ds, vy_ds)

    # load surface elevation raster
    dem_ds = rio.open(dat_path + dem_ds, 'r')

    # read in thickness measurements - this is currently set up to read a geopackage, but could easily by swapped for a csv file using:
    # rdata = pd.read_csv('file.csv'), so long as the csv has x, y, h columns
    rdata = gpd.read_file(dat_path + rdata)
    rdata = rdata[~rdata.h.isna()]
    # projet radar data to same coordinate sys as velocity data - change epsg to appropriate numerical id in following line
    rdata = rdata.to_crs(epsg=3413)
    rdata['x'] = rdata.centroid.x
    rdata['y'] = rdata.centroid.y
    rdata = rdata.sort_values(['y'], ascending=True)

    # get cell centroids and area, vertex pair midpoints, vertex pair dx and dy
    cx, cy, mx, my, dx, dy, area = build_mesh(verts_x, verts_y)

    # for each set of flowband, get average input thickness  and start position - closest midpoint within flowband to average thickness measurement
    h_avg, start_pos = get_input_thickness(verts_x, verts_y, mx, my, rdata)
 
    # sample vx, vy, and elev - we'll take average raster value in between flowline vertices
    vx = sample_2d_raster(verts_x, verts_y, vx_ds, mean=True)           # take the mean x-component velocity between each set of transverse vertices
    vy = sample_2d_raster(verts_x, verts_y, vy_ds, mean=True)           # take the mean y-component velocity between each set of transverse vertices
    elev_mp = sample_2d_raster(mx, my, dem_ds)                          # take the elevtion at modpoint between each set of transverse vertices
    elev = sample_2d_raster(cx, cy, dem_ds)                             # take dem elevation value at cell's center

    # verts_x[-90:,-1] = np.nan
    # verts_y[-90:,-1] = np.nan
    # cx[-90:,-1] = np.nan
    # cy[-90:,-1]=np.nan

    # get surface mass balance
    smb = get_smb(elev, mb, ela)

    # conserve max and get along-flowline thicknesses
    h = conserve_mass(dx, dy, area, vx, vy, smb, dhdt, h_avg, start_pos, gamma, direction)

    # generate gridded output
    grid(mx, my, h, res=100, epsg=3414, outpath = 'h_grid.tif')

    # generate comparison dictionary with modeled v. observed thicknesses
    h_comp_dict = thickness_comp_stats(verts_x, verts_y, mx, my, rdata)
    with open('h_model_v_obs.json', 'w') as fp:
        json.dump(h_comp_dict, fp)

    # trim unreasonable thicknesses - we'll set anything greater than 950 m thick to nan, as our deepest amp thickness meaurements are ~920 m
    # h[h > 940] = np.nan

    if out_name:
        print(out_name)
        # export output xyz points
        mx = np.ravel(mx, order='F')
        my = np.ravel(my, order='F')
        h = np.ravel(h, order='F')
        elev_mp = np.ravel(elev_mp, order='F')
        # subtrace thickness from surface elevation to get bed elevation
        z = elev_mp - h                                

        out = np.column_stack((mx,my,h,z))
        out_df = pd.DataFrame(data=out, columns=['x','y','h','z'])
        out_df.to_csv(out_name)
        print('point cloud exported to:\t' + str(out_name))

# execute if run as a script
if __name__ == '__main__':
    main()