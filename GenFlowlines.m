%{
Create Glacier Flowlines
Brandon S. Tober
20220828
%}

clc; clear all; close all;
% load x velocity 
dat_path            = 'C:\Users\btober\OneDrive\Documents\MARS\targ\modl\mass_con\ruth\data\';
[vx,R]              = readgeoraster(strcat(dat_path, 'ALA_G0120_0000_vx_clip.tif'));
vx                  = double(vx);
% replace NaN values if any exist
vx(vx == -32767)    = NaN;

% load y velocity 
[vy,R]              = readgeoraster(strcat(dat_path, 'ALA_G0120_0000_vy_clip.tif'));
vy                  = double(vy);
% replace NaN values if any exist
vy(vy == -32767)    = NaN;

% create meshgrid from x and y raster bounds
% the x and y array bounds may have to be modified here. for ITS_LIVE
% rasters, the array columns start at the north, but the y-extents are
% reversed.
if R.ColumnsStartFrom == 'north'
    if R.YWorldLimits(1) > R.YWorldLimits(2)
        yExtent = [R.YWorldLimits(1), R.YWorldLimits(2)];
    else
        yExtent = [R.YWorldLimits(2), R.YWorldLimits(1)];
    end
elseif R.ColumnsStartFrom == 'south'
     if R.YWorldLimits(1) < R.YWorldLimits(2)
        yExtent = [R.YWorldLimits(1), R.YWorldLimits(2)];
    else
        yExtent = [R.YWorldLimits(2), R.YWorldLimits(1)];
     end  
else
    disp('Unexpected Raster Y-extetnt sorting')
end

x                   = linspace(R.XWorldLimits(1), R.XWorldLimits(2), R.RasterSize(2));
y                   = linspace(yExtent(1), yExtent(2), R.RasterSize(1));
[X,Y]               = meshgrid(x,y);

% load upglacier seed pts - this was created in qgis by exporting the x-y
% coordinates of the two vertices between a simple line created upglacier
% seed_pts            = csvread(strcat(dat_path, 'seed_pts/1.csv'),1,0);
seed_pts            = csvread(strcat(dat_path, 'seed_pts/seeds_122322.csv'),1,0);
% sort by y-coordinate
seed_pts            = sortrows(seed_pts, 1, 'ascend');

xs = seed_pts(:,1);
ys = seed_pts(:,2);

% create evenly spaced nodes between seed endpoints
N = 15;
xcoords = linspace(xs(1), xs(2), N);
ycoords = linspace(ys(1), ys(2), N);


% stepSize            = 1050; % [m]  810     distance between seedpoints in meters
% xcoords = xs(1);
% ycoords = ys(1);

% % loop through vertices and make equidistant flowline seed points between  them
% for i=1:length(seed_pts) - 1
%     
%     % first determine number of points based on desired spacing
%     if i == 1
%         dist                = sqrt((xs(i+1) - xs(i))^2 + (ys(i+1) - ys(i))^2 );
%         % get angle between two endpoints
%         theta               = atan((ys(i+1) - ys(i))/(xs(i+1) - xs(i)));     
%     elseif i > 1
%         dist                = sqrt((xs(i+1) - xcoords(end))^2 + (ys(i+1) - ycoords(end))^2 );
%         % get angle between two endpoints
%         theta               = atan((ys(i+1) - ycoords(end))/(xs(i+1) - xcoords(end)));     
%     end
%     N                   = idivide(dist, int16(stepSize));
% 
%     % get operator based on theta - if theta is positive, we add below,
%     % negative we subtract
%     if theta > 0
%         op = 1;
%     elseif theta < 0
%         op = -1;
%     end
% 
%     % if on last segment, add one seed point to go to last vertex
%     if i==length(seed_pts)-1
%         N=N+1;
%     end
%     % loop through steps and add appropriate x and y components
%     for j = 1:N
%         xcoords(end+1) = xcoords(end) + (op*stepSize*cos(theta));
%         ycoords(end+1) = ycoords(end) + (op*stepSize*sin(theta));
%     end
% end
%% manually edit last two seeds
% xcoords = xs;
% ycoords = ys;
% xcoords = xcoords(1:5);
% ycoords = ycoords(1:5);
% xcoords(6) = xs(3);
% ycoords(6) = ys(3);
%%
% generate streamlines using stream2 function
step                = 1;
maxvert             = 500;
verts               = stream2(X, Y, vx, vy, xcoords, ycoords, [step maxvert]);
%% plot
plot(xs,ys,'.','MarkerSize',10)
hold on
plot(xcoords,ycoords,'.')
streamline(verts)
%% get vertices for each flowline and export vx vy arrays
nverts              = 75;
nverts              = 300;
% [s,d]               = cellfun(@size,verts);
% nverts              = min(s);

for i = 1:length(verts)
    for j=1:nverts
        if(isnan(verts{i}(j,1)))
            continue
        end
        tmp((i*nverts)+j, 1) = verts{i}(j,1);
        tmp((i*nverts)+j, 2) = verts{i}(j,2);
        xout(j,i) = verts{i}(j,1);
        yout(j,i) = verts{i}(j,2);
    end
end
%% export each output array as csv
out_dir = strcat(dat_path,'sensitivity/', num2str(N), '_seeds');
mkdir(out_dir);
writematrix(xout,strcat(out_dir, '/verts_x.csv'));
writematrix(yout,strcat(out_dir, '/verts_y.csv'));