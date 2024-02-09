%{
Create Glacier Flowlines
Brandon S. Tober
20230113
%}

clc; clear all; close all;
% load x velocity raster
dat_path            = 'C:\Users\btober\OneDrive\Documents\MARS\targ\modl\mass_con\ruth\data\';
[vx,R]              = readgeoraster(strcat(dat_path, 'ALA_G0120_0000_vx_clip.tif'));
vx                  = double(vx);
% replace NaN values if any exist
vx(vx == -32767)    = NaN;

% load y velocity raster
[vy,R]              = readgeoraster(strcat(dat_path, 'ALA_G0120_0000_vy_clip.tif'));
vy                  = double(vy);
% replace NaN values if any exist
vy(vy == -32767)    = NaN;

% create meshgrid from x and y raster bounds
% the x and y array bounds may have to be modified here. for ITS_LIVE
% rasters, the array columns start at the north, but the y-extents are
% reversed. The lines below SHOULD take care of getting the proper extents
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
% coordinates of two endpoints created upglacier
seed_pts            = csvread(strcat(dat_path, 'seed_pts/seeds_122322.csv'),1,0);
% sort by x or y-coordinate if desired
seed_pts            = sortrows(seed_pts, 1, 'ascend');
% parse x and y coordinates
xs = seed_pts(:,1);
ys = seed_pts(:,2);

%% create evenly spaced nodes between seed endpoints
N = 7;
xcoords = linspace(xs(1), xs(2), N);
ycoords = linspace(ys(1), ys(2), N);

%%
% alternitavely, we can specify the spacing between seed points in meters
% (uncomment block below)


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
%%
% generate streamlines using stream2 function
step                = 1;
maxvert             = 500;
verts               = stream2(X, Y, vx, vy, xcoords, ycoords, [step maxvert]);
%% plot for inspection
plot(xs,ys,'.','MarkerSize',10)
hold on
plot(xcoords,ycoords,'.')
streamline(verts)
for i = 1:length(verts)
    plot(verts{i}(end,1),verts{i}(end,2), 'r.')
end
%% get vertices for each flowline and export x and y vertex coordinate arrays
% define how many vertices to export along each flowline
% nverts              = 75;
% nverts              = 400;
% alternatiuvely, take the minimum - number of vertices from all flowlines
[s,d]               = cellfun(@size,verts);
nverts              = min(s);

% parse x and y coordinate arrays for each flowline - ncols=# of flowlines,
% nrows=# of vertices along each flowline
for i = 1:length(verts)
    for j=1:nverts
        if j > size(verts{i},1)
            xout(j,i) = nan;
            yout(j,i) = nan;
        else
%         if(isnan(verts{i}(j,1)))
%             continue
%         end
%         tmp((i*nverts)+j, 1) = verts{i}(j,1);
%         tmp((i*nverts)+j, 2) = verts{i}(j,2);

        xout(j,i) = verts{i}(j,1);
        yout(j,i) = verts{i}(j,2);
        end
    end
end
%% export each output array as csv
out_dir = strcat(dat_path, 'sensitivity/', num2str(N), '_seeds/');
mkdir(out_dir);
writematrix(xout,strcat(out_dir, num2str(N),'verts_x.csv'));
writematrix(yout,strcat(out_dir, num2str(N),'verts_y.csv'));