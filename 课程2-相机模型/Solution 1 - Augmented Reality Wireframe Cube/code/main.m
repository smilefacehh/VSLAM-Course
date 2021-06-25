close all;
clear all;

% Load camera poses
% Each row i of matrix 'poses' contains the transformations that transforms
% points expressed in the world frame to points expressed in the camera frame.

pose_vectors = load('../data/poses.txt');

% Define 3D corner positions

% [Nx3] matrix containing the corners of the checkerboard as 3D points
% (X,Y,Z), expressed in the world coordinate system

square_size = 0.04;
num_corners_x = 9; num_corners_y = 6;
num_corners = num_corners_x * num_corners_y;

[X, Y] = meshgrid(0:num_corners_x-1, 0:num_corners_y-1);
p_W_corners = square_size * [X(:) Y(:)];
p_W_corners = [p_W_corners zeros(num_corners,1)]';

% Load camera intrinsics

K = load('../data/K.txt'); % calibration matrix      [3x3]
D = load('../data/D.txt'); % distortion coefficients [2x1]

% Load one image with a given index
img_index = 1;

img = rgb2gray(imread(['../data/images/',sprintf('img_%04d.jpg',img_index)]));

% Project the corners on the image

% Compute the 4x4 homogeneous transformation matrix that maps points from the world
% to the camera coordinate frame
T_C_W = poseVectorToTransformationMatrix(pose_vectors(img_index,:));

% Transform 3d points from world to current camera pose
p_C_corners = T_C_W * [p_W_corners; ones(1,num_corners)];
p_C_corners = p_C_corners(1:3,:);

projected_pts = projectPoints(p_C_corners, K, D);

figure()
imshow(img); hold on;
plot(projected_pts(1,:), projected_pts(2,:), 'r.');
hold off;


% Undistort image with bilinear interpolation
tic;
img_undistorted = undistortImage(img,K,D,1);
disp(['Undistortion with bilinear interpolation completed in ' num2str(toc)]);

% Vectorized undistortion without bilinear interpolation
tic;
img_undistorted_vectorized = undistortImageVectorized(img,K,D);
disp(['Vectorized undistortion completed in ' num2str(toc)]);

figure();
subplot(1, 2, 1);
imshow(img_undistorted);
title('With bilinear interpolation');
subplot(1, 2, 2);
imshow(img_undistorted_vectorized);
title('Without bilinear interpolation');

% Draw a cube on the undistorted image
offset_x = 0.04 * 3; offset_y = 0.04;
s = 2 * 0.04;
[X, Y, Z] = meshgrid(0:1, 0:1, -1:0);
p_W_cube = [offset_x + X(:)*s, offset_y + Y(:)*s, Z(:)*s]';

p_C_cube = T_C_W * [p_W_cube; ones(1,8)];
p_C_cube = p_C_cube(1:3,:);

cube_pts = projectPoints(p_C_cube, K, zeros(4,1));

figure();
imshow(img_undistorted); hold on;

lw = 3;

% base layer of the cube
line([cube_pts(1,1), cube_pts(1,2)],[cube_pts(2,1), cube_pts(2,2)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,1), cube_pts(1,3)],[cube_pts(2,1), cube_pts(2,3)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2), cube_pts(1,4)],[cube_pts(2,2), cube_pts(2,4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3), cube_pts(1,4)],[cube_pts(2,3), cube_pts(2,4)], 'color', 'red', 'linewidth', lw);

% top layer
line([cube_pts(1,1+4), cube_pts(1,2+4)],[cube_pts(2,1+4), cube_pts(2,2+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,1+4), cube_pts(1,3+4)],[cube_pts(2,1+4), cube_pts(2,3+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2+4), cube_pts(1,4+4)],[cube_pts(2,2+4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3+4), cube_pts(1,4+4)],[cube_pts(2,3+4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);

% vertical lines
line([cube_pts(1,1), cube_pts(1,1+4)],[cube_pts(2,1), cube_pts(2,1+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,2), cube_pts(1,2+4)],[cube_pts(2,2), cube_pts(2,2+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,3), cube_pts(1,3+4)],[cube_pts(2,3), cube_pts(2,3+4)], 'color', 'red', 'linewidth', lw);
line([cube_pts(1,4), cube_pts(1,4+4)],[cube_pts(2,4), cube_pts(2,4+4)], 'color', 'red', 'linewidth', lw);

hold off;
set(gca,'position',[0 0 1 1],'units','normalized')

