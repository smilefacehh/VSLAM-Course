function [projected_points] = projectPoints(points_3d, K, D)
% Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
% distortion coefficients (4x1).

% if distortion vector D is missing, assume zero distortion
if nargin <= 2
    D = zeros(4,1);
end

% get image coordinates
projected_points = K * points_3d;
projected_points = projected_points  ./ projected_points (3,:);

% apply distortion
projected_points = distortPoints(projected_points(1:2,:),D,K);
end
