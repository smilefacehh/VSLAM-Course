function T = poseVectorToTransformationMatrix(pose_vec)
% Converts a 6x1 pose vector into a 4x4 transformation matrix

omega = pose_vec(1:3);
t = pose_vec(4:6);

theta = norm(omega);
k = omega/theta;
kx = k(1); ky = k(2); kz = k(3);
K = [[0,-kz,ky];[kz,0,-kx];[-ky,kx,0]];

R = eye(3) + sin(theta) * K + (1-cos(theta)) * K^2;

T = eye(4);
T(1:3,1:3) = R;
T(1:3,4) = t;

end

