function x_d = distortPoints(x, D, K)
% Applies lens distortion D(2x1) to 2D points x(2xN) on the image plane.

k1 = D(1); k2 = D(2);

u0 = K(1,3);
v0 = K(2,3);

xp = x(1,:)-u0; yp = x(2,:)-v0;


r2 = xp.^2 + yp.^2;
xpp = u0+xp .* (1 + k1*r2 + k2*r2.^2);
ypp = v0+yp .* (1 + k1*r2 + k2*r2.^2);

x_d = [xpp; ypp];

end

