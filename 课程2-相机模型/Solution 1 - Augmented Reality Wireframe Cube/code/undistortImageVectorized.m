function undimg = undistortImageVectorized(img, K, D)
  [X, Y] = meshgrid(1:size(img, 2), 1:size(img, 1));
  px_locs = [X(:)-1, Y(:)-1]';
  dist_px_locs = distortPoints(px_locs, D, K);
  
  
 
  intensity_vals = img(round(dist_px_locs(2, :)) + ...
      size(img, 1) * round(dist_px_locs(1, :)));
  undimg = uint8(reshape(intensity_vals, size(img)));
end
