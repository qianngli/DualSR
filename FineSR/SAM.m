function sam = SAM(imagery1,imagery2)
tmp = (sum(imagery1.*imagery2, 3) + eps) ...
      ./ (sqrt(sum(imagery1.^2, 3)) + eps) ./ (sqrt(sum(imagery2.^2, 3)) + eps);
sam = mean2(real(acos(tmp)));
% this ignores the pixels that have zero norm (all values zero - no color)
sam = sam*180/pi;
end
