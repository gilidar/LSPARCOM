function [imout, sLi] = preprocess(im, apsvd, vec)

% Normalize movie intensity to be between [1, 256] 
MAXim = max(im(:));
if MAXim>0
im0 = 256*im/MAXim;
else
im0 = im;
end

if apsvd
%% <><><><><><>REMOVE BACKGROUND from image array<><><><><><>
im = SVDfilt(im0, 'ind', vec, 1);
else
im = im0;
end

% Remove temporal median of the movie - to remove constant background
imout  = im - repmat(median(im, 3), [1 1 size(im, 3)]); 

[sLi,~,~]= size(im); 
end

% MaxVal = max(max(mean(abs(im), 3)));