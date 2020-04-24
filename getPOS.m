function [] = getPOS(im, pix_size, name)

[SIZE,~] = size(im);
xnano = {}; ynano = {}; intensity = {};

for i = 1:SIZE
    for j = 1:SIZE
       if im(i,j)>0 %emitter detected
           x = pix_size/2 + (j-1)*pix_size;
           y = pix_size/2 + (i-1)*pix_size;
           v = im(i,j);
           xnano = cat(1, xnano, {x});
           ynano = cat(1, ynano, {y});
           intensity = cat(1,intensity, {v});
       end
    end
end

T = table(xnano, ynano, intensity);
writetable(T,[name '.csv'])


end

