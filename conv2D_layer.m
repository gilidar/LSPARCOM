function [res] = conv2D_layer(weights, input, pad_type)

%% Using Matlab's funcitons
dlX = dlarray(single(input),'SSC');
if strcmp(pad_type,'valid')
    res = dlconv(dlX, weights, 0);
else
    res = dlconv(dlX, weights, 0, 'Padding', 'same');
end
res = extractdata(res);

%% MY VERSION
% [ker,~,in_channels, out_channels] = size(weights);
% [M, N, IN_CHANNELS] = size(input);
% 
% if IN_CHANNELS~=in_channels
%     error('Filter size incompatiable with data')
% end
% 
% if strcmp(pad_type,'valid')
%     res = zeros(M-ker+1, N-ker+1, out_channels); 
% else 
%     res = zeros(M, N, out_channels); 
% end
% 
% parfor i = 1:out_channels
%     for j = 1:in_channels
%     res(:,:,i) = res(:,:,i) + filter2(squeeze(weights(:,:,j,i)), input(:,:,j), pad_type); 
%     end
% end

end

