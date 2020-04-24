function [RES] = LSPARCOM_IMGV(NET_WEIGHTS, numfolds, M, im, UPfactor, show, GT, apsvd, vec, overlap, preprocessA, type)                   
%% LSPRACOM via FISTA minimzation
% If the algorithm detects an emitter at position [i,j] then it places its
% estimated variance value at that position, while if no emitter is detected, 
% then a value of zero is placed instead.
%% <><><><><><><><><><><><><> INPUT ARGUMENTS<><><><><><><><><><><><><>
% M - processing block size, scalar (square input assumed), power of 2.
% im - input low-resolution image array
% UPfactor - Upscaling factor
% show - 1 to create figure with reconstruction, every other number not to
% GT - Ground truth, high-resolution reconstruction, can be zeros if show!=1
% apsvd - 1 to apply svd for background removal, 0 not to
% vec - a vector containing the cutoff indices for the SVD, matters only if apsvd = 1

%% <><><><><><><><><><><><><> Data location & reading <><><><><><><><><><><><><>
imI = sum(im,3); %Input low-resolution frame
if preprocessA
[im, sL] = preprocess(im, apsvd, vec);
else
[sL, ~, ~]= size(im);
end
sH = sL*UPfactor; %N = M*UPfactor

%% <><><><><><><><><><><><><> APPLY LSPARCOM <><><><><><><><><><><><><>
disp 'running LSPARCOM'

if sL <= M %single block
    if ~preprocessA
    [im, ~] = preprocess(im, apsvd, vec);
    end
    g = imresize(var(im,0,3), UPfactor);
    %% <><><><><><><><><><><><><><><> FISTA <><><><><><><><><><><><><><>
    if type == 1
    RES = LSPARCOM_VAR_NET(g, NET_WEIGHTS, numfolds);
    else
    RES = LSPARCOM_VAR_NET2(g, NET_WEIGHTS, numfolds);
    end
    
else %blocks with overlap
    MU = M*UPfactor;
    if overlap == 1
    c_vec = M/2:M/2:sL-M/2; %half block overlap
    imp = im;
    RES = zeros(sH,sH);
    Apod = window(@tukeywin, MU, 0.95);
    Apod = repmat(Apod.', MU, 1); Apod = Apod.*Apod.';
    else
    c_vec = M/2:M:sL-M/2; %no overlap 
    imp = im;
    RES = zeros(sH,sH);
    Apod = ones(size(MU));  
    end
    
    L = length(c_vec);
    cU_vec = c_vec*UPfactor; 
            
    count = 1;
    for i = 1:L
        for j = 1:L
            imc = imp(c_vec(i)-M/2+1:c_vec(i)+M/2,c_vec(j)-M/2+1:c_vec(j)+M/2,:);
            if ~preprocessA
            [imc, ~] = preprocess(imc, apsvd, vec);
            end
            g = imresize(var(imc,0,3), UPfactor);
            %% <><><><><><><><><><><><><><><> FISTA <><><><><><><><><><><><><><>
            if type == 1
            R1 = LSPARCOM_VAR_NET(g, NET_WEIGHTS, numfolds);
            else
            R1 = LSPARCOM_VAR_NET2(g, NET_WEIGHTS, numfolds);    
            end
            RES(cU_vec(i)-MU/2+1:cU_vec(i)+MU/2,cU_vec(j)-MU/2+1:cU_vec(j)+MU/2) = R1.*Apod + RES(cU_vec(i)-MU/2+1:cU_vec(i)+MU/2,cU_vec(j)-MU/2+1:cU_vec(j)+MU/2);
            disp(['completed block #' num2str(count) ' out of ' num2str(L^2) ' blocks']);
            count = count +1;
        end
    end
   
end

if show == 1
figure; 
subplot(1,2,1); imagesc(GT)
title 'Ground truth'
subplot(1,2,2); imagesc(RES)
title 'reconstrcuted image'
end

end

