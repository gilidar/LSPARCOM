%% Run LSPARCOM for contest

read_new = 1;
t_im_pix = 100; %training pixel size

if read_new
%% GET DATA
%1LS 
name = '1LS';
im_pix = 100; SIZE = 128; %T = 10001; 
dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\1LS\sequence\';
% Timings for MaxSIZE = 32: 13.124339 sec, 13.384467 sec --> avg 13.2544 sec

% %1HD 
% name = '1HD';
% im_pix = 100; SIZE = 128; %T = 1001; 
% dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\1HD\sequence\';
% % Timings for MaxSIZE = 32: 6.891216 sec, 6.948177 sec --> avg 6.9197 sec

% %3LS 
% name = '3LS';
% im_pix = 100; SIZE = 200; %T = 10000; 
% dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\3LS\sequence\';
% % % Timings for MaxSIZE = 128: 18.24 sec, 18.38 sec --> 18.31 avg sec

% %3HD 
% name = '3HD';
% im_pix = 100; SIZE = 200; %T = 702; 
% dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\3HD\sequence\';
% % % Timings for MaxSIZE = 64: 11.858730 sec, 11.815689 sec --> avg 11.8372 sec

Files=dir(dir_); isd = [Files.isdir]; Files = Files(isd==0,:);
T = length(Files);
im = zeros(SIZE,SIZE,T,'single');
   for k=1:T
       t = Tiff([dir_ Files(k).name],'r');
       im(:,:,k) = read(t);
   end

end
%% Parameters 
UPfactor = 4; 
show = 0; overlap = 1; preprocessA = 0; apsvd = 0; vec = 0;

MaxSIZE = 32;
% MaxSIZE = 64;
% MaxSIZE = 128;

%% RUN LSPARCOM
    
NET_WEIGHTS = load('weights750TU.mat'); numfolds = 10;  %12 hours of training, 750 epochs, LR = 1e-4, lambda = 1, sigma0 = 1,1. 9058 parameters, 1166 considering radial constraint  - checkfreePARAMS(29)
tic
RES_LTU = LSPARCOM_IMGV(NET_WEIGHTS, numfolds, MaxSIZE, im, UPfactor, show, 0, apsvd, vec, overlap, preprocessA, 2);
toc

figure; imagesc(RES_LTU); colormap 'hot'
getPOS(RES_LTU, t_im_pix/UPfactor, [name 'TU'])

%%%%
%% Incompatiable pixel size
% %2LS 
% name = '2LS';
% im_pix = 150; SIZE = 128; %T = 12002; 
% dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\2LS\sequence\';

% %2HD 
% name = '2HD';
% im_pix = 150; SIZE = 128; %T = 204; 
% dir_ = 'E:\DATA\POSTDOC\TrainingData_SMLM_2D\TEST\REALISTIC_SIMULATION\2HD\sequence\';
