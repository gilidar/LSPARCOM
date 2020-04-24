%% Create dataset for training
clearvars; close all; clc

numinstack = 350;  
data_num = 10005;
stand_prob = 1; %0.75; %0.25; %0.5;
sp_prob = (1-stand_prob)/4;
size_dat = 16;
upF = 4;
size_datN = size_dat*upF;
IM_ds = zeros(size_datN, size_datN, data_num);
GT_ds = zeros(size_datN, size_datN, data_num);

% type = 'BTLS'; 
% SIZE_in = 64;
% T = 12000; 
% num2sum = 40;   
% [IM, GT] = read_DATA_AND_GT_BATCH(type, upF, 1, T);

tiff_filename = 'TUI4.tif';
csv_filename = 'POS_TUI4.csv';
num2sum = 5;  

im_pix = 100; 
IM = ReadStackFromTiff(tiff_filename);
[SIZE_in,~,T] = size(IM);
GT = imSfromXL(csv_filename, SIZE_in*upF, im_pix/upF);

ROIsfromIM = (SIZE_in/size_dat)^2;
max_shift = size_datN/4;
maxind = SIZE_in*upF - size_datN;

jj = 1;

while jj<=data_num %Draw new image
IM_t = zeros(SIZE_in, SIZE_in, numinstack);
GT_t = zeros(SIZE_in*upF, SIZE_in*upF, numinstack); 

rotnum = round(3*rand(1,1)); 

    for ii = 1:numinstack
    vecT = randsample(T, num2sum);
    IM_t(:,:,ii) = IM_t(:,:,ii) + sum(IM(:, : ,vecT),3);
    GT_t(:,:,ii) = GT_t(:,:,ii) + sum(GT(:, :, vecT),3);
    end
     
[IM_cr, ~] = preprocess(IM_t, 0, 0);
       
GT_ds_b = rot90(var(GT_t,0,3), rotnum);
IM_ds_b = rot90(imresize(var(IM_cr,0,3), upF), rotnum);
    
    for r = 1:ROIsfromIM
        cc = 0;
        nummix = 1; %round(rand(1)+2);
        while cc<nummix
        %CROP BLOCK
        raf = rand(1);
           if raf < stand_prob
            r_i = round(randrange(1, 1, maxind));
            c_i = round(randrange(1, 1, maxind)); 
           elseif raf < stand_prob + sp_prob
            r_i = round(randrange(1, 1, max_shift));
            c_i = round(randrange(1, 1, max_shift));
          elseif raf < stand_prob + 2*sp_prob
            r_i = round(randrange(1, 1, max_shift));
            c_i = round(randrange(1, maxind - max_shift, maxind));
          elseif raf < stand_prob + 3*sp_prob
            r_i = round(randrange(1, maxind - max_shift, maxind)); 
            c_i = round(randrange(1, 1, max_shift));
          else
            r_i = round(randrange(1, maxind - max_shift, maxind)); 
            c_i = round(randrange(1, maxind - max_shift, maxind));
           end

           GT_temp =  GT_ds_b(r_i:r_i + size_datN-1, c_i:c_i + size_datN-1);
           if sum(GT_temp(:))>0
            GT_ds(:,:,jj) = GT_ds(:,:,jj) + GT_temp;
            IM_ds(:,:,jj) = IM_ds(:,:,jj) + IM_ds_b(r_i:r_i + size_datN-1, c_i:c_i + size_datN-1);
            cc = cc + 1;
           end
        end
        jj = jj + 1;
    end
end
  
