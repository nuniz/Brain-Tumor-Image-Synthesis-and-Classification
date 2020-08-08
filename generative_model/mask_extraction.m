% Edited by Marom Dadon
% 18/05/2020
% Data Preprocessing - Skull outline extraction

clear all
close all
clc

% Data folder path (.mat)
path = 'brain_tumor/';

% Files in directory
files = dir(fullfile(path,'*.mat')); % pattern to match filenames.
im_dim = zeros(2, numel(files));

% Dilation disc definition
se_c = strel('disk',25);


for k = 1:numel(files)
    f = fullfile(path,files(k).name);
    D = load(f);
    target = D.cjdata.image;
    % Eliminate image of size 256x256
    if size(target,1)==256
        continue
    end
    
    % Load tumor outline (part of the given dataset)
    tumor_bound = boundarymask(D.cjdata.tumorMask);
    bw = imbinarize(target);   %thresholded to remove background and white/gray matter
    
    % Fill holes inside the binary image
    insideSkull = imfill(bw, 'holes'); 
    mask = imfill(imclose(insideSkull,se_c),'holes');
    
    % Define 2D filter in order to smooth edges
    width = 50;
    kernel = ones(width) / width^2;
    blurryImage = conv2(double(mask), kernel, 'same');
    smooth_mask = blurryImage > 0.5;
    
    % Draw the boundary and translate to binary mask
    boundary = bwboundaries(smooth_mask,'noholes');
    skull_bound = zeros(size(target));
    idx = sub2ind(size(target),boundary{1}(:,1), boundary{1}(:,2));
    skull_bound(idx) = 1;
    skull_bound = logical(skull_bound);
    % Combine skull and tumor outlines
    output_mask = skull_bound + tumor_bound;
    
    % Save in Pix2Pix input dataset format
    folder_s = fullfile(num2str(D.cjdata.label), 'A');
    if ~exist(folder_s, 'dir')
        mkdir(folder_s)
    end
    folder_t = fullfile(num2str(D.cjdata.label), 'B');
    if ~exist(folder_t, 'dir')
        mkdir(folder_t)
    end
    imwrite(uint16(target),...
            fullfile(folder_s,strrep(files(k).name,'.mat','.png')))
    imwrite(output_mask,...
            fullfile(folder_t,strrep(files(k).name,'.mat','.png')))
end
