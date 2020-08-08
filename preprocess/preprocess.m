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

% Dilation disc definition
se = strel('disk',25);
% Define 2D filter in order to smooth edges
width = 50;
kernel = ones(width) / width^2;

for k = 1:numel(files)
    f = fullfile(path,files(k).name);
    D = load(f);
    % Eliminate image of size 256x256
    if size(D.cjdata.image,1)==256
        continue
    end
    
    % Load tumor outline (part of the given dataset)
    tumor_bound = boundarymask(D.cjdata.tumorMask);
    % Extract skull outline
    skull_bound = mask_extraction(D.cjdata.image, se, kernel);
    
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
    imwrite(uint16(D.cjdata.image),...
            fullfile(folder_s,strrep(files(k).name,'.mat','.png')))
    imwrite(output_mask,...
            fullfile(folder_t,strrep(files(k).name,'.mat','.png')))
end
