% Edited by Marom Dadon
% 01/06/2020
% Data Augmentation 

clear all
close all
clc

% Data folder path (.mat)
label = 1;
path = ['brain_tumor/' num2str(label) '/'];
path_s = [num2str(label) '/'];

% Files in directory
files = dir(fullfile(path,'*.mat')); % pattern to match filenames.

% Dilation disc definition
se = strel('disk',25);

% Number of images to augment
n = 5000;

for i = 1:n
    
    % Load original data and skull & tumor outlines mask
    file = files(randi([1, numel(files)])).name;
    D = load(fullfile(path, file));
    target = D.cjdata.image; 
    skulltumor = load(fullfile(path_s, 'B', file));
    
    % Subtract tumor outline from skull & tumor mask
    tumor_bound = boundarymask(D.cjdata.tumorMask);
    skull_bound = skull_bound - tumor_bound;
    
    % Augment new random tumor by performing image transformations 
    new_tumor_bound = boundarymask(tumor_aug(D.cjdata.tumorMask));
    
    output_mask = skull_bound + new_tumor_bound;
    
    % Save new augmented masks
    folder = fullfile(num2str(D.cjdata.label), 'augN');
    if ~exist(folder, 'dir')
        mkdir(folder)
    end
    imwrite(output_mask, fullfile(folder, i, '.png'))
end
