function aug = tumor_aug(imag) 
% tumor_aug Performs random augmentation of binary mask of brain tumor
%   aug = tumor_aug(imag) returns the mask in imag after random affine
%   transformation
%
%   imag - Binary mask of a brain tumor
%
%   aug - Binary mask of the tumor after random affine transformation
%         including rotation, shear and scaling

    % Bring the tumor to center
    [imag, ind_c] = bring2center(imag);
    % Uniformly aquire affine transformation (no translation)
    ax = (0.25*rand+1)*0.8;  % Scaling factor
    ay = (0.25*rand+1)*0.8;  % Scaling factor
    b = (rand-0.5)*0.5;      % Shear factor
    c = (rand-0.5)*30;       % Rotation angle
    
    T = affine2d([ ax  0   0;...
                   0   ay  0;...
                   0   0   1] );    
    imag = bring2center(imwarp(imag,T, 'OutputView', imref2d(size(imag))));
    imag = bring2center(imrotate(imag, c, 'crop'));    
    T = affine2d([ 1 b 0;...
                   b 1 0;...
                   0 0 1] );
    % Apply affine transformation, retain original dimenstions
    imag = bring2center(imwarp(imag,T,'Outputview', imref2d(size(imag))));
    [ri, rj] = ind2sub(size(imag), find(imag(:)));
    ind_r = [ri, rj];
    ind_r = ind_r - size(imag)/2 + ind_c;
    aug = zeros(512);
    aug(sub2ind(size(imag), ind_r(:, 1), ind_r(:, 2))) = 1;    
end

function [c_im, center] = bring2center(im)
% bring2center centralized binary object in im
%   [c_im, center] = bring2center(im)
%   im - Binary image of an object 
%
%   c_im - Binary image with centralized object
%   center - Original center coordinates

    c_im = zeros(size(im));
    [x, y] = ind2sub(size(im), find(im(:)));
    ind = [x, y];
    center = round(mean(ind));
    ind = ind - center + size(im)/2;
    c_im(sub2ind(size(im), ind(:, 1), ind(:, 2))) = 1;
end

