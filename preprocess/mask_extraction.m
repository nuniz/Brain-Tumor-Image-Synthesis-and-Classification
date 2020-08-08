function skull_bound = mask_extraction(imag, se, kernel)
% mask_extraction Extracts skull boundary out of grayscale MRI image
%   skull_bound = mask_extraction(imag, se, kernel)
%
%   imag - Grayscale MRI image
%   se - Morphological closing structure
%   kernel - Morphological structure for lowpass filter
%
%   skull_bound - Binary skull outline

    % Otsu's thresholded to remove background and white/gray matter
    bw = imbinarize(imag);   
    
    % Fill holes inside the binary image
    insideSkull = imfill(bw, 'holes'); 
    mask = imfill(imclose(insideSkull,se),'holes');
    
    % 2D filter in order to smooth edges
    blurryImage = conv2(double(mask), kernel, 'same');
    smooth_mask = blurryImage > 0.5;
    
    % Draw the boundary and translate to binary mask
    boundary = bwboundaries(smooth_mask,'noholes');
    skull_bound = zeros(size(target));
    idx = sub2ind(size(target),boundary{1}(:,1), boundary{1}(:,2));
    skull_bound(idx) = 1;
    skull_bound = logical(skull_bound);
end

