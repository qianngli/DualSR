function single_psnr = PSNR(HR_image, SR_image)
    HR_image = reshape(HR_image, size(HR_image,1)*size(HR_image, 2),  size(HR_image,3));   
    SR_image = reshape(SR_image, size(SR_image,1)*size(SR_image, 2),  size(SR_image,3));    
    L = 1;
    width = size(HR_image,2);
    height = size(HR_image,1);
    if( width ~= size(SR_image,2) || height ~= size(SR_image,1) )
        disp('Please check the input image have the same size');
        return
    end
    
    HR_image = double(HR_image);
    SR_image = double(SR_image);
    MES = mean((HR_image(:) - SR_image(:)).^2);
    single_psnr=10*log10(L^2/MES);
end