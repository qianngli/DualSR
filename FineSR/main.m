clc 
clear 
upscale_factor = 4;
srPath = ['data/CAVE/', num2str(upscale_factor), '/CoarSR/'];
lsrPath = ['data/CAVE/', num2str(upscale_factor/2), '/CoarSR2/'];
srFile=fullfile(srPath);
srdirOutput=dir(fullfile(srFile));
srfileNames={srdirOutput.name}';
number = length(srfileNames)-2;

SR_PSNR = 0;
SR_SAM = 0;
SR_SSIM = 0;
    
sr_PSNR = 0;
sr_SAM = 0;
sr_SSIM = 0;
    
for index = 1:length(srfileNames)
    name = char(srfileNames(index));
    if(isequal(name,'.')||... % 去除系统自带的两个隐文件夹
            isequal(name,'..'))
        continue;
    end
    disp(['---upscale_factor:',num2str(upscale_factor),'----deal with:',num2str(index-2),'----name:',name]);
    singlePath= [srPath, name];
    lsinglePath= [lsrPath, name];
    
    load(singlePath)
    SRZ = SR;
    load(lsinglePath)
    LR = SR;
    [sr, psnr] = backprojection(SRZ, LR, 1, HR);

    SR_PSNR = SR_PSNR + PSNR(SRZ, HR);
    SR_SAM = SR_SAM + SAM(SRZ, HR);
    SR_SSIM = SR_SSIM + SSIM(SRZ, HR);
    
    sr_PSNR = sr_PSNR + PSNR(sr, HR);
    sr_SAM = sr_SAM + SAM(sr, HR);
    sr_SSIM = sr_SSIM + SSIM(sr, HR);
  
end
disp(['---upscale_factor:',num2str(upscale_factor),'----No Post-processing----PSNR:',num2str(SR_PSNR/number),...
      '----SSIM:',num2str(SR_SSIM/number),'----SAM:',num2str(SR_SAM/number)]);
 disp(['---upscale_factor:',num2str(upscale_factor),'----Post-processing-------PSNR:',num2str(sr_PSNR/number),...
       '----SSIM:',num2str(sr_SSIM/number),'----SAM:',num2str(sr_SAM/number)]);

% 
