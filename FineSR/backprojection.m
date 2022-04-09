function [im_h, psnr] = backprojection(im_h, im_l, maxIter, HR)

upscale_factor = 2;
im_h = double(im_h);
im_l = double(im_l);

 for ii = 1:maxIter
       im_l_s = imresize(im_h, 1/upscale_factor, 'bicubic');    
      c = SAM(im_l_s, im_l);
      
      im_diff = im_l - im_l_s;
      
      if c > 1
         c = 1;
      end      
      
     im_diff = c*im_diff;        
     im_diff = imresize(im_diff, upscale_factor, 'bicubic');   
     
     im_h = im_h + im_diff;
     
     psnr(ii) = PSNR(im_h, HR);
 end