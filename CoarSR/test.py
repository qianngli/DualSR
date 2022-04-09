import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import  opt
from data_utils import is_image_file
from model_two import pre_Net
import scipy.io as scio  
from eval import PSNR, SSIM, SAM
from scipy.misc import imresize
import pdb
import time  
                   
def main():

    input_path = '/media/hdisk/liqiang/hyperSR/test/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/' 
    out_path = '/media/hdisk/liqiang/hyperSR/result/' +  opt.datasetName + '/' + str(opt.upscale_factor) + '/' + opt.method + '/'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)      
    PSNRs = []
    SSIMs = []
    SAMs = []


    if not os.path.exists(out_path):
        os.makedirs(out_path)
                    
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = pre_Net(opt)

    if opt.cuda:
        model = nn.DataParallel(model).cuda()    
        
    checkpoint  = torch.load(opt.model_name)
    model.load_state_dict(checkpoint['model']) 
    model.eval()       
    
    images_name = [x for x in listdir(input_path) if is_image_file(x)]           
       
    for index in range(len(images_name)):

        mat = scio.loadmat(input_path + images_name[index]) 
        LR = mat['LR'].astype(np.float32).transpose(2,0,1)
        HR = mat['HR'].astype(np.float32).transpose(2,0,1)  

        bic = np.zeros(HR.shape, dtype=np.float32)    
                    
        for i in range(bic.shape[0]):
            bic[i,:,:] = imresize(LR[i,:,:], (bic.shape[1], bic.shape[2]), 'bicubic', mode='F')  
            	
        bicu = Variable(torch.from_numpy(bic).float(), volatile=True).contiguous().view(1, -1, bic.shape[1], bic.shape[2])                  	            	      	    	        	
       
        input = Variable(torch.from_numpy(LR).float(), volatile=True).contiguous().view(1, -1, LR.shape[1], LR.shape[2])              
        SR = np.array(HR).astype(np.float32) 


        first_state = None
              	
        if opt.cuda:
            input = input.cuda()
            bicu =bicu.cuda()

        neigbor_l = []
        neigbor_r = []

        for  i in range(input.shape[1]):
        	            
            if i==0:
                                
                neigbor_l.append(input[:,3,:,:].data.unsqueeze(1))
                neigbor_l.append(input[:,1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,2,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,4,:,:].data.unsqueeze(1))
                	 
            elif i==1:
               
                neigbor_l.append(input[:,3,:,:].data.unsqueeze(1))
                neigbor_l.append(input[:,0,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,2,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,4,:,:].data.unsqueeze(1))
                	                	                	
            elif i==input.shape[1]-2:

                neigbor_l.append(input[:,i-3,:,:].data.unsqueeze(1))
                neigbor_l.append(input[:,i-1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i+1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i-2,:,:].data.unsqueeze(1))
                	
                
                	                	                
            elif i==input.shape[1]-1:
                neigbor_l.append(input[:,i-3,:,:].data.unsqueeze(1))
                neigbor_l.append(input[:,i-1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i-2,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i-4,:,:].data.unsqueeze(1))
                	
            else:
                neigbor_l.append(input[:,i-2,:,:].data.unsqueeze(1))
                neigbor_l.append(input[:,i-1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i+1,:,:].data.unsqueeze(1))
                neigbor_r.append(input[:,i+2,:,:].data.unsqueeze(1))
                	       	    	            	 	              	         
            x = input[:,i,:,:] 
            single_bicu = bicu[:,i,:,:]  
            left =  Variable(torch.cat(neigbor_l, 1)) 
            right =  Variable(torch.cat(neigbor_r, 1)) 		                 	                                    	
            output, first_state = model(i, x, single_bicu, left, right, first_state)                          
                    
            SR[i,:,:] = output.cpu().data[0].numpy() 
      
                   
        SR[SR<0] = 0             
        SR[SR>1.] = 1.
        psnr = PSNR(SR, HR)
        ssim = SSIM(SR, HR)
        sam = SAM(SR, HR)
        
        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        
        SR = SR.transpose(1,2,0)   
        HR = HR.transpose(1,2,0)  
        	                    
        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR':SR})  
        print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index+1,  psnr, ssim, sam, images_name[index]))                 
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs))) 

if __name__ == "__main__":
    main()
