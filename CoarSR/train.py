#coding:utf-8
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from option import  opt
from model import pre_Net
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR, SSIM, SAM
from torch.optim.lr_scheduler import MultiStepLR

   
def main():

       
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
		
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    
    # Loading datasets
    train_set = TrainsetFromFolder('/media/hdisk/liqiang/hyperSR/train/'+ opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)    
    val_set = ValsetFromFolder('/media/hdisk/liqiang/hyperSR/test/' + opt.datasetName + '/' + str(opt.upscale_factor))
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size= 1, shuffle=False)
        
      
    # Buliding model     
    model = pre_Net(opt)
    criterion = nn.L1Loss() 
    
    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 
                   
    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(),  lr=opt.lr,  betas=(0.9, 0.999), eps=1e-08)    

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)         
            opt.start_epoch = checkpoint['epoch'] + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))       

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90, 140, 175], gamma=0.1, last_epoch = -1) 

    # Training 
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"])) 
        train(train_loader, optimizer, model, criterion, epoch)         
        val(val_loader, model, epoch)              
        save_checkpoint(epoch, model, optimizer) 
         
    	
def train(train_loader, optimizer, model, criterion, epoch):
	
    model.train()   
      
    for iteration, batch in enumerate(train_loader, 1):
        input, bicu, label = Variable(batch[0]),  Variable(batch[1]), Variable(batch[2], requires_grad=False)
        
        first_state = None

        if opt.cuda:
            input = input.cuda()
            bicu = bicu.cuda()
            label = label.cuda() 

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
            single_label = label[:,i,:,:]
            single_bicu = bicu[:,i,:,:]  
            left =  Variable(torch.cat(neigbor_l, 1)) 
            right =  Variable(torch.cat(neigbor_r, 1)) 	
                                   	
            SR, first_state = model(i, x, single_bicu, left, right, first_state)     
       
            first_state.detach_()
            first_state = first_state.detach()
            first_state = Variable(first_state.data, requires_grad=False)                 
            loss = criterion(SR, single_label)
            
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()         
                            
        if iteration % 20 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.data[0]))

        if opt.show:
            writer.add_scalar('Train/Loss', loss.data[0], niter) 


def val(val_loader, model, epoch):
	            	            
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    for iteration, batch in enumerate(val_loader, 1):
        input, bicu, HR = Variable(batch[0], volatile=True),  Variable(batch[1], volatile=True), Variable(batch[2])
        SR = np.zeros((HR.shape[1], HR.shape[2], HR.shape[3])).astype(np.float32) 

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
      
        val_psnr += PSNR(SR, HR.data[0].numpy()) 
        val_ssim += SSIM(SR, HR.data[0].numpy())
        val_sam += SAM(SR, HR.data[0].numpy())
           
    
    print("PSNR = {:.3f}   SSIM = {:.4F}    SAM = {:.3f}".format(val_psnr / len(val_loader), val_ssim / len(val_loader), val_sam / len(val_loader)))               
            
    
    if opt.show:
        writer.add_scalar('Val/PSNR',val_psnr, epoch)  
   
        
def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_model_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")     	
    torch.save(state, model_out_path)
 
          
if __name__ == "__main__":
    main()

