import base_network
import torch
import torch.nn as nn
import numpy as np 
                    
class block(nn.Sequential):
    def __init__(self,  conv2d, wn, n_feats, kernel_size, padding, dilation, n_blocks):
    	
        body = [
            base_network.ResBlock(
                conv2d, wn, n_feats, kernel_size, padding, dilation
            ) for _ in range(n_blocks)
        ]
        
        super(block, self).__init__(*body)    	
                     	
                   

class reduce_D(nn.Sequential):
    def __init__(self,  conv2d, wn, input_feats, output_feats, kernel_size):
    	
        body =  [wn(conv2d(input_feats, output_feats, kernel_size))]  
        
        super(reduce_D, self).__init__(*body) 
                

class left_right_group(nn.Module):
    def __init__(self, conv2d, wn, n_feats, kernel_size, n_blocks, padding, dilation):
        super(left_right_group, self).__init__()
        
        body = [
            block(
                conv2d, wn, n_feats, kernel_size, padding, dilation,  n_blocks
            ) 
        ] 

        self.body = nn.Sequential(*body)
        
    def forward(self, x):
        
        out = self.body(x) + x         
        return out 
                        	
class group(nn.Module):
    def __init__(self, conv2d, wn, n_feats, kernel_size, n_blocks):
        super(group, self).__init__()
        
        self.left_one_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1)         
        self.left_two_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1) 
        self.left_three_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1)   

        self.right_one_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1)         
        self.right_two_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1) 
        self.right_three_body = left_right_group(conv2d, wn, n_feats, kernel_size, n_blocks, padding = 1, dilation =1)  
                
        self.fusion_one =  reduce_D(conv2d, wn, n_feats*3, n_feats, kernel_size = 1)
        self.fusion_two =  reduce_D(conv2d, wn, n_feats*3, n_feats, kernel_size = 1)     
        self.fusion_three =  reduce_D(conv2d, wn, n_feats*3, n_feats, kernel_size = 1)
                                             
        self.gamma = nn.Parameter(torch.ones(3,3))
        self.gamma_inter = nn.Parameter(torch.ones(3))
                
    def forward(self, group_one, group_two, group_three):

        intra_group_one = self.left_one_body(group_one)
        intra_group_two = self.left_two_body(group_two)
        intra_group_three = self.left_three_body(group_three)
        
        fusion_one = self.fusion_one(torch.cat([intra_group_one*self.gamma[0,0], intra_group_two*self.gamma[0,1], intra_group_three*self.gamma[0,2]], 1))                        
        fusion_two = self.fusion_two(torch.cat([intra_group_one*self.gamma[1,0], intra_group_two*self.gamma[1,1], intra_group_three*self.gamma[1,2]], 1))                        
        fusion_three = self.fusion_three(torch.cat([intra_group_one*self.gamma[2,0], intra_group_two*self.gamma[2,1], intra_group_three*self.gamma[2,2]], 1))                        
         
        inter_group_one = self.right_one_body(fusion_one)	         
        inter_group_two = self.right_two_body(fusion_two)	       
        inter_group_three = self.right_three_body(fusion_three)	
        
        inter_group_one = torch.add(inter_group_one, group_one)
        inter_group_two = torch.add(inter_group_two, group_two)        
        inter_group_three = torch.add(inter_group_three, group_three)                       
        
        group = torch.cat([inter_group_one.unsqueeze(2)*self.gamma_inter[0], inter_group_two.unsqueeze(2)*self.gamma_inter[1], inter_group_three.unsqueeze(2)*self.gamma_inter[2]], 2)
                   
        return  group
         	         
class Unit(nn.Module):
    def __init__(self,  conv3d, conv2d,  wn, n_feats, kernel_size, act):
        super(Unit, self).__init__()      		
        	

        self.threeunit = base_network.threeUnit(conv3d, wn, n_feats,  act=act)       
        
    def forward(self, x):

        out = self.threeunit(x) + x
        
        return out[:,:,0,:,:], out[:,:,1,:,:], out[:,:,2,:,:]
        
                 	        
class pre_Net(nn.Module):
    def __init__(self, args, conv2d=base_network.default_conv2d, conv3d=base_network.default_conv3d):
        super(pre_Net, self).__init__()
        
        self.scale = args.upscale_factor        
        n_feats = args.n_feats 
        res_scale = args.res_scale    
        n_blocks = args.n_blocks

        self.n_module = args.n_module
                  
        kernel_size = 3
        act = nn.ReLU(inplace=True)
        self.gamma_rnn = nn.Parameter(torch.ones(2))        
               
        wn = lambda x: torch.nn.utils.weight_norm(x)                
                        
        # define head module for band group      	

        self.head_group_one = wn(conv2d(1, n_feats, kernel_size))        
        self.head_group_two = wn(conv2d(3, n_feats, kernel_size))        
        self.head_group_three =wn(conv2d(3, n_feats, kernel_size))
              
        
        # define body module for  band group
        self.GFM_intra_one = group(conv2d, wn, n_feats, kernel_size, n_blocks) 
        self.GFM_intra_two = group(conv2d, wn, n_feats, kernel_size, n_blocks) 
                                            
                       
#        ## define unit module 
        self.GFM_inter_one = Unit(conv3d, conv2d, wn, n_feats, kernel_size, act)
        self.GFM_inter_two = Unit(conv3d, conv2d, wn, n_feats, kernel_size, act)            
                      
        # define tail module
        tail = [
            base_network.Upsampler(conv2d, wn, self.scale, n_feats, act=False),
            wn(conv2d(n_feats, 1, kernel_size))
        ]
        self.tail = nn.Sequential(*tail)     
        
        
        rnn_reduceD = [wn(conv2d(n_feats*2, n_feats, kernel_size=1))]       
        self.rnn_reduceD = nn.Sequential(*rnn_reduceD)               
                          		              
        conv = [wn(conv2d(n_feats, n_feats, kernel_size=3))]       
        self.conv = nn.Sequential(*conv) 
        
        last_reduceD = [wn(conv2d(n_feats*3, n_feats, kernel_size=1))]       
        self.last_reduceD = nn.Sequential(*last_reduceD)  
               
               
    def forward(self, band_index, x, bicu, neigbor_l, neigbor_r, first_state = None):

        ## initial feature extraction
        x = x.unsqueeze(1)
        group_one =  self.head_group_one(x)
        group_two =  self.head_group_two(torch.cat([neigbor_l[:,1,:,:].unsqueeze(1), x, neigbor_r[:,0,:,:].unsqueeze(1)], 1))                       
        group_three =  self.head_group_three(torch.cat([neigbor_l[:,0,:,:].unsqueeze(1), x, neigbor_r[:,1,:,:].unsqueeze(1)], 1)) 
        skip =	group_one
        res = []
        res.append(group_one)
        res.append(group_two)
        res.append(group_three)        
        
        ## feature extraction for fusion
        out = self.GFM_intra_one(group_one, group_two, group_three)                                 
        group_one, group_two, group_three = self.GFM_inter_one(out)        	        

        group_one = torch.add(group_one, res[0])
        group_two = torch.add(group_two, res[1])
        group_three = torch.add(group_three, res[2])

        out = self.GFM_intra_two(group_one, group_two, group_three)
        group_one, group_two, group_three = self.GFM_inter_two(out) 
        
        group_one = torch.add(group_one, res[0])
        group_two = torch.add(group_two, res[1])
        group_three = torch.add(group_three, res[2])      

 
        out = torch.cat([group_one, group_two, group_three], 1)
        out = self.last_reduceD(out)
        out = self.conv(out)     
        out = out + skip       
           
        if band_index != 0:    
           out = torch.cat([self.gamma_rnn[0]*out, self.gamma_rnn[1]*first_state], 1)
           out = self.rnn_reduceD(out)
          
        first_state = out                         
        out = out + skip
        
        ## image reconstruction 
        out = self.tail(out)
        out = out.squeeze(1)
        out = torch.add(out.squeeze(1), bicu.squeeze(1))    
        return out, first_state
        
