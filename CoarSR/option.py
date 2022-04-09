import argparse
# Training settings
parser = argparse.ArgumentParser(description="Super-Resolution")
parser.add_argument("--upscale_factor", default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=85, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", help="show Tensorboard")
parser.add_argument("--lr", type=int, default=1e-4, help="lerning rate")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=12, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)  checkpoint/model_epoch_95.pth")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")               
parser.add_argument("--datasetName", default="CAVE", type=str, help="data name")

# Network settings
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_bands', type=int, default=31, help='number of band')
parser.add_argument('--n_module', type=int, default=3, help='number of n_module')
parser.add_argument('--n_blocks', type=int, default=2, help='number of blocks for  band group')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')


# Test image
parser.add_argument('--model_name', default='checkpoint/CAVE_model_4_epoch_XX.pth', type=str, help='super resolution model name ')
parser.add_argument('--method', default='ASFS', type=str, help='super resolution method name')

opt = parser.parse_args() 
