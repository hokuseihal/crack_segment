import os
import torch
import torch.nn.functional as F
import multiprocessing
from dataset import MetaOWNCrackDataset,MetaCOCOMaskDataset
from operation import maml_operation,own_test
from loss.loss import DiceLoss
from models.unet import UNet as Model
from utils import savedic,save,addvalue

def train(args):
    dataset=MetaCOCOMaskDataset('VOC2017/train2017','VOC2017/val2017','VOC2017/annotations/instances_train2017.json','VOC2017/annotations/instances_val2017.json',1,args.num_shots,(args.size,args.size))
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
    testdataset=MetaOWNCrackDataset('owncrack',args.num_shots,(args.size,args.size))
    testdataloader=torch.utils.data.DataLoader(testdataset,batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
    model = Model()
    model.to(device=args.device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf=DiceLoss()
    writer=[]
    # Training loop
    for e in range(args.num_epoch):
        maml_operation(args,'train',{'model':model,'optimizer':meta_optimizer,'dataloader':dataloader,'lossf':lossf,'accf':F.mse_loss,'writer':writer,'e':e})
        own_test(args,'test',{'model':model,'optimizer':meta_optimizer,'dataloader':testdataloader,'accf':F.mse_loss,'lossf':lossf,'writer':writer,'e':e})
        save(e,model,'data',writer)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str,
        help='Path to the folder the data is downloaded to.',default='data')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of examples per class (k in "k-shot", default: 5).')
    #parser.add_argument('--num-ways', type=int, default=5,
    #    help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--size', type=int, default=64,
        help='Size*2 of image (default: 64*2).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=32,
        help='Number of tasks in a mini-batch of tasks (default:32).')
    parser.add_argument('--num-batches', type=int, default=50,
        help='Number of batches the model is trained over (default: 50).')
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
        help='Number of workers for data loading (default: max).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--not-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--num-epoch',type=int,default=100)

    args = parser.parse_args()
    args.device = torch.device('cuda' if  not args.not_cuda
        and torch.cuda.is_available() else 'cpu')
    print(args.device)
    train(args)
