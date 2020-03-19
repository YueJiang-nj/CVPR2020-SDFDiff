import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torchvision

from models import Encoder, Decoder, Refiner
from dataset import Dataset

import torch.nn as nn
import matplotlib.pyplot as plt
from differentiable_rendering import differentiable_rendering, grid_construction_sphere_big, loss_fn, calculate_sdf_value
import math
from random import *

i = 0
num_rec = 0
num_epochs = 0
sample = 0
factor = 15
avg = 1000000
directory = "../result/"
overall_loss = []
latent_loss = []
rec_loss = []
num_rec_list = []
num_print = 0

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

camera_list = []
angle = 0
h = 3
for i in range(24):
    camera_list.append(Tensor([math.cos(angle) * math.sqrt(25-h**2), h, math.sin(angle) * math.sqrt(25-h**2)]))
    angle += math.pi / 12


def generate_samples(test_loader, shape_encoder, shape_decoder, sketch_encoder, args):
    with torch.no_grad():

        sketch, shape = next(iter(test_loader))

        torchvision.utils.save_image(sketch[0], "./" + directory + "/sketch" + str(num_rec) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        sketch_feature, _ = sketch_encoder(sketch)
        sketch_out = shape_decoder(sketch_feature)
        shape_feature = shape_encoder(shape)
        shape_out = shape_decoder(shape_feature)
        images = torch.cuda.FloatTensor(24, 1, args.image_res, args.image_res) ####
        print("----", F.mse_loss(shape, sketch_out), shape.shape[0])
        # print(out.shape[0])
        for i in range(18, 26):
            # self.fake_image[i,0,:,:] = differentiable_rendering(fake_sdf[i,0,:,:,:], self.fake_sdf.shape[-1], self.opt.crop_size, camera_list[self.index[i]])
            images[i-18,0,:,:] = differentiable_rendering(sketch_out[0,0,:,:,:], sketch_out.shape[-1], args.image_res, camera_list[i])
            images[i-10,0,:,:] = differentiable_rendering(shape_out[0,0,:,:,:], shape_out.shape[-1], args.image_res, camera_list[i])
            images[i-2,0,:,:] = differentiable_rendering(shape[0,0,:,:,:], shape.shape[-1], args.image_res, camera_list[i])
    return images


def train(train_loader, encoder, decoder, refiner, optimizer, args):
    global i
    global num_rec
    global overall_loss
    global latent_loss
    global rec_loss
    global factor
    global avg
    global num_print
    j=0
    loss_curr = 0
    latent_loss_curr = 0
    rec_loss_curr = 0
    avg_curr = 0
    num_img = 20

    for image, shape in train_loader:
        if image.shape[0] < num_img:
            break    

        for iteration in range(1): 
            j += 1 

            shape = shape.cuda()
            image = image.cuda()

            optimizer.zero_grad()
            
            latent = encoder(image)
            result = decoder(latent)
            result = refiner(result)

            img_res = 256 

            images = torch.cuda.FloatTensor(num_img * 2, 1, img_res, img_res) ####
            show_images = torch.cuda.FloatTensor(num_img, 1, img_res, img_res)
            
            loss = 0 

            rand = torch.randint(0, 24, (num_img,))
            if j % 100 == 1:
              for i in range(num_img):
                  cam = rand[i]
                  images[i,0,:,:], _ = differentiable_rendering(result[i,0,:,:,:], result.shape[-1], img_res, camera_list[cam])
                  images[i+num_img,0,:,:], _ = differentiable_rendering(shape[i,0,:,:,:], shape.shape[-1], img_res, camera_list[cam])
                  if j % 100 == 1:  
                      if i % 2 == 0:
                          show_images[int(i/2),0,:,:] = images[i,0,:,:] 
                          show_images[int(i/2) + int(num_img / 2),0,:,:] = images[i+num_img,0,:,:]
              if j % 100 == 1: 
                  grid = make_grid(show_images, nrow=int(num_img/2))
                  torchvision.utils.save_image(grid, "../result/" + args.category + "/train_" + str(num_rec) + "_" + str(j) + ".png", nrow=6, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

            obj_loss = 0

            # narrow band
            mask = torch.abs(result[0,0]) < 0.1
            mask = mask.float()

            # sdf loss
            image_loss, sdf_loss = loss_fn(images[:num_img][0,0], images[num_img:][0,0], result[0,0] * mask, 4/64., 64, 64, 64, 256, 256)
            obj_loss += sdf_loss / (64**3) * 0.02
 
            # laplancian loss
            conv_input = (result[0,0] * mask).unsqueeze(0).unsqueeze(0)
            conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
            Lp_loss = torch.sum(F.conv3d(conv_input, conv_filter) ** 2) / (64**3)
            obj_loss += Lp_loss * 0.02

            # image loss
            obj_loss += F.mse_loss(images[:num_img], images[num_img:]) * 15 * (256 * 256 / img_res / img_res)

            # back probagate
            loss = obj_loss
            loss.backward()
            optimizer.step() 
    

def Train(args):
    save_filename = './models/{0}'.format(args.output_folder)

    train_dataset = Dataset(True, args.category) 

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=int(6), shuffle=True)  

    encoder = EncoderSimple(args).to(args.device)
    decoder = DecoderSimple(args).to(args.device)
    refiner = Refiner(args).to(args.device)
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(refiner.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr) 
    
    for epoch in range(args.num_epochs):  
        global i
        global num_rec
        train(train_loader, encoder, decoder, refiner, optimizer, args) 
        num_rec += 1
        
        print("======= Finished Epoch " + str(num_rec) + " =======")
 
        if num_rec % 1 == 0:
            with open("../models/" + args.category + '/ccencoder{0}.pt'.format(epoch + 1), 'wb') as f:
                torch.save(encoder.state_dict(), f)
            with open('../models/' + args.category + '/ccdecoder{0}.pt'.format(epoch + 1), 'wb') as f:
                torch.save(decoder.state_dict(), f)   
            with open('../models/' + args.category + '/ccrefiner{0}.pt'.format(epoch + 1), 'wb') as f:
                torch.save(refiner.state_dict(), f)  


if __name__ == '__main__':
    
    import argparse
    import os
    import multiprocessing as mp
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='SDFDiff')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=300, #500,#int(6678 / 14), #int(6678/14),
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=50000,
        help='number of epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=0.1,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--sdf-res', type=str, default=32,
        help='SDF resolution')
    parser.add_argument('--image-res', type=str, default=64,
        help='image resolution')
    parser.add_argument('--dataset-size', type=str, default=6678*10,
        help='the size of the dataset')
    parser.add_argument('--category', type=str, default='vessel',
        help='the category of the dataset')


    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0


    Train(args)
