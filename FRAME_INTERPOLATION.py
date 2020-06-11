#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import model
import dataloader
import platform
from tqdm import tqdm
import torchvision
import cv2
import math
import numpy as np
import datetime


# For parsing commandline arguments
parser = argparse.ArgumentParser()

# parser.add_argument("--video", type=str, required=False, help='path of video to be converted')
parser.add_argument("--input_frames_path", type=str, required=True, help='path to frames folder to be converted')
parser.add_argument("--patch_height", type=int, default=512, required=False, help='tile height')
parser.add_argument("--patch_width", type=int, default=512, required=False, help='tile width')
parser.add_argument("--pad_px", type=int, default=64, required=False, help='padding width in pixels')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
# parser.add_argument("--output", type=str, default="output.mp4", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    return error


def get_list_of_first_level_directories(root_dir):
    entries = os.listdir(root_dir)
    all_dirs = []

    for entry in entries:
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            all_dirs.append(entry)

    return all_dirs


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")

    # Create extraction folder and extract frames
    # Assuming UNIX-like system where "." indicates hidden directories
    # extractionDir = "/" + extractionDir
    extractionDir = "./data/" + timestamp
    if not os.path.exists(extractionDir):
        os.mkdir(extractionDir)

    extractionPath = os.path.join(extractionDir, "input_tiles")
    outputPath     = os.path.join(extractionDir, "output_tiles")
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
 
    #CREATE TILES
    input_im_names=[]
    for file_name in sorted(os.listdir(args.input_frames_path)):
        input_im_names.append(os.path.join(args.input_frames_path, file_name))
        
    imgheight, imgwidth = cv2.imread(input_im_names[0], cv2.IMREAD_GRAYSCALE).shape

    fit_imgheight = math.ceil(imgheight/args.patch_height)*args.patch_height
    fit_imgwidth = math.ceil(imgwidth/args.patch_width)*args.patch_width

    global_array = np.zeros((fit_imgheight, fit_imgwidth, int(len(input_im_names))), dtype=np.uint8)
    print("global_array.shape", global_array.shape)

    for in_im_index, input_im_name in enumerate(input_im_names):
        print("input_im_name", input_im_name)
        global_array[:,:,in_im_index:in_im_index+1] = np.expand_dims(np.pad(cv2.imread(input_im_name, cv2.IMREAD_GRAYSCALE),\
             ((0,fit_imgheight-imgheight), (0, fit_imgwidth-imgwidth)), mode='constant'), axis=-1)

    global_array_pad = np.pad(global_array, ((args.pad_px,args.pad_px),(args.pad_px,args.pad_px),(0,0)), mode='constant')

    global_array_pad_height, global_array_pad_width, global_array_pad_channels = global_array_pad.shape

    for in_image_index in range(len(input_im_names)):
        print("image proccess", in_image_index)

        for i in range(0, imgheight, args.patch_height):
            for j in range(0, imgwidth, args.patch_width):
                z=0
                x02d = global_array_pad[i:i+args.patch_height+args.pad_px*2, j:j+args.patch_width+args.pad_px*2,in_image_index]

                gen_dir_final = os.path.join(extractionPath,  "Y{:0>3d}_X{:0>3d}".format(int(i/args.patch_height+1),int(j/args.patch_width+1)))

                if not os.path.exists(gen_dir_final):
                    os.makedirs(gen_dir_final)  

                cv2.imwrite(os.path.join(gen_dir_final,  "Z{:0>3d}_Y{:0>3d}_X{:0>3d}.png".format(args.sf*in_image_index+z+1,int(i/args.patch_height+1),int(j/args.patch_width+1))), x02d)
                z += 1

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.5, 0.5, 0.5]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    
    subdirectories = get_list_of_first_level_directories(extractionPath)
    print("subdirectories", subdirectories)
    
    for subdir in sorted(subdirectories):
        current_root = os.path.join(extractionPath, subdir)
        print("current_root",current_root)

        # Write to subdirectories structure
        if not os.path.exists(os.path.join(outputPath, subdir)):
            os.mkdir(os.path.join(outputPath, subdir))
            outputPath_subdir = os.path.join(outputPath, subdir)

        # Load data
        videoFrames = dataloader.Video(root=current_root, transform=transform)
        videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        flowComp = model.UNet(6, 4)
        flowComp.to(device)
        for param in flowComp.parameters():
            param.requires_grad = False
        ArbTimeFlowIntrp = model.UNet(20, 5)
        ArbTimeFlowIntrp.to(device)
        for param in ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
        flowBackWarp = flowBackWarp.to(device)



        dict1 = torch.load(args.checkpoint, map_location='cpu')
        ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])

        flowComp.load_state_dict(dict1['state_dictFC'])

        # Save models for two used U-Nets as .pt files
        # torch.save(ArbTimeFlowIntrp, "./checkpoints/ArbTimeFlowIntrp.pt")
        # torch.save(flowComp, "./checkpoints/flowComp.pt")

        # Save models for two used U-Nets as .onnx files
        # dummy_input_1 = Variable(torch.randn(args.batch_size, 20, args.patch_height+args.pad_px, args.patch_width+args.pad_px)).cuda()
        # dummy_input_2 = Variable(torch.randn(args.batch_size, 6, args.patch_height+args.pad_px, args.patch_width+args.pad_px)).cuda()

        # torch.onnx.export(ArbTimeFlowIntrp, dummy_input_1 , 'ArbTimeFlowIntrp_opset10.onnx', opset_version=10, verbose=False)
        # torch.onnx.export(flowComp, dummy_input_2 , 'flowComp_opset10.onnx', opset_version=10, verbose=False)

        # Interpolate frames
        frameCounter = 0

        with torch.no_grad():
            for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

                I0 = frame0.to(device)
                I1 = frame1.to(device)

                flowOut = flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]

                # Save reference frames in output folder
                for batchIndex in range(args.batch_size):
                    (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath_subdir, "Z{:03d}".format(frameCounter + args.sf * batchIndex) + "_{}".format(subdir) + ".png"))
                frameCounter += 1

                # Generate intermediate frames
                for intermediateIndex in range(1, args.sf):
                    t = float(intermediateIndex) / args.sf
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0

                    g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(args.batch_size):
                        (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath_subdir, "Z{:03d}".format(frameCounter + args.sf * batchIndex)+ "_{}".format(subdir) + ".png"))
                    frameCounter += 1

                    if in_image_index  == len(input_im_names)-1:
                        current_root = os.path.join(extractionPath, subdir)
                        entries = sorted(os.listdir(current_root))

                        last_frame = Image.open(os.path.join(current_root,entries[-1]))
                        last_frame.resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath_subdir, "Z{:03d}".format(frameCounter + args.sf * batchIndex) + "_{}".format(subdir) + ".png"))


                # Set counter accounting for batching of frames
                frameCounter += args.sf * (args.batch_size - 1)

if __name__ == "__main__":
    main()
