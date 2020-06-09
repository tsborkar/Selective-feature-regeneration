import argparse
import numpy as np
from PIL import Image
import h5py
import caffe
from matplotlib import pyplot as plt
caffe.set_device(0)
caffe.set_mode_gpu()
img_crop = 224
np.random.seed(1000)
import json

label_path = "imagenet_labels.json"
with open(label_path) as f:
    label_dict = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--input',default="test_image_icecream.JPEG",help='path to the input image')
parser.add_argument('--dnn', default='ResNet152',choices=['CaffeNet', 'VGG_F', "GoogLeNet", "VGG16","ResNet152"], help='DNN arch to be used')
parser.add_argument('--load', help='path to trained caffemodel and weights')
parser.add_argument('--attack',default='UAP',
                    choices=["UAP","FFF","SFool","NAG","GAP","GUAP","sPGD"],help='Type of universal attack to use')
parser.add_argument('--defense', default='True', choices=['True', 'False'], help='Switch between baseline (no defense) and our proposed defense')
args = parser.parse_args()

if not args.load:
    print('Error: Model weights not provided')
else:
    if args.defense=='True':
        model_proto = 'Prototxt/'+args.dnn+'/deploy_'+args.dnn.lower()+'_FRU.prototxt'
    else:
        model_proto = 'Prototxt/'+args.dnn+'/deploy_'+args.dnn.lower()+'.prototxt'

    fig, ax = plt.subplots(1,2)
    net = caffe.Classifier(model_proto, args.load, caffe.TEST)

    img = Image.open(args.input).convert("RGB")
    w, h = img.size

    # image resizing for VGG16 maintains the original aspect ratio of the image
    if args.dnn=="VGG16":
        if w >= h:
            img = img.resize((256*w//h, 256))
        else:
            img = img.resize((256, 256*h//w))
    else:
        img = img.resize((256,256))


    img = np.transpose(np.asarray(img), (2, 0, 1))
    img = img[[2, 1, 0], :, :]
    img = img.astype(np.float32)
    img = img[:, (img.shape[1] - img_crop) // 2:(img.shape[1] + img_crop) // 2,
               (img.shape[2] - img_crop) // 2:(img.shape[2] + img_crop) // 2]

    ax[0].imshow(np.uint8(np.transpose(img,(1,2,0)))[:,:,[2,1,0]])
    ax[0].axis('off')
    ax[0].set_title('Original image')

    if args.attack=='UAP':
        # Model specific UAP examples are provided for UAP
        adv_img = h5py.File('Attack_samples/UAP/'+args.dnn+'/'+args.dnn.lower()+'_adv_iter_1.h5','r')
        # Transpose operation performed as MATLAB h5 format is C x W x H
        adv_img = np.transpose(adv_img['v'][:], (0,2,1))
        img += adv_img
        # No clipping was used during training
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='FFF':
        adv_img = np.load('Attack_samples/FFF/perturbation_caffenet_mean.npy')
        adv_img = np.transpose(adv_img[0,1:225,1:225,:],(2,0,1))
        img += adv_img[[2,1,0],:,:]
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='NAG':
        adv_img = np.load('Attack_samples/NAG/perturbation19.npy')
        adv_img = np.transpose(adv_img, (2,0,1))
        img += adv_img[[2,1,0], :, :]
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='SFool':
        adv_img = h5py.File('Attack_samples/SFool/alexnet_singular_uap_5_c3.h5','r')
        adv_img = adv_img['v'][:,1:225, 1:225]
        img += adv_img
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='GAP':
        adv_img = h5py.File('Attack_samples/GAP/res152_guided_GAP_epoch_10.h5','r')
        adv_img  = adv_img['v']
        adv_img = 16*adv_img[0,:]
        adv_img = adv_img[[2,1,0],:,:]
        adv_img = adv_img[:,30:254,30:254]
        img += adv_img
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='GUAP':
        adv_img = np.load('Attack_samples/GUAP/resnet152_with_data.npy')
        print adv_img.shape
        adv_img = adv_img[0,:]
        adv_img = np.transpose(adv_img,(2,0,1))
        adv_img = 1.5*adv_img[[2,1,0],:,:]
        img += adv_img
        img = np.clip(img, 0.0, 255.0)

    elif args.attack=='sPGD':
        adv_img = h5py.File('Attack_samples/sPGD/spgd_resnet152.h5','r')
        img += adv_img['v'][:]
        img  = np.clip(img, 0.0, 255.0)
    #


    ax[1].imshow(np.uint8(np.transpose(img,(1,2,0)))[:,:,[2,1,0]])
    ax[1].axis('off')
    ax[1].set_title('Adversarially perturbed image')
    img = img[np.newaxis,:]

    # Mean subtraction values for ResNet152v2 model are different than the other 4 models provided
    if args.dnn=='ResNet152':
        img[:, 0, :, :] -= 102.98
        img[:, 1, :, :] -= 115.947
        img[:, 2, :, :] -= 122.772
    else:
        img[:, 0, :, :] -= 103.939
        img[:, 1, :, :] -= 116.779
        img[:, 2, :, :] -= 123.68

    net.blobs['data'].reshape(*img.shape)
    net.blobs['data'].data[...] = img

    net.forward()
    pred = net.blobs['prob'].data[0]
    pred_ids = np.argsort(pred)[-5:][::-1]

    print("Top 1 prediction: ",  label_dict[str(pred_ids[0])][1], ", Confidence score: ", str(np.max(pred)))
    print("Top 5 predictions: ", [label_dict[str(pred_ids[k])][1] for k in range(5)])
    plt.show()
