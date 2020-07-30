import argparse
import numpy as np
import os
from os import listdir
# set to 1 or zero for debug and output information from caffe
os.environ['GLOG_minloglevel'] = '2'
from PIL import Image
import h5py
import caffe
from matplotlib import pyplot as plt
caffe.set_device(0)
caffe.set_mode_gpu()
img_crop = 224
np.random.seed(0)


base_acc = {'CaffeNet':  0.564,
            'VGG_F': 0.584,
            'GoogLeNet': 0.686,
            'VGG16': 0.684,
             'ResNet152': 0.79}




def compute_test_accuracy(y_pred, img_labl):
    top1_err = 0
    top5_err = 0

    for val_id in range(y_pred.shape[0]):
        assert(np.isnan(np.sum(y_pred[val_id,:]))==0),"\n Nan value found in prediction labels"

        # print '\n class label '+str(np.argmax(y_pred[val_id,:]))+'\t GT label '+str(img_labl[val_id,0])
        if (np.argmax(y_pred[val_id, :]) != img_labl[val_id]):
            # if y_pred_ori[val_id]==1:
            top1_err += 1
            # y_pred_ori[val_id] = 0


    for val_id in range(y_pred.shape[0]):
        y_sort = np.sort(y_pred[val_id, :])[::-1]
        assert(np.isnan(np.sum(y_sort))==0),"\n Nan value found in sorted prediction labels"


        err_flag = 0

        for s_id in range(5):
            if (np.nonzero(y_pred[val_id, :] == y_sort[s_id])[0][0]== img_labl[val_id]):
                err_flag = 1
                break
        if err_flag == 0:
            # if y_pred_ori[val_id]==1:
            top5_err += 1



    top1_acc = 1.0 -(top1_err)/float(y_pred.shape[0])
    top5_acc = 1.0 -(top5_err)/float(y_pred.shape[0])
    print '\n top 1 acc : '+str(top1_acc)
    print '\n top 5 acc : '+str(top5_acc)

    return top1_acc, top5_acc
#




def get_test_acc(net, num_test=10,use_noise=False):
    # path to Imagenet validation data
    val_folder_loc = args.input+'/Class_'

    batch_size = 200
    if args.dnn=='ResNet152':
        mini_b = 20
    elif args.dnn=='VGG16':
        mini_b = 25
    elif args.dnn == 'GoogLeNet':
        mini_b = 50
    else:
        mini_b = 100
    print '\n -------------------Testing on IMAGENET validation set ---------------------\n'
    ref_test_acc = np.empty((2,2), np.float32)
    y_pred = np.empty((1000*num_test, 1000), np.float32)
    y_label = np.empty((1000*num_test), np.float32)
    indx = 0
    print '\n------------------Original--------------------------\n'
    for batch_id in range(1000 / batch_size):
        print '\n processing batch ' + str(batch_id)
        img_data = np.empty((num_test * batch_size, 3, img_crop, img_crop), np.float32)
        img_label = np.empty((num_test * batch_size), np.float32)

        for class_id in range(batch_id * batch_size, (batch_id + 1) * batch_size):

            filelist = listdir(str(val_folder_loc + str(class_id) + '/'))
            # file_id = np.random.permutation(num_test)
            for img_id in range(num_test):

                im = Image.open(val_folder_loc + str(class_id) + '/' + filelist[img_id]).convert(
                    "RGB")
                w, h = im.size
                if args.dnn == "CaffeNet" or args.dnn=="GoogLeNet":
                    im = im.resize((256,256))
                else:
                    if w >= h:
                        im = im.resize((256 * w // h, 256))
                    else:
                        im = im.resize((256, 256 * h // w))
                img_temp = np.transpose(np.asarray(im), (2, 0, 1))
                img_temp = img_temp[[2, 1, 0], :, :]
                img_temp = img_temp.astype(np.float32)

                img_temp = img_temp[:, (img_temp.shape[1] - img_crop) // 2:(img_temp.shape[1] + img_crop) // 2,
                           (img_temp.shape[2] - img_crop) // 2:(img_temp.shape[2] + img_crop) // 2]

                if use_noise==True :
                    adv_img = h5py.File('Attack_samples/UAP/' + args.dnn + '/' + args.dnn.lower() +
                                        '_adv_iter_'+str(np.random.randint(1,6))+'.h5','r')
                    # adv_img = h5py.File('/home/tejas/universal/matlab/caffenet_adv_iter' + str(np.random.randint(1, 11)) + '.h5', 'r')
                    adv_img = np.transpose(adv_img['v'][:], (0,2,1))
                    img_temp += adv_img

                img_data[num_test * (class_id - batch_size * batch_id) + img_id, :, :, :] = img_temp.copy()
                img_label[num_test * (class_id - batch_size * batch_id) + img_id] = class_id
                indx+=1

        if args.dnn == 'ResNet152':
            img_data[:, 0, :, :] -= 102.98
            img_data[:, 1, :, :] -= 115.947
            img_data[:, 2, :, :] -= 122.772
        else:
            img_data[:, 0, :, :] -= 103.939
            img_data[:, 1, :, :] -= 116.779
            img_data[:, 2, :, :] -= 123.68


        for bsz in range(batch_size * num_test / mini_b):

            batch_in = img_data[bsz * mini_b:(bsz + 1) * mini_b, :]

            net.blobs['data'].reshape(*batch_in.shape)
            net.blobs['data'].data[...] = batch_in

            net.forward()
            y_pred[
            batch_id * num_test * batch_size + bsz * mini_b:batch_id * num_test * batch_size + (bsz + 1) * mini_b, :] = \
            net.blobs['prob'].data

        y_label[batch_size * num_test * batch_id:(batch_id + 1) * num_test * batch_size] = img_label

        del img_data, img_label

    ref_test_acc[0, 0], ref_test_acc[0, 1] = compute_test_accuracy(y_pred, y_label)
 #   out_file.create_dataset('dc_vgg', data=y_pred)
    del y_pred, y_label

    return ref_test_acc








parser = argparse.ArgumentParser()
parser.add_argument('--input',help='path to the ILSVRC2012 validation set root folder')
parser.add_argument('--dnn', default='CaffeNet',choices=['CaffeNet', 'VGG_F', "GoogLeNet", "VGG16","ResNet152"], help='DNN arch to be used')
parser.add_argument('--load', help='path to trained caffemodel and weights')
parser.add_argument('--defense', default='True', choices=['False', 'True'], help='Switch between baseline (no defense) and our proposed defense (FRU)')
args = parser.parse_args()

if not args.load:
    print('Error: Model weights not provided')
else:
    if args.defense=='True':
        model_proto = 'Prototxt/'+args.dnn+'/deploy_'+args.dnn.lower()+'_FRU.prototxt'
    else:
        model_proto = 'Prototxt/'+args.dnn+'/deploy_'+args.dnn.lower()+'.prototxt'

    net = caffe.Classifier(model_proto, args.load, caffe.TEST)
    print('\n Accuracy on original images')
    clean_acc = get_test_acc(net,num_test=20,use_noise=False)
    print('\n Accuracy on adversarial images')
    adv_acc = get_test_acc(net, num_test=20, use_noise=True)
    print('\n Restoration rate: '+str((float(clean_acc[0,0])+float(adv_acc[0,0]))
                                      /(2*float(base_acc[args.dnn]))))