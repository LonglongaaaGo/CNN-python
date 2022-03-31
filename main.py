from Network import *
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist


def init_data(train_img,train_lab,batch,shuffle = False):

    if shuffle == True:
        shuffle_ = np.random.shuffle(np.arange(len(train_img)))
        TrainX = train_img[shuffle_]
        TrainY = train_lab[shuffle_]
    else:
        TrainX = train_img
        TrainY = train_lab
    len_ = len(train_img)
    num = len_/float(batch)
    num = np.ceil(num)

    return TrainX,TrainY,num

def get_data(train_img,train_lab,batch,idx):
    len_ = len(train_img)
    if batch>len_: return train_img,train_lab,0
    start = min(idx * batch,len_-1)
    end = min(idx * batch+batch,len_)
    return train_img[start:end],train_lab[start:end]



if __name__ == '__main__':
    """
    run here.
    """

    objects = mnist
    (train_img, train_lab), (test_imag, test_lab) = objects.load_data()

    para = {
        #kh,kw,c,cout
        "Conv_layer1":{"stride":1,"pad":1,"Kernel":(3,3,1,8),"bias":(1,1,1,8)},
        "Relu1": {},
        "Pool1": {'mode':"max","size":2,"stride":2},
        "Conv_layer2": {"stride": 1, "pad": 2, "Kernel": (5, 5, 8, 8), "bias": (1, 1, 1, 8)},
        "Drop_out_layer1": {"rate":0.5},
        "Relu2": {},
        "Pool2": {'mode': "max", "size": 2, "stride": 2},
        "Conv_layer3": {"stride": 1, "pad": 2, "Kernel": (5, 5, 8, 8), "bias": (1, 1, 1, 8)},
        "Drop_out_layer2": {"rate": 0.5},
        "Relu3": {},
        # "Pool3": {'mode': "max", "size": 2, "stride": 2},
        # "Conv_layer4": {"stride": 1, "pad": 1, "Kernel": (3, 3, 8, 4), "bias": (1, 1, 1, 4)},
        # "Drop_out_layer3": {"rate": 0.5},
        # "Relu4": {},
        #cout,cin
        "Fullyconn1": {"linear": (10, 7*7*8), "bias": (10, 1)},
        # "Drop_out_layer4": {"rate": 0.5},
        # "Relu5": {},
        # "Fullyconn2": {"linear": (10, 3 * 3 * 4), "bias": (10, 1)},

        # "Fullyconn1": { "linear": (10, 9), "bias": (10,1)},
        # "Fullyconn1": {"linear": (10, 7*7*4), "bias": (10, 1)},

    }
    # X_ = train_img[:10]
    # Y_ = train_lab[:10]

    net = Network(para=para)
    # print(Y_)

    epoch = 10
    batch = 8
    learning_rate = 0.1
    for epo in range(epoch):
        train_X,train_Y,num = init_data(train_img,train_lab,batch,shuffle=True)
        for idx in range(int(num)):
            X_, Y_ = get_data(train_img, train_lab, batch,idx)
            X_ = np.expand_dims(X_, axis=-1)
            Y_ = np.expand_dims(Y_, axis=-1)

            out = net.forward(X_)
            # print(out)
            aa = np.argmax(out, axis=0)
            print(aa)
            print(Y_)

            # numpy.max(A, axis=0)
            loss = net.get_loss(out, Y_)
            print(loss)
            net.backward(out, Y_)
            net.update(learning_rate=learning_rate)

        test_X,test_Y,num = init_data(test_imag,test_lab,batch,shuffle=False)

        count = 0
        for idx in range(int(num)):
            X_, Y_ = get_data(test_X, test_Y, batch,idx)
            X_ = np.expand_dims(X_, axis=-1)
            Y_ = np.expand_dims(Y_, axis=-1)
            out = net.forward(X_)
            aa = np.argmax(out, axis=0)
            correct = aa == Y_.squeeze()
            correct = np.sum(correct)
            count+=correct
            print("temp_acc:",correct/len(X_))

        print("accuarcy:",count/len(test_imag))











