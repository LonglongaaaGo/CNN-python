import numpy as np
import sys
class BasicLayer():
    def __init__(self,*arg,**kwargs):
        self.others_keys = ["size","stride","pad","mode"]

    def forward(self,*arg,**kwargs):
        pass
    def backward(self,*arg,**kwargs):
        pass
    def update(self,*arg,**kwargs):
        pass

    def init_module(self,key,value):
        if (key == "Kernel"):
            assert len(value) ==4
            #value = (4, 4, 3, 8)  (k_h,kw,channel,out_channel)
            return np.random.randn(value[0],value[1],value[2],value[3]) * 0.01
        elif (key == "linear"):
            assert len(value) ==2
            #value = (3, 8)  (k_h,kw,channel,out_channel)
            return np.random.randn(value[0],value[1]) * 0.01

        elif (key == "bias"):
            #value = (1, 1, 1, 8)  (1,1,1,out_channel)
            return np.zeros(value)
        elif key in self.others_keys:
            pass
            # print(key)
        else:
            print("Please use the correct keys")
            sys.exit(1)

class Sigmoid(BasicLayer):
    def __init__(self):
        BasicLayer.__init__(self)

        self.pre_tensor = None

    def forward(self,Z):
        """
        Implements the sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape

        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """

        A = 1 / (1 + np.exp(-Z))
        self.pre_tensor = Z
        return A

    def backward(self,dA):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = self.pre_tensor
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

class Relu(BasicLayer):
    def __init__(self):
        BasicLayer.__init__(self)

        self.pre_tensor = None

    def forward(self,Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """

        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)

        self.pre_tensor = Z
        return A
    def backward(self,dA):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = self.pre_tensor
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ




class Pooling(BasicLayer):
    def __init__(self,para={}):
        BasicLayer.__init__(self)

        self.hparameters = para
        #{'mode':"max","size":2,"stride":2}
        self.weight = {}
        for key in para.keys():
            self.weight[key] = self.init_module(key,para[key])
        self.pre_tensor = None


    def forward(self,A_prev):
        # GRADED FUNCTION: pool_forward
        """
        Implements the forward pass of the pooling layer
        单线程的下采样操作
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        #A_prev 上一层所输出的特征图(m, n_H_prev, n_W_prev, n_C_prev)  (数量，高，宽，通道数)

        hparameters -- python dictionary containing "f" and "stride"
        hparameters = "f" 下采样的采样核大小  即 f*f ，"stride"  为相应步长，
        输出的大小其实和卷积操作计算时一样的，只是没有填充（pad）操作。
        mode -- thepad pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        输出 下采样后的特征图  (m, n_H, n_W, n_C)(数量，采样后的高，采样后的宽，通道数)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters

        cache = (A_prev, hparameters)   为了反向梯度求导做准备
        """

        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve hyperparameters from "hparameters"
        f = self.hparameters["size"]
        stride = self.hparameters["stride"]
        mode = self.hparameters["mode"]
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        ### START CODE HERE ###
        for i in range(m):  # loop over the training examples
            for h in range(n_H):  # loop on the vertical axis of the output volume
                for w in range(n_W):  # loop on the horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)

        ### END CODE HERE ###

        # Store the input and hparameters in "cache" for pool_backward()
        # cache = (A_prev, hparameters)
        self.pre_tensor = A_prev

        # Making sure your output shape is correct
        assert (A.shape == (m, n_H, n_W, n_C))
        return A

    def backward(self,dA):
        """
        Implements the backward pass of the pooling layer

        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        下采样时的反向传播，原理同卷积一样，“怎么回去怎么回来”
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """

        ### START CODE HERE ###

        # Retrieve information from cache (≈1 line)
        A_prev = self.pre_tensor

        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = self.hparameters['stride']
        f = self.hparameters['size']
        mode =self.hparameters["mode"]

        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(np.shape(A_prev))

        for i in range(m):  # loop over the training examples

            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i, :, :, :]

            for h in range(n_H):  # loop on the vertical axis
                for w in range(n_W):  # loop on the horizontal axis
                    for c in range(n_C):  # loop over the channels (depth)

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Compute the backward propagation in both modes.
                        if mode == "max":

                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = self.create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask,
                                                                                                       dA[i, h, w, c])
                        elif mode == "average":

                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, shape)

        ### END CODE ###

        # Making sure your output shape is correct
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev

    def create_mask_from_window(self,x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        在下采样反向传播时，要用到，这是因为要根据下采样之前的样子，来进行还原，
        下采样之前哪个哪个值最大， 则，给它位置记录下来，在还原的时候，就直接根据采样之前的带下还原维度，然后根据本次计算的最大值的位置，
        把最大值放上去，其他的都填充0
        Arguments:
        x -- Array of shape (f, f)

        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """

        ### START CODE HERE ### (≈1 line)
        mask = (x == np.max(x))
        ### END CODE HERE ###
        return mask

    def distribute_value(self,dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        用的时平局值下采样时的反向传播
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """

        ### START CODE HERE ###
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape

        # Compute the value to distribute on the matrix (≈1 line)
        average = float(dz) / (n_H * n_W)

        # Create a matrix where every entry is the "average" value (≈1 line)
        a = average * np.ones(shape)
        ### END CODE HERE ###

        return a


class Zero_Padding(BasicLayer):
    def __init__(self):
        BasicLayer.__init__(self)

    # GRADED FUNCTION: zero_pad
    # 0填充操作，
    # 如果为小数，如1.5，则左边填充1，右边填充2，上边填充1，下边填充2
    # 如果为整数，则四边都填充相同的大小
    def forward(self,X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
        as illustrated in Figure 1.

        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        (m, n_H, n_W, n_C)表示特征图的 ： m = 数量 ， n_H = 高度， n_W = 宽度 ， n_C 维度
        pad -- float, amount of padding around each image on vertical and horizontal dimensions
        #0填充操作，
        #如果为小数，如1.5，则左边填充1，右边填充2，上边填充1，下边填充2
        #如果为整数，则四边都填充相同的大小
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        ### START CODE HERE ### (≈ 1 line)
        if (pad == 0):
            return X

        if ((pad * 2) % 2 != 0):
            pad_left = int(pad)
            pad_right = pad_left + 1
        else:
            pad_left = int(pad)
            pad_right = int(pad)
        X_pad = np.pad(X, ((0, 0), (pad_left, pad_right), (pad_left, pad_right), (0, 0)), 'constant')
        ### END CODE HERE ###

        return X_pad

    def backward(self,A_prev_pad, pad):
        # pad填充操作的反向传播，其实就是恢复原来大小
        if (pad == 0):
            return A_prev_pad

        if (int(2 * pad) % 2 != 0):
            pad_left = int(pad)
            pad_right = pad_left + 1
        else:
            pad_left = int(pad)
            pad_right = pad_left

        if (A_prev_pad.ndim == 3):
            A_prev = A_prev_pad[pad_left:-pad_right, pad_left:-pad_right, :]
        elif (A_prev_pad.ndim == 4):
            A_prev = A_prev_pad[:, pad_left:-pad_right, pad_left:-pad_right, :]

        return A_prev


# GRADED FUNCTION: image2vector
def tensor2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    #在卷积核的最后一层到  全连接层的过度操作
    #实质上时flatten  ，就是把所有特征图拉成一条1向量
    #保留形状维度参数大小，以便恢复
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[1] * image.shape[2] * image.shape[3], image.shape[0]))
    ### END CODE HERE ###
    shape = (image.shape[0], image.shape[1], image.shape[2], image.shape[3])

    return v, shape


# GRADED FUNCTION: image2vector
def vector2tensor(vector, img_shape):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    #恢复操作，全连接梯度计算完毕之后，开始反向计算前一层的卷积核的梯度，那么恢复原状以便计算

    (a, b, c, d) = img_shape
    ### START CODE HERE ### (≈ 1 line of code)
    img = vector.reshape((a, b, c, d))
    ### END CODE HERE ###

    return img


class Fully_connect_layer(BasicLayer):
    def __init__(self, para={}):
        BasicLayer.__init__(self)

        self.hparameters = para

        self.weight = {}
        for key in para.keys():
            self.weight[key] = self.init_module(key,para[key])
        self.pre_tensor = None

        self.weight_gradient = {}
        self.pre_tensor_shape = None

    def forward(self,pre_A):
        """
        Implement the linear part of a layer's forward propagation.
        #全连接操作
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        W = self.weight["linear"]
        b = self.weight["bias"]

        if pre_A.ndim ==4:
            self.pre_tensor_shape =pre_A.shape
            pre_A,_ = tensor2vector(pre_A)


        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W, pre_A) + b
        ### END CODE HERE ###
        assert (Z.shape == (W.shape[0], pre_A.shape[1]))

        self.pre_tensor = pre_A
        return Z

    def backward(self,dZ):

        pre_A = self.pre_tensor
        W = self.weight["linear"]
        b = self.weight["bias"]

        dA_prev = np.dot(W.T,dZ)
        dW = dZ.dot(pre_A.T)
        db = dZ

        self.weight_gradient['linear'] = dW
        self.weight_gradient['bias'] = db

        assert (dA_prev.shape == pre_A.shape) and dW.shape == W.shape

        if self.pre_tensor_shape is not None and len(self.pre_tensor_shape) == 4:
            dA_prev = vector2tensor(dA_prev,self.pre_tensor_shape)

        return dA_prev


class Drop_out_layer(BasicLayer):
    def __init__(self, para={}):
        BasicLayer.__init__(self)

        self.hparameters = para
        self.mask = None

    def forward(self,pre_A):
        pKeep = self.hparameters["rate"]
        self.mask = np.random.rand(pre_A.shape) <pKeep

        Z = np.multiply(pre_A, self.mask)
        Z /= pKeep

        return Z

    def backward(self,dZ):

        pKeep = self.hparameters["rate"]

        dA_prev = dZ * self.mask/pKeep

        return dA_prev


class Cross_Entropy_Loss(BasicLayer):
    #
    def __init__(self):
        BasicLayer.__init__(self)

        self.softmax = softmax_layer()

        self.pre_tensor = None
        self.label = None

    def forward(self, X,Y):

        score = self.softmax.forward(X)
        Y = np.squeeze(Y)
        num_train = X.shape[1]
        #indicate
        # correct_class_score = score[Y, range(num_train)]
        correct_class_score = score[Y,range(num_train)]

        loss = -np.sum(np.log(correct_class_score))/float(num_train)

        self.pre_tensor = score
        self.label = Y
        return loss

    def backward(self):

        dZ = self.pre_tensor
        dZ[self.label,range(len(self.label))] -= 1

        # for i in range(len(self.label)):
        #     dZ[self.label[i], i] -= 1

        dZ = dZ/float(len(self.label))
        return dZ


class softmax_layer(BasicLayer):
    def __init__(self):
        BasicLayer.__init__(self)

    def forward(self, X):
        # print '--=------softmax_loss_vectorized----------------------------------------------'
        # softmax_forward前向传播
        """
        Softmax loss function, naive implementation (with loops)

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.

        that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        # Initialize the loss and gradient to zero.
        exp_scores = np.exp(X)
        sum_exp_scores = np.sum(exp_scores, axis=0)
        exp_scores = exp_scores / sum_exp_scores[np.newaxis,: ]

        return exp_scores
    def backword(self):
        pass


class Conv_layer(BasicLayer):
    def __init__(self,para={}):
        BasicLayer.__init__(self)
        self.hparameters = para

        self.weight = {}
        for key in para.keys():
            if key in self.others_keys: continue
            self.weight[key] = self.init_module(key,para[key])
        self.pre_tensor = None

        self.weight_gradient = {}

        self.zero_pad = Zero_Padding()

    def forward(self,A_prev):
        ### START CODE HERE ###
        # Retrieve dimensions from A_prev's shape (≈1 line)
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve information from "hparameters" (≈2 lines)
        stride = self.hparameters["stride"]
        pad = self.hparameters["pad"]
        W = self.weight["Kernel"]
        b = self.weight["bias"]

        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape

        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
        n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad.forward(A_prev, pad)

        for i in range(m):  # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i, :, :, :]  # Select ith training example's padded activation

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = stride * h
                        vert_end = vert_start + f
                        horiz_start = stride * w
                        horiz_end = horiz_start + f

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (Z.shape == (m, n_H, n_W, n_C))

        # Save information in "cache" for the backprop
        # cache = (A_prev, W, b, hparameters)

        self.pre_tensor = A_prev
        return Z

    def conv_single_step(self,a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
        of the previous layer.
        #一次卷积操作，即特征图和卷积核相同大小维度的一次卷积操作，返回1个值、
        #一次卷积操作为：卷积核移动到某个位置，然后取出和特征图对应维度大小的特征图的局部块，然后进行对应相乘求和，
        #返回1个标量值
        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        #a_slice_prev 上1层的特征图的 和卷积核的某个位置的对应局部块
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        # W 卷积核的参数(f, f, n_C_prev)  (卷积核高，卷积核宽，卷积核维度_与上一层特征图通道数对应)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        # b 卷积核的偏置值
        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        a scalar value  标量值
        """
        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between a_slice and W. Do not add the bias yet.
        s = a_slice_prev * W
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z + b
        ### END CODE HERE ###

        return Z

    def backward(self,dZ):
        """
        Implement the backward propagation for a convolution function
        卷积操作的反向传播
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        既然是反向传播，那么计算方式就和卷积的前向传播相反，即"怎么过去的，就怎么回来"，所以输入自然是卷积后的输出层的反向梯度作为输入
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        cache --之前保留下来的卷积核，在这里进行取出计算
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        输出就是对应的 前层的输入特征图，反向传播就像  栈操一样，有点像 前像传播压进去，反向传播弹出计算
        原先在前向传播时的输入，变成了反向传播的输出
        原先在前向传播时的输除，变成了反向传播的输入
        因为计算梯度时，要从后往前，梯度计算回去，也就是链式求导
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        这层卷积核的梯度（f, f, n_C_prev, n_C）对应维度的含义跟前向传播一样：
        有 _prev  后缀的 为前一层的意思
        有  _C  后缀的 为 通道数的意思
        f  代表核的大小  ，可能时卷积核，也可能时 下采样核，因为计算原理时相同的
        之后不再赘述
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """

        ### START CODE HERE ###
        # Retrieve information from "cache"
        # Retrieve information from "hparameters"
        A_prev = self.pre_tensor
        stride = self.hparameters['stride']
        pad = self.hparameters['pad']
        W = self.weight['Kernel']
        b = self.weight['bias']

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        # 反向传播的时候，必须要把维度计算回去
        A_prev_pad = self.zero_pad.forward(A_prev, pad)
        dA_prev_pad = self.zero_pad.forward(dA_prev, pad)

        for i in range(m):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            # dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            dA_prev[i, :, :, :] = self.zero_pad.backward(da_prev_pad, pad)
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        self.weight_gradient['Kernel'] = dW
        self.weight_gradient['bias'] = db

        return dA_prev

    def update(self, learning_rate):
        for key in self.weight.keys():
            self.weight[key] = self.weight[key] - self.weight_gradient[key]*learning_rate

