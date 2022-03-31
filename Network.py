from Layer import *

class Network(BasicLayer):
    def __init__(self,para={}):

        self.model = []
        #para={
        #   "Conv_layer":{XX..}
        # }
        for key in para.keys():
            if "Conv" in key:
                self.model.append(Conv_layer(para[key]))
            elif "Fully" in key:
                self.model.append(Fully_connect_layer(para[key]))
            elif "Pool" in key:
                self.model.append(Pooling(para[key]))
            elif "Relu" in key:
                self.model.append(Relu())
            elif "Sigmoid" in key:
                self.model.append(Sigmoid())

        self.loss_fun = Cross_Entropy_Loss()

    def forward(self,X):
        feature = X
        for layer in self.model:
            feature = layer.forward(feature)
        return feature

    def get_loss(self,X,Y):
        return self.loss_fun.forward(X,Y)

    def backward(self,X,Y):
        loss = self.loss_fun.forward(X, Y)
        gradient = self.loss_fun.backward()

        for layer in self.model[::-1]:
            gradient = layer.backward(gradient)

    def update(self,learning_rate = 0.1):
        for layer in self.model[::-1]:
            layer.update(learning_rate)
