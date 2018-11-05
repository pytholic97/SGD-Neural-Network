"""
Neural network
"""
import numpy as np
import time
from datetime import timedelta

class neural_net():

    def __init__(self,shape,output_layer='sigmoid'):
        self.shape = shape
        self.layer_count = len(shape)
        self.hidden_count = self.layer_count - 2
        self.output_layer = output_layer
        np.random.seed(1)
        self.w = [np.random.randn(self.shape[i],self.shape[i+1]) for i in range(self.layer_count - 1)]
        #for i in range(1,self.layer_count):
        #self.w.append(np.random.randn(self.shape[i],self.shape[i-1]))
        self.bias = []
        self.bias.append(np.zeros((self.shape[0],1)))
        #self.bias = [for i in range(self.layer_count - 1)]
        for i in range(1,self.layer_count):
            self.bias.append(np.random.randn(self.shape[i],1))
        #self.acts = [np.zeros(self.shape[i]) for i in range(self.layer_count)]


    def feed_forward(self,X):
        self.acts = []
        self.z = []

        self.acts.append(X)
        self.z.append(X)
        for layer in range(self.layer_count - 2):
            new_z = self.acts[layer].dot(self.w[layer])
            new_z = new_z.T + np.hstack(tuple([self.bias[layer + 1] for _ in range(X.shape[0])]))
            self.z.append(new_z.T)
            self.acts.append(sigmoid(new_z).T)

        new_z = self.acts[-1].dot(self.w[self.layer_count - 2])
        new_z = new_z.T + np.hstack(tuple([self.bias[self.layer_count - 1] for _ in range(X.shape[0])]))
        self.z.append(new_z.T)

        if self.output_layer == 'softmax':
            self.acts.append(softmax(self.z[-1]))
        elif self.output_layer == 'sigmoid':
            self.acts.append(sigmoid(new_z).T)

        return self.acts[-1]

    def process_batch(self, batch_in, batch_output, b_size, eta, lmbda, cost='mean square'):
        self.grad_w = []
        self.grad_b = []

        self.hyp = self.feed_forward(batch_in) #Forward pass of entire batch -> returns hypothesis
        acc = self.compute_accuracy(self.hyp,batch_output)

        #Code for backpropagation follows:

        #Computing the delta for the last layer for entire batch
        if cost == 'mean square':
            delta_curr = np.multiply((self.hyp - batch_output),sigmoid_dx(self.z[-1]))
        elif cost == 'cross-entropy':
            delta_curr = self.hyp - batch_output
        elif cost == 'log-like':
            delta_curr = self.hyp - batch_output

        cum_del_w = np.zeros((self.shape[-2],self.shape[-1]))

        #Summing gradients of the weights in the input of the last/output layer
        for row in range(delta_curr.shape[0]):
            cum_del_w += np.dot(self.acts[-2][row].reshape(-1,1),delta_curr[row].reshape(1,-1))

        self.grad_w.append(cum_del_w/b_size) # gradient of last pair of weights appended after averaging
        self.grad_b.append(np.sum(delta_curr,axis=0).reshape(self.shape[-1],1)/b_size) #output layer bias gradients
        delta_prv = delta_curr

        for layer_index in range(self.layer_count - 2,0,-1):
            delta_curr = np.multiply(np.dot(delta_prv,self.w[layer_index].T),\
            sigmoid_dx(self.z[layer_index]))
            cum_del_w = np.zeros(self.w[layer_index - 1].shape)
            for row in range(delta_curr.shape[0]):
                cum_del_w += np.dot(self.acts[layer_index - 1][row].reshape(-1,1),delta_curr[row].reshape(1,-1))
            self.grad_w.append(cum_del_w/b_size)
            self.grad_b.append(np.sum(delta_curr,axis=0).reshape(self.shape[layer_index],1)/b_size)
            delta_prv = delta_curr

        self.grad_w.reverse()
        self.grad_b.reverse()

        self.__update_weights(eta, b_size, lmbda)

        return self.compute_cost(self.hyp, batch_output, b_size, lmbda, cost=cost),acc


    def __update_weights(self,eta, b_size, lmbda):

        self.w = [a*(1-(lmbda*eta)/(b_size))-(eta*b) for (a,b) in zip(self.w,self.grad_w)] #regularised version
        self.b = [a-(eta*b) for (a,b) in zip(self.bias[1:],self.grad_b)]


    def compute_accuracy(self,hyp,true_val):
        h = np.copy(hyp)
        h[h >= 0.5] = 1.0
        h[h != 1] = 0.0
        fl = (h.shape[0]*h.shape[1] - (h == true_val).sum())/2
        return (h.shape[0] - fl)*100/h.shape[0]

    def compute_cost(self, hyp, true_value, b_size, lmbda, cost='mean square'):
        if cost  == 'mean square':
            return (np.sum(np.square(hyp - true_value))+lmbda*sum([np.sum(np.square(h)) for h in self.w]))/(2*b_size) # Quadratic cost function
        elif cost == 'cross-entropy':
            return ((-1)*np.sum(np.add(np.multiply(true_value,np.log(hyp)), np.multiply(1-true_value,np.log(1-hyp))))\
            +(lmbda/2)*sum([np.sum(np.square(h)) for h in self.w]))/b_size
        elif cost == 'log-like':
            return ((-1)*np.sum(np.sum(np.multiply(true_value,np.log(hyp)),axis=1).reshape(-1,1)) + \
            (lmbda/2)*sum([np.sum(np.square(h)) for h in self.w]))/b_size

    def sgd(self, train_samples, train_labels, batch_size, epochs, eta, lmbda=0.0, cost='mean square', en_val=False, vldt_labels=None, vldt=None, k_fold=False):


        if cost == 'log-like' and self.output_layer != 'softmax':
            print('Unsupported combination of cost and output layer activation.')
            exit(0)

        ftr = train_samples.shape[0]//batch_size
        samples = [train_samples[i:i+batch_size].reshape(batch_size,-1) for i in range(ftr)]
        labels = [train_labels[i:i+batch_size].reshape(batch_size,-1) for i in range(ftr)]
        if train_samples.shape[0]%batch_size != 0:
            t1 = train_labels[ftr*batch_size:].reshape(train_labels.shape[0] - batch_size*ftr,-1).all()
            labels.append(t1)
            samples.append(train_samples[ftr*batch_size:].reshape(train_samples.shape[0] - batch_size*ftr,-1))

        if en_val:
            v_size = vldt_labels.shape[0]
            v_cost_list = []
            v_acc = []
        train_cost_list = []
        train_acc = []
        b_acc = 0.0

        for epoch in range(1,epochs+1):
            st = time.monotonic()

            for (sample,label) in zip(samples,labels):
                b_cost,b_acc = self.process_batch(sample,label,batch_size,eta,lmbda, cost)

            train_acc.append(b_acc)
            train_cost_list.append(b_cost)
            if en_val:
                val_hyp = self.feed_forward(vldt)
                v_acc.append(self.compute_accuracy(val_hyp,vldt_labels))
                v_cost = self.compute_cost(val_hyp, vldt_labels, v_size, lmbda, cost=cost)
                v_cost_list.append(v_cost)
                print("Epoch {0}/{1}.....Training Cost: {2}, Validation loss:{3}".format(epoch,epochs,b_cost,v_cost))
            else:
                print("Epoch {0}/{1}.....Training Cost: {2}".format(epoch,epochs,b_cost))
            end_t = time.monotonic()
#                print("Epoch {0}/{1}...Execution time: ".format(epoch,epochs),end='')
#                print(timedelta(seconds=end_t - st),end='s')
            print()

        if en_val:
            return train_cost_list,v_cost_list,train_acc,v_acc
        else:
            return train_cost_list,train_acc

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_dx(x):
    return sigmoid(x)*(1 - sigmoid(x))

def softmax(x):
    expon = np.exp(x)
    smax_cum_sum = []
    line_sum = np.sum(expon,axis=1).reshape(-1,1)

    for i,v in enumerate(expon):
        smax_cum_sum.append(v.reshape(-1,1)/line_sum[i][0])
    return np.hstack(tuple(smax_cum_sum)).T
