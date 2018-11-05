from nn import neural_net
import numpy as np

obj = neural_net([5,3,2])
#print("weight 0, act 0, bias 0:")
#print(obj.w[0].shape)

'''print(obj.acts[0].shape)
print(obj.bias[0].shape)
print("biases")
print(obj.bias)
print(len(self.grad_w),len(self.grad_b))
print(len(self.w),len(self.bias))
print(self.grad_w[-1].shape,self.grad_b[-1].shape)
'''
tr = obj.sgd(np.random.randn(10,5),np.random.randn(10,2),3,5,1.5)

#X: (batch_size, input_layer_count)
#Output dim = acts dim: (batch_size, output_layer_count)
#print(obj.acts[-1].shape)
#print(obj.z[-1].shape)

'''
delta_prv = delta_curr
for layer_index in range(self.layer_count - 2,0,-1)
    delta_curr = np.multiply(np.dot(delta_prv,self.w[layer_index]),\
    sigmoid_dx(self.z[layer_index]))
    cum_del_w = np.zeros(self.w[layer_index - 1].shape)
    for row in range(del_curr.shape[0]):
        cum_del_w += np.dot(self.acts[layer_index - 1][row].reshape(-1,1),del_curr[0].reshape(1,-1))
'''
