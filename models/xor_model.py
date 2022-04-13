
import torch
import tensorflow as tf
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import onnx
from onnx_tf.backend import prepare


# Init model weights
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)

# Training data
X = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.Tensor([[0],[1],[1],[0]])

# Model
class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x

model = XOR()
weights_init(model)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

# Training
epochs = 2001
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point], requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)
        
        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()
        
    if i % 500 == 0:
        print ("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))

# Exporting to ONNX
input_var = Variable(torch.FloatTensor(X))
torch.onnx.export(
    model,
    input_var,
    'xor.onnx',
    input_names = ['input'],
    output_names = ['output'],
)

# Exporting to TensorFlow
onnx_model = onnx.load("xor.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("xor.pb")


# Exporting to TF Lite
converter = tf.lite.TFLiteConverter.from_saved_model('xor.pb')
tflite_model = converter.convert()
with open('xor.tflite', 'wb') as f:
    f.write(tflite_model)
