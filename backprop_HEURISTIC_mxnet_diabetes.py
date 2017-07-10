import mxnet as mx
import numpy as np
# First, the symbol needs to be defined
from sklearn.metrics.regression import mean_squared_error

data = mx.sym.Variable("data") # input features, mxnet commonly calls this 'data'
label = mx.sym.Variable("lro")

# One can either manually specify all the inputs to ops (data, weight and bias)
w1 = mx.sym.Variable("weight1")
b1 = mx.sym.Variable("bias1")
l1 = mx.sym.FullyConnected(data=data, num_hidden=128, name="layer1", weight=w1, bias=b1)
a1 = mx.sym.Activation(data=l1, act_type="relu", name="act1")
w2 = mx.sym.Variable("weight2")
b2 = mx.sym.Variable("bias2")
l2 = mx.sym.FullyConnected(data=a1, num_hidden=128, name="layer2", weight=w2, bias=b2)
a2 = mx.sym.Activation(data=l2, act_type="relu", name="act2")

w3 = mx.sym.Variable("weight3")
b3 = mx.sym.Variable("bias3")
l3 = mx.sym.FullyConnected(data=a2, num_hidden=32, name="layer3", weight=w3, bias=b3)
a3 = mx.sym.Activation(data=l3, act_type="relu", name="act3")

# Or let MXNet automatically create the needed arguments to ops
l4 = mx.sym.FullyConnected(data=a3, num_hidden=1, name="layer4")

# Create some loss symbol
cost_classification = mx.sym.LinearRegressionOutput(data=l4, label=label, name='lro')

# Bind an executor of a given batch size to do forward pass and get gradients
batch_size = 3
input_shapes = {"data": (batch_size, 10), "lro": (batch_size, )}
executor = cost_classification.simple_bind(ctx=mx.cpu(0),
                                           grad_req='write',
                                           **input_shapes)
# The above executor computes gradients. When evaluating test data we don't need this.
# We want this executor to share weights with the above one, so we will use bind
# (instead of simple_bind) and use the other executor's arguments.
executor_test = cost_classification.bind(ctx=mx.cpu(0),
                                         grad_req='null',
                                         args=executor.arg_arrays)

# initialize the weights
for r in executor.arg_arrays:
    r[:] = np.random.randn(*r.shape)*0.2

from sklearn.datasets import load_diabetes

data = load_diabetes()
trIdx = data['data'][:360]
teIdx = data['data'][360:]
teIdy = data['target'][360:]
for epoch in range(35):
    np.random.shuffle(trIdx)

    for x in range(0, len(trIdx), batch_size):
        # extract a batch from mnist
        batchX = trIdx[x:x+batch_size]
        batchY = data['target'][x:x+batch_size]

        # our executor was bound to 128 size. Throw out non matching batches.
        if batchX.shape[0] != batch_size:
            continue
        # Store batch in executor 'data'
        executor.arg_dict['data'][:] = batchX
        # Store label's in 'softmax_label'
        executor.arg_dict['lro'][:] = batchY
        executor.forward()
        executor.backward()

        momentum = 1e-4
        # do weight updates in imperative
        for pname, W, G in zip(cost_classification.list_arguments(), executor.arg_arrays, executor.grad_arrays):
            # Don't update inputs
            # MXNet makes no distinction between weights and data.
            if pname in ['data', 'lro']:
                continue
            # what ever fancy update to modify the parameters
            auto_momentum = mx.nd.minimum(momentum, mx.nd.power(mx.nd.sum(G), 2.0))
            auto_k = mx.nd.minimum(0, mx.nd.minimum(1, 1 - auto_momentum))
            vw = W * auto_momentum - .001 * G
            vn = W * momentum - .001 * G
            # print(auto_momentum.asnumpy(), auto_k.asnumpy())
            W[:] = W + auto_k * vn + (1 - auto_k) * vw

    # Evaluation at each epoch
    output = []
    for x in range(0, len(teIdx), batch_size):
        batchX = teIdx[x:x+batch_size]
        batchY = teIdy[x:x+batch_size]
        if batchX.shape[0] != batch_size:
            continue
        # use the test executor as we don't care about gradients
        executor_test.arg_dict['data'][:] = batchX
        executor_test.forward()
        output.extend(executor_test.outputs[0].asnumpy().tolist())
    # print (str(num_correct) + ",")
    print(mean_squared_error(teIdy[:len(output)], output))
