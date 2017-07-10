import mxnet as mx
import numpy as np
import minpy.numpy as mp
# First, the symbol needs to be defined
data = mx.sym.Variable("data") # input features, mxnet commonly calls this 'data'
label = mx.sym.Variable("softmax_label")

# One can either manually specify all the inputs to ops (data, weight and bias)
w1 = mx.sym.Variable("weight1")
b1 = mx.sym.Variable("bias1")
l1 = mx.sym.FullyConnected(data=data, num_hidden=16, name="layer1", weight=w1, bias=b1)
a1 = mx.sym.Activation(data=l1, act_type="relu", name="act1")

# Or let MXNet automatically create the needed arguments to ops
l2 = mx.sym.FullyConnected(data=a1, num_hidden=10, name="layer2")

# Create some loss symbol
cost_classification = mx.sym.SoftmaxOutput(data=l2, label=label)

# Bind an executor of a given batch size to do forward pass and get gradients
batch_size = 128
input_shapes = {"data": (batch_size, 28*28), "softmax_label": (batch_size, )}
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
    r[:] = np.random.randn(*r.shape)*0.02

# Using skdata to get mnist data. This is for portability. Can sub in any data loading you like.
from skdata.mnist.view import OfficialVectorClassification

data = OfficialVectorClassification()
trIdx = data.sel_idxs[:]
teIdx = data.val_idxs[:]
for epoch in range(20):
    np.random.shuffle(trIdx)

    for x in range(0, len(trIdx), batch_size):
        # extract a batch from mnist
        batchX = data.all_vectors[trIdx[x:x+batch_size]]
        batchY = data.all_labels[trIdx[x:x+batch_size]]

        # our executor was bound to 128 size. Throw out non matching batches.
        if batchX.shape[0] != batch_size:
            continue
        # Store batch in executor 'data'
        executor.arg_dict['data'][:] = batchX / 255.
        # Store label's in 'softmax_label'
        executor.arg_dict['softmax_label'][:] = batchY
        executor.forward()
        executor.backward()

        momentum=1e-3
        # do weight updates in imperative
        for pname, W, G in zip(cost_classification.list_arguments(), executor.arg_arrays, executor.grad_arrays):
            # Don't update inputs
            # MXNet makes no distinction between weights and data.
            if pname in ['data', 'softmax_label']:
                continue
            # what ever fancy update to modify the parameters
            auto_momentum = mx.nd.minimum(momentum, mx.nd.power(mx.nd.sum(G), 2.0))
            auto_k = mx.nd.minimum(0, mx.nd.minimum(1, 1 - auto_momentum))
            vw = W * auto_momentum - .001 * G
            vn = W * momentum - .001 * G
            # print(auto_momentum.asnumpy(), auto_k.asnumpy())
            W[:] = W + auto_k * vn + (1-auto_k) * vw

    # Evaluation at each epoch
    num_correct = 0
    num_total = 0
    for x in range(0, len(teIdx), batch_size):
        batchX = data.all_vectors[teIdx[x:x+batch_size]]
        batchY = data.all_labels[teIdx[x:x+batch_size]]
        if batchX.shape[0] != batch_size:
            continue
        # use the test executor as we don't care about gradients
        executor_test.arg_dict['data'][:] = batchX / 255.
        executor_test.forward()
        num_correct += sum(batchY == np.argmax(executor_test.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)
    print (str(num_correct / float(num_total)) + ",")
