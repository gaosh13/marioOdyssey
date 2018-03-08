# import the usual suspects
import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
import time
import copy
import os
from mxnet.gluon.model_zoo import vision as models

ctx = mx.gpu()
batch_size = 16

# for plotting purposes
import cv2

root = "operation"

# 
# l = []
# for root, dirs, files in os.walk(root):
#     path = root.split(os.sep)
#     print((len(path) - 1) * '---', os.path.basename(root))
#     for file in files:
#         if file.endswith(".jpg"):
#             img = cv2.imread(os.path.join(*path, file))
#             l.append(img)
# allimg = np.array(l[:1000], 'float32')
# allimg = allimg / 255
# print(allimg.shape)
# means = allimg.mean(axis = (0,1,2))
# stds = allimg.std(axis = (0,1,2))
# print(means, stds)

means = [0.41863686, 0.64764392, 0.64764392]
stds = [0.14449726, 0.14238554, 0.2524929]

# write a little function to get our image in mxnet format
def preprocess(img):
    # convert the image to ndarray
    data = nd.array(img)
    # crop down to 224x224 pixels
    data, _ = mx.image.center_crop(data, (224, 224))
    # add in the batch dimension
    data = data.expand_dims(0)
    # transpose the channels to get format right
    data = data.transpose((0, 3, 1, 2))
    # normalize for pretrained net
    data = mx.image.color_normalize(data/255,
                    mean=mx.nd.array(means).reshape((1,3,1,1)),
                    std=mx.nd.array(stds).reshape((1,3,1,1)))
    data = data.as_in_context(ctx)
    return data

def transform(data, label):
    data = data.astype('float32')
    data, _ = mx.image.center_crop(data, (224, 224))
    data = data.transpose((2,0,1))
    data = mx.image.color_normalize(data/255,
                    mean=mx.nd.array(means).reshape((3,1,1)),
                    std=mx.nd.array(stds).reshape((3,1,1)))
    return data, label



# import json
# class_dict = json.load(open("labels.json"))

# # get the output guesses of the network
# def predict(net, img):
#     img = preprocess(img).as_in_context(ctx)
#     output = net(img)
#     guesses = nd.topk(output, k=5)[0]
#     confidence = nd.softmax(output)[0][guesses]
#     return [class_dict[str(int(x))] for x in list(guesses.asnumpy())], confidence

# # get the raw features of the network
# def get_features(net, img):
#     img = preprocess(img).as_in_context(ctx)
#     output = net.features(img)
#     return output

dataset = gluon.data.vision.ImageFolderDataset(root,  transform=transform)
np.random.shuffle(dataset.items)
total_size = len(dataset.items)
train_size = int(total_size * 0.8)
train_set = copy.deepcopy(dataset)
test_set = copy.deepcopy(dataset)
train_set.items = train_set.items[:train_size]
test_set.items = test_set.items[train_size:]
print("Train size: {}, test size: {}".format(len(train_set.items), len(test_set.items)))

train_data = gluon.data.DataLoader(train_set, batch_size, num_workers = 0)
test_data = gluon.data.DataLoader(test_set, batch_size, num_workers = 0)
print("End loading")

# lets use a pretrained mobilenet
# this a model known for being decently good accuracy at a low computational cost
mobilenet = models.mobilenet0_5(pretrained=True, prefix="copycat_", ctx=ctx)

########Sanity check##########
# sample_data = gluon.data.DataLoader(train_set, 1, shuffle=True)
# for i, (data, label) in enumerate(sample_data):
#     print(data.shape)
#     print (label.asnumpy()[0])
#     if i == 5:
#         break
# img = cv2.imread(os.path.join(root, "0", "frame_0_1.jpg"))
# print(img.shape)

# predict(mobilenet, img)
##############################

# gluon.Block is the basic building block of models.
# You can define networks by composing and inheriting Block:
class CopyNet(gluon.Block):
    def __init__(self, available_actions_count):
        super(CopyNet, self).__init__()
        with self.name_scope():
            self.features = mobilenet.features
            self.output = OutputLayers(available_actions_count)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x

class OutputLayers(gluon.Block):
    def __init__(self, available_actions_count):
        super(OutputLayers, self).__init__()
        with self.name_scope():
            self.denses = gluon.nn.Sequential()
            with self.denses.name_scope():
                self.denses.add(gluon.nn.Dense(128, activation='relu'))
                self.denses.add(gluon.nn.Dropout(.5))
                self.denses.add(gluon.nn.Dense(64, activation='relu'))
                self.denses.add(gluon.nn.Dropout(.5))
            self.action_pred = gluon.nn.Dense(available_actions_count)
            self.value_pred = gluon.nn.Dense(1)

    def forward(self, x):
        x = self.denses(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return probs

gamma = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.001
momentum_param = 0.05

num_actions = 9
copynet = CopyNet(num_actions)
copynet.collect_params().load("data/copycat-18.params", ctx=ctx)
# copynet.output.collect_params().initialize(mx.init.Xavier(),ctx=ctx)

########Sanity check##########
print("******** Sanity check ********")
for i, (data, label) in enumerate(train_data):
    data = data.as_in_context(ctx)
    print("data.dtype = ", data.dtype)
    print("shapes", copynet(data).shape, label.shape)
    if i == 0:
        break
print("******** Sanity check ********")
##############################

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(copynet.output.collect_params(), 'adam', 
                          {'learning_rate': learning_rate,  
                           "beta1": beta1,  
                           "beta2": beta2, 
                           "epsilon": epsilon}
                         )

# Dealing with inbalanced loss

labels = [label for data, label in dataset.items]
_, label_ratios = np.unique(labels, return_counts=True)
label_ratios = nd.array(label_ratios).as_in_context(ctx)
label_ratios = label_ratios / len(train_set.items)
print(label_ratios)

def unbalanced_loss(loss_func, z, y):
    # get ratios for each label in y
    y_index = y.reshape([-1])
    # get the ratio for each label in y
    ratios = label_ratios[y_index]
    # discourage common labels
    ratios = 1 - ratios
    # compute normal loss
    regular_loss = loss_func(z, y)
    # scale the normal loss
    scaled_loss = regular_loss * ratios
    return scaled_loss

# Evaluation function

def evaluate(data_iterator, net):
    acc = mx.metric.Accuracy()
    loss_avg = 0.
    # iterate through all the data
    for i, (data, label) in enumerate(data_iterator):
        # move the data and label to the proper device
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # run the data through the network
        with autograd.predict_mode():
            output = net(data)
            loss = unbalanced_loss(loss_func, output, label)
        # check what our guess is
        predictions = nd.argmax(output, axis=1)
        # compute accuracy and update our running tally
        acc.update(preds=predictions, labels=label)
        loss_avg = loss_avg*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
        if i % 10 == 0:
            nd.waitall()
        if i % 100 == 0:
            print("evaluation", i)
    # return the accuracy
    return acc.get()[1], loss_avg


epochs = 10
best_acc = .0

def train(net, epochs):
    for epoch in range(epochs):
        for i, (d, l) in enumerate(train_data):
            data = d.as_in_context(ctx)
            label = l.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = unbalanced_loss(loss_func, output, label)
            loss.backward()
            optimizer.step(data.shape[0], ignore_stale_grad=True)
            if i % 10 == 0:
                nd.waitall()
            if i % 100 == 0:
                print("batch", i)

        print("Evaluating")
        test_acc, test_loss = evaluate(test_data, net)
        print("Epoch %d. Test:  Loss %s, Acc %s" % (epoch, test_loss, test_acc))
        # train_acc, train_loss = evaluate(train_data, net)
        # print("Epoch %d. Train: Loss %s, Acc %s" % (epoch, train_loss, train_acc))
        
        # global best_acc
        # if test_acc > best_acc:
            # best_acc = test_acc
            # print('Best validation f1 found. Checkpointing...')
        net.collect_params().save('data/copycat-%d.params'%(epoch))

print("******** Training ********")
# copynet.collect_params().save('data/copycat-%d.params'%(0))
train(copynet, 50)
# print(evaluate(test_data, copynet))
# img = cv2.imread(os.path.join(root, "1", "frame_0_196.jpg"))
# print(img.shape)
# print(nd.argmax(copynet(preprocess(img)), axis = 1))

# img = cv2.imread(os.path.join(root, "2", "frame_0_537.jpg"))
# print(img.shape)
# print(nd.argmax(copynet(preprocess(img)), axis = 1))

# img = cv2.imread(os.path.join(root, "5", "frame_0_618.jpg"))
# print(img.shape)
# print(nd.argmax(copynet(preprocess(img)), axis = 1))