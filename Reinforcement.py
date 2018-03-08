import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import random
import numpy as np
import math
import time
from time import sleep
import mxnet.ndarray as F
import itertools as it
import sys
from VolleyBall import Volleyball, GameState
from mxnet.gluon.model_zoo import vision

EPISODES = 1000000  # Number of episodes to be played
LEARNING_STEPS = 60000  # Maximum number of learning steps within each episodes
DISPLAY_COUNT = 1  # The number of episodes to play before showing statistics.
LOAD = False
gamma = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.0001
momentum_param = 0.05
learning_rates = [0.0001, 0.01]
frame_repeat = 5
ctx = mx.cpu()
num_action = 9

game = Volleyball()

# class Net(gluon.Block):
#     def __init__(self, available_actions_count):
#         super(Net, self).__init__()
#         with self.name_scope():
#             self.conv1 = gluon.nn.Conv2D(16, kernel_size=5, strides=2)
#             self.bn1 = gluon.nn.BatchNorm()
#             self.conv2 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
#             self.bn2 = gluon.nn.BatchNorm()
#             self.conv3 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
#             self.bn3 = gluon.nn.BatchNorm()
#             #self.lstm = gluon.rnn.LSTMCell(128)
#             self.dense1 = gluon.nn.Dense(128, activation='relu')
#             self.dense2 = gluon.nn.Dense(64, activation='relu')
#             self.action_pred = gluon.nn.Dense(available_actions_count)
#             self.value_pred = gluon.nn.Dense(1)
#         #self.states = self.lstm.begin_state(batch_size=1, ctx=ctx)

#     def forward(self, x):
#         x = nd.relu(self.bn1(self.conv1(x)))
#         x = nd.relu(self.bn2(self.conv2(x)))
#         x = nd.relu(self.bn3(self.conv3(x)))
#         x = nd.flatten(x).expand_dims(0)
#         #x, self.states = self.lstm(x, self.states)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         probs = self.action_pred(x)
#         values = self.value_pred(x)
#         return mx.ndarray.softmax(probs), values

mobilenet = vision.mobilenet0_5(pretrained=True, prefix="copycat_", ctx=ctx)

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
        return mx.ndarray.softmax(probs), values

loss = gluon.loss.L2Loss()
model = CopyNet(num_action)
model.collect_params().load("data/copycat-11.params", ctx=ctx)
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate,  "beta1": beta1,  "beta2": beta2, "epsilon": epsilon})

means = [0.41863686, 0.64764392, 0.64764392]
stds = [0.14449726, 0.14238554, 0.2524929]

def preprocess(img):
    # convert the image to ndarray
    data = nd.array(img.astype("float32"))
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

# def transform(raw_frame):
#     data = nd.array(raw_frame, mx.cpu()).astype('float32')
#     data, _ = mx.image.center_crop(data, (224, 224))
#     data = data.transpose((2,0,1))
#     data = mx.image.color_normalize(data/255,
#                     mean=mx.nd.array(means).reshape((3,1,1)),
#                     std=mx.nd.array(stds).reshape((3,1,1)))
#     return data

def train():
    print("Start the training!")
    episode_rewards = 0
    final_rewards = 0

    running_reward = 10
    train_episodes_finished = 0
    train_scores = [0]
    epoch_count = 0
    if LOAD:
        model.collect_params().load("data/volleyball.params", ctx=ctx)
        with open("data/epoch.txt", "r") as f:
            epoch_count = int(f.read())
    else:
        pass
    
    try:
        for episode in range(0, EPISODES):
            if not game.wait_for_start():
                raise Exception('stop')
            alive, state, score, frame = game.step()
            if not alive:
                raise Exception('stop')
            print("Start an episode")
            s1 = preprocess(frame)
            rewards = []
            values = []
            actions = []
            heads = []
            with autograd.record():
                for learning_step in range(LEARNING_STEPS):
                    # Converts and down-samples the input image
                    with autograd.predict_mode():
                        prob, value = model(s1)
                    # dont always take the argmax, instead pick randomly based on probability
                    index, logp = mx.nd.sample_multinomial(prob, get_prob=True)
                    action = index.asnumpy()[0].astype(np.int64)
                    # skip frames
                    reward = 0
                    for skip in range(frame_repeat):
    #                     do some frame math to make it not all jumpy and weird
                        alive, state, score, frame = game.step(action)
                        if not alive:
                            raise Exception('stop')
    #                     can render image if we want
    #                     renderimage(proper_frame)
                        reward += score
                    nd.waitall()
                    isterminal = (state in (GameState.OVER, GameState.HANG))
                    rewards.append(reward)
                    actions.append(action)
                    values.append(value)
                    heads.append(logp)

                    if isterminal:
                        #print("finished_game")
                        break
                    s1 = preprocess(frame) if not isterminal else None

                train_scores.append(np.sum(rewards))
                # reverse accumulate and normalize rewards
                R = 0
                for i in range(len(rewards) - 1, -1, -1):
                    R = rewards[i] + gamma * R
                    rewards[i] = R
                # rewards = np.array(rewards)
                # rewards -= rewards.mean()
                # rewards /= rewards.std() + np.finfo(rewards.dtype).eps

                # compute loss and gradient
                L = sum([loss(value, mx.nd.array([r]).as_in_context(ctx)) for r, value in zip(rewards, values)])
                final_nodes = [L]
                for logp, r, v in zip(heads, rewards, values):
                    reward = r - v.asnumpy()[0, 0]
                    # Here we differentiate the stochastic graph, corresponds to the
                    # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
                    # Optimizer minimizes the loss but we want to maximizing the reward,
                    # so use we use -reward here.
                    final_nodes.append(logp * (-reward))
                autograd.backward(final_nodes)

            print("game over point: %f" % train_scores[-1])
            optimizer.step(s1.shape[0])

            if episode % DISPLAY_COUNT == 0:
                train_scores = np.array(train_scores)
                print("Episodes {}\t".format(episode + epoch_count),
                      "Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                      "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max(),
                      "actions: ", np.unique(actions, return_counts=True))
                train_scores = []
            if episode % 50 == 0 and episode != 0:
                model.collect_params().save("data/volleyball.params")
                with open("data/epoch.txt", "w") as f:
                    f.write(str(episode + epoch_count))
            
    except:
        import sys, os
        game.close()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise
    finally:
        model.collect_params().save("data/volleyball_temp.params")
        with open("data/epoch_temp.txt", "w") as f:
            f.write(str(episode + epoch_count))

train()