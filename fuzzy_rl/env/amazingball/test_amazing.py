import sys
sys.path.append("")

import tensorflow as tf
import numpy as np
import Amazing
import matplotlib.pyplot as plt
import pid
import time
import pickle
import pybullet as p

env = Amazing.AmazingEnv(g=10.)
# env.reset()
# env.render()
# time.sleep(100)
p.connect(p.GUI)
def test(actor, env, seed=123, render=True):
    o, _ = env.reset()
    high = env.action_space.high
    low = env.action_space.low
    os = []
    for _ in range(200):
        a = actor(o)*(high - low)/2.0 + (high + low)/2.0
        o, r, d, t, i, = env.step(a)
        os.append(o)
        # time.sleep(0.01)
        p.stepSimulation()
    print("END OF TEST!")
    return np.array(os)

saved = tf.saved_model.load("Amazing/actor")
actor = lambda x: saved(np.array([x]))[0]

for i in range(5):
    a = test(actor, env)
    env.reset()

# runs = np.array(list(map(lambda i: test(actor, env, seed=17+i, render=True)[:,1], range(5))))
