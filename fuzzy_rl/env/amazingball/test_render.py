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

# p.connect(p.GUI)
# p.resetSimulation()
# p.setRealTimeSimulation(0)

# targid = p.loadURDF("C:\\Users\\Han\\Desktop\\CS\\454\\FuzzyActorCritic\\fuzzy_rl\env\\amazingball\\assets\\plate.urdf")
# obj = targid

# for step in range(500):
#     pos, _ = p.getBasePositionAndOrientation(targid)
#     p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0,0,0])
#     p.stepSimulation()
#     time.sleep(.01)

env = Amazing.AmazingEnv(g=-9.8)
saved = tf.saved_model.load("Amazing/actor")
actor = lambda x: saved(np.array([x]))[0]
p.connect(p.GUI)


o, _ = env.reset()
high = env.action_space.high
low = env.action_space.low
os = []
for _ in range(200):
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0,0,0])
    a = actor(o)*(high - low)/2.0 + (high + low)/2.0
    o, r, d, t, i, = env.step(a)
    os.append(o)
    p.stepSimulation()
    time.sleep(0.01)