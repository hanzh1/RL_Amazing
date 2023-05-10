import sys
sys.path.append("")

import rl_algs.ddpg.ddpg as rl_alg
import Amazing
import numpy as np
import time
import pickle
import tensorflow as tf

def on_save(actor, q_network, epoch, replay_buffer):
    actor.save("Amazing/actor")
    q_network.save("Amazing/critic")

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("Amazing/actor"), tf.keras.models.load_model("Amazing/critic")

rl_alg.ddpg(lambda: Amazing.AmazingEnv(g=-9.8)
    , hp = rl_alg.HyperParams(
        seed=int(time.time()* 1e5) % int(1e6),
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes":(32,32),
            "critic_hidden_sizes":(256,256),
            "obs_normalizer": np.array([0.5, 0.25, 1, 1]) #MBW
        },
        pi_bar_variance=[0.0, 0.0, 0.0, 0.0],
        start_steps=1000,
        replay_size=int(1e5),
        gamma=0.9,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=200,
        act_noise=0.01,
        max_ep_len=200,
        epochs=200,
        train_every=50,
        train_steps=30,
    )
    , on_save=on_save
)