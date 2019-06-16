import numpy as np
import tensorflow as tf
from algo.common.policies import nature_cnn
from algo.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample


class AcerCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi_logits = fc(h, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            q = fc(h, 'q', nact)

        a = sample(tf.nn.softmax(pi_logits))  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.pi_logits = pi_logits
        self.q = q
        self.vf = q

        def step(ob, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0 = sess.run([a, pi], {X: ob})
            return a0, pi0, []  # dummy state

        def out(ob, *args, **kwargs):
            pi0, q0 = sess.run([pi, q], {X: ob})
            return pi0, q0

        def act(ob, *args, **kwargs):
            return sess.run(a, {X: ob})

        self.step = step
        self.out = out
        self.act = act
