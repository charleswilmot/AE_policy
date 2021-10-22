import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from simulation import SimulationPool, MODEL_PATH
from buffer import Buffer


class Procedure:
    def __init__(self, config):
        self.encoder = models.Sequential([
            layers.Dense(50, activation=tf.nn.relu),
            layers.Dense(50, activation=tf.nn.relu),
            layers.Dense(7, activation=tf.tanh),
        ])
        self.decoder = models.Sequential([
            layers.Dense(50, activation=tf.nn.relu),
            layers.Dense(50, activation=tf.nn.relu),
            layers.Dense(25, activation=None),
        ])
        self.buffer = Buffer(config.buffer_size)
        self.simulations = SimulationPool(
            config.n_simulations,
            scene=MODEL_PATH + '/custom_timestep.ttt',
            # guis=[],
            guis=[0],
        )
        self.simulations.set_simulation_timestep(0.2)
        self.simulations.create_environment()
        self.simulations.set_reset_poses()
        self.simulations.set_control_loop_enabled(False)
        self.simulations.start_sim()
        self.simulations.step_sim()
        print("[procedure] all simulation started")
        self.optimizer = optimizers.Adam(config.learning_rate)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.simulations.close()

    def collect_data(self, n_iterations):
        all_states = []
        n_pool_iterations = n_iterations // self.simulations.size
        if n_pool_iterations * self.simulations.size != n_iterations:
            raise ValueError("Number of iterations not divisible by the Pool size")
        for iteration in range(n_pool_iterations):
            states = np.array(self.simulations.get_state())
            all_states.append(states)
            actions = self.encoder(states)
            with self.simulations.distribute_args():
                self.simulations.apply_action(tuple(a for a in actions))
        self.buffer.integrate(np.concatenate(all_states))

    def train(self, batch_size):
        states = self.buffer.sample(batch_size)
        with tf.GradientTape() as tape:
            reconstructions = self.decoder(self.encoder(states))
            loss = tf.reduce_sum(tf.reduce_mean((states - reconstructions) ** 2, axis=-1))
            vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        return loss

    def replay(self):
        pass
