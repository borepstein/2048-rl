import logging

import tensorflow as tf
import numpy as np

from py_2048_rl.game.game import Game
from py_2048_rl import episodes
from py_2048_rl import models

logger = logging.getLogger('py2048')


def random_action_callback(game):
    return np.random.choice(game.available_actions())


class Agent:
    def __init__(
            self,
            batch_size=10000,
            mem_size=50000,
            input_dims=[16],
            lr=0.001,
            gamma=0.99,
            gamma1=0.99,
            gamma2=0.99,
            epsilon=1,
            epsilon_dec=1e-3,
            epsilon_min=0.01,
            model_load_file=None,
            model_save_file='model.h5',
            model_auto_save=True,
            log_dir="/tmp/",
            training_epochs=1,
            report_sample_size =50,
            **kwargs
        ):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims[::]
        self.lr = lr
        self.gamma = gamma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_load_file = model_load_file
        self.model_save_file = model_save_file
        self.model_auto_save = model_auto_save
        self.log_dir = log_dir
        self.training_epochs = training_epochs
        self.report_sample_size = report_sample_size

        self.episode_db = episodes.EdpisodeDB(
            self.mem_size,
            self.input_dims,
        )

        if self.model_load_file:
            self.model = self.load_model()
        else:
            self.model = models.DEFAULT_MODEL
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss='mean_squared_error',
                metrics=['accuracy']
            )
        self.last_game_score = 0
        self.last_move_count = 0

    def learn(self, run):
        self.accumulate_episode_data()

        states, states_, actions, rewards, scores, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(self.batch_size)
        q_target[batch_index, actions] = 1 / tf.math.exp(
            tf.math.l2_normalize(
                rewards +
                self.gamma * np.max(q_next, axis=1) +
                self.gamma1 * scores.numpy() +
                self.gamma2 * scores.numpy() * dones.numpy()
            )
        )

        callbacks = []
        if self.log_dir:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                profile_batch='500,520'
            )
            callbacks.append(tb_callback)

        history = self.model.fit(
            states.numpy(),
            q_target,
            callbacks=callbacks,
            epochs=self.training_epochs
        )

        # Adjust the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec

        file_writer = tf.summary.create_file_writer(self.log_dir)
        file_writer.set_as_default()
        tf.summary.scalar('Epsilon', data=self.epsilon, step=run)

        # Log
        tf.summary.scalar('Game score', data=self.last_game_score, step=run)
        tf.summary.scalar('Game move', data=self.last_move_count, step=run)

        if run == 0:
            self.score_sample_arr = []

        self.score_sample_arr.append( self.last_game_score )
        if run > self.report_sample_size:
            self.score_sample_arr.pop(0)

        tf.summary.scalar('Sample minimum: last ' + self.report_sample_size.__str__(),
                          data=min(self.score_sample_arr),
                          step=run
                          )

        tf.summary.scalar('Sample average: last ' + self.report_sample_size.__str__(),
                          data=sum(self.score_sample_arr)/len(self.score_sample_arr),
                          step=run
                          )

        tf.summary.scalar('Sample maximum: last ' + self.report_sample_size.__str__(),
                          data=max(self.score_sample_arr),
                          step=run
                          )

        tf.summary.scalar('Sample range (min to max): last ' + self.report_sample_size.__str__(),
                          data=max(self.score_sample_arr) - min(self.score_sample_arr),
                          step=run
                          )

        for name in history.history:
            tf.summary.scalar(name, data=history.history[name][-1], step=run)

        # Close the writer
        file_writer.close()

    def learn_on_repeat(self, n_games=1):
        min_score = 0
        max_score = 0
        sum_scores = 0

        for i in range(n_games):
            self.learn(i)
            self.play_game(self.action_greedy_epsilon)

            if self.model_auto_save:
                self.save_model()

            if i != 0:
                min_score = min(min_score, self.last_game_score)
            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            avg_score = sum_scores / (i+1)

            logger.info('Step %d: min=%s avg=%s last=%s max=%s',
                        i, max_score, avg_score, self.last_game_score, max_score)

    def accumulate_episode_data(self):
        # Bail if there's nothing to do.
        if self.episode_db.mem_cntr >= self.batch_size:
            return

        logger.debug("Initial data accumulation. Collection size = %s episodes.",
                     self.mem_size)
        while self.episode_db.mem_cntr < self.batch_size:
            self.play_game(random_action_callback)
        logger.debug("Initial data accumulation completed.")

    def play_game(self, action_callback):
        game = Game()

        while not game.game_over():
            action = action_callback(game)
            state = np.matrix.flatten(game.state())
            reward = game.do_action(action)
            state_ = np.matrix.flatten(game.state())
            episode = episodes.Episode(
                state=state,
                next_state=state_,
                action=action,
                reward=reward,
                score=game.score(),
                done=game.game_over()
            )
            self.episode_db.store_episode(episode)

        self.last_game_score = game.score()
        self.last_move_count = game.move_count

    def action_greedy_epsilon(self, game):
        if np.random.random() < self.epsilon:
            return random_action_callback(game)

        state = game.state()
        state = np.matrix.reshape(state, (1, 16))

        actions = self.model.predict(state)
        actions = actions[0][game.available_actions()]
        return np.argmin(actions)

    def save_model(self):
        self.model.save(self.model_save_file)

    def load_model(self):
        return tf.keras.models.load_model(self.model_load_file)
