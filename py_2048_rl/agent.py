import importlib
import logging

import tensorflow as tf
import numpy as np

from py_2048_game import Game
from py_2048_rl import episodes

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
            gamma1=0.0,
            gamma2=0.99,
            gamma3=0.0,
            min_base = 1e-06,
            q_base = 1.0,
            epsilon=1,
            epsilon_dec=1e-3,
            epsilon_min=0.01,
            model_path='py_2048_rl.models.DEFAULT_MODEL',
            model_load_file=None,
            model_save_file='model.h5',
            model_auto_save=True,
            model_collect_random_data=True,
            log_dir="/tmp/",
            training_epochs=1,
            game_qc_threshold=0.5,
            game_max_replay_on_fail=500,
            **kwargs
        ):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims[::]
        self.lr = lr
        self.gamma = gamma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.min_base = min_base
        self.q_base = q_base
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_path = model_path
        self.model_load_file = model_load_file
        self.model_save_file = model_save_file
        self.model_auto_save = model_auto_save
        self.model_collect_random_data = model_collect_random_data
        self.log_dir = log_dir
        self.training_epochs = training_epochs
        self.game_qc_threshold = game_qc_threshold
        self.game_max_replay_on_fail = game_max_replay_on_fail

        self.episode_db = episodes.EdpisodeDB(
            self.mem_size,
            self.input_dims,
        )

        if self.log_dir:
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                profile_batch='500,520'
            )

        if self.model_load_file:
            try:
                self.model = self.load_model()
                # Disabling pre-loading random data for model pre-loaded from file.
                self.model_collect_random_data = False
            except OSError as err:
                logger.warning("Cannot load model %s: %s", self.model_load_file, err)
                logger.warning("Building new model")
                self.model = self._make_model()
        else:
            self.model = self._make_model()

        self.game_count = 0
        self.infer_game_count = 0
        self.last_game_score = 0
        self.last_move_count = 0
        self.q_val_opt_max=True
        self.max_game_score = 0
        self.min_game_score = 0

    def get_epsilon(self):
        val = self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec

        return val

    def arg_sel_func(self):
        if self.q_val_opt_max:
            return np.argmax
        return np.argmin

    def get_q_mod_func(self):
        if self.q_val_opt_max:
            return self.get_modeling_data_q_max
        return self.get_modeling_data_q_min

    def _make_model(self):
        class_name = self.model_path.split('.')[-1]
        module_path = '.'.join([i for i in self.model_path.split('.')][:-1])
        models = importlib.import_module(module_path)
        model = getattr(models, class_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mean_squared_error'
        )
        return model

    def get_modeling_data_q_min(self):
        # Return None if there is no data to learn from
        if self.episode_db.mem_cntr == 0:
            return None, None

        sel_size, states, states_, actions, rewards, scores, n_moves, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(sel_size)
        q_target[batch_index, actions] = 1/(
            self.q_base +
            rewards +
            self.gamma *
            1/(self.min_base + np.min(q_next, axis=1)) +
            self.gamma1 * scores.numpy() +
            self.gamma2 * scores.numpy() *
            dones.numpy() +
            self.gamma3 * n_moves.numpy()
        )

        return states, q_target

    def get_modeling_data_q_max(self):
        # Return None if there is no data to learn from
        if self.episode_db.mem_cntr == 0:
            return None, None

        sel_size, states, states_, actions, rewards, scores, n_moves, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(sel_size)
        q_target[batch_index, actions] = tf.math.l2_normalize(
            self.q_base +
            rewards +
            self.gamma * np.max(q_next, axis=1) +
            self.gamma1 * tf.math.reduce_sum(tf.cast(tf.equal(states, 0), tf.int32), axis=1
                                             ).numpy() +
            self.gamma2 * scores.numpy() *
            dones.numpy() +
            self.gamma3 * np.max(states, axis=1)
        )

        return states, q_target

    def learn(self, run):
        if self.model_collect_random_data:
            self.accumulate_episode_data()

        q_mod_func = self.get_q_mod_func()
        states, q_target = q_mod_func()

        if q_target is None:
            return

        callbacks = []
        if self.log_dir:
            callbacks.append(self.tb_callback)

        history = self.model.fit(
            states.numpy(),
            q_target,
            callbacks=callbacks,
            epochs=self.training_epochs
        )

        # Log
        for name in history.history:
            tf.summary.scalar(name, data=history.history[name][-1], step=run)

    def learn_on_repeat(self,
                        n_cycles=1,
                        games_per_cycle=1,
                        refill_episode_db=False,
                        inference_on_learn=False):
        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_cycles):
            self.learn(i)

            if self.model_auto_save and self.episode_db.mem_cntr > 0:
                self.save_model()

            episode_count = 0
            cycle_game_count = 0

            while True:
                self.play_game(self.action_greedy_epsilon)
                last_training_score = data=self.last_game_score
                last_training_mc = self.last_move_count

                if inference_on_learn:
                    self.play_on_repeat(tb_logging=False, record_in_episode_db=True)
                    tf.summary.scalar('Game score (inference)', data=self.last_game_score, step=self.infer_game_count)

                cycle_game_count += 1
                episode_count += last_training_mc
                self.game_count += 1

                if self.game_count == 1:
                    min_score = last_training_score
                else:
                    min_score = min(min_score, last_training_score)

                max_score = max(max_score, last_training_score)
                sum_scores += last_training_score
                avg_score = sum_scores / self.game_count

                logger.info('Game %d: min=%s avg=%s last=%s max=%s',
                            self.game_count, min_score, avg_score, self.last_game_score, max_score)

                tf.summary.scalar('Game score', data=last_training_score, step=self.game_count)
                tf.summary.scalar('Game score (average)', data=avg_score, step=self.game_count)
                tf.summary.scalar('Game move', data=last_training_mc, step=self.game_count)
                tf.summary.scalar('Epsilon', data=self.epsilon, step=self.game_count)
                tf.summary.scalar('Gamma', data=self.gamma, step=self.game_count)
                tf.summary.scalar('Gamma1', data=self.gamma1, step=self.game_count)
                tf.summary.scalar('Gamma2', data=self.gamma2, step=self.game_count)
                tf.summary.scalar('Gamma3', data=self.gamma3, step=self.game_count)
                tf.summary.scalar('q_base', data=self.q_base, step=self.game_count)
                tf.summary.scalar('lr', data=self.lr, step=self.game_count)
                tf.summary.scalar('Episode DB: mem_size', data=self.episode_db.mem_size, step=self.game_count)
                tf.summary.scalar('Episode DB: total_thruput', data=self.episode_db.total_thruput, step=self.game_count)
                tf.summary.scalar('Selection batch size', data=self.batch_size, step=self.game_count)

                if ((not refill_episode_db) and
                    (cycle_game_count >= games_per_cycle)) or \
                    ((refill_episode_db) and
                     (episode_count >= self.episode_db.mem_size)):
                    break

            if self.log_dir:
                file_writer.flush()

        if self.log_dir:
            file_writer.close()

    def accumulate_episode_data(self):
        # Bail if there's nothing to do.
        if self.episode_db.mem_cntr >= self.mem_size:
            return

        if not self.model_collect_random_data:
            return

        logger.debug("Initial data accumulation. Collection size = %s episodes.",
                     self.mem_size)
        while self.episode_db.mem_cntr < self.mem_size:
            self.play_game(random_action_callback)
        logger.debug("Initial data accumulation completed.")

    def game_qc(self, game):
        # Return True is no data thus far.
        if self.max_game_score == 0:
            return True

        if game.score < self.game_qc_threshold * self.max_game_score:
            return False

        return True

    def play_game(self,
                  action_callback,
                  replay_on_fail=True,
                  max_replays=0,
                  record_in_episode_db=True
                  ):
        game = Game()
        episode_arr = []
        score_pass = False
        replay_cnt = 0

        replay_lim = max_replays

        if replay_on_fail and replay_lim == 0:
            replay_lim = self.game_max_replay_on_fail

        while True:
            episode_arr = []

            while not game.game_over():
                action = action_callback(game)
                state = np.matrix.flatten(game.state)
                reward = game.do_action(action)
                state_ = np.matrix.flatten(game.state)
                episode_arr.append(
                    episodes.Episode(
                        state=state,
                        next_state=state_,
                        action=action,
                        reward=reward,
                        score=game.score,
                        n_moves=game.move_count,
                        done=game.game_over()
                    )
                )

            score_pass = self.game_qc(game)
            replay_cnt += 1

            if not record_in_episode_db:
                break

            if score_pass:
                break

            if not replay_on_fail:
                break

            if replay_on_fail and replay_cnt >= replay_lim:
                break

        if score_pass and record_in_episode_db:
            for e in episode_arr:
                self.episode_db.store_episode(e)

        self.max_game_score = max(self.max_game_score, game.score)

        if self.min_game_score == 0:
            self.min_game_score = game.score
        else:
            self.min_game_score = min(self.min_game_score, game.score)

        self.last_game_score = game.score
        self.last_move_count = game.move_count

    def action_greedy_epsilon(self, game):
        if np.random.random() < self.get_epsilon():
            return random_action_callback(game)

        return self.action_greedy(game)

    def action_greedy(self, game):
        state = game.state
        state = np.matrix.reshape(state, (1, 16))

        pred_actions = self.model.predict(state)[0]
        avai_actions = game.available_actions()
        sel_f = self.arg_sel_func()
        return avai_actions[sel_f(pred_actions[avai_actions])]

    def play_on_repeat(self, n_games=1, tb_logging=True, record_in_episode_db=False):
        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir and tb_logging:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_games):
            self.play_game(self.action_greedy,record_in_episode_db=record_in_episode_db)
            self.infer_game_count += 1

            if i == 0:
                min_score = self.last_game_score
            else:
                min_score = min(min_score, self.last_game_score)

            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            avg_score = sum_scores / (i+1)

            logger.info('Step %d: min=%s avg=%s last=%s max=%s',
                        i, min_score, avg_score, self.last_game_score, max_score)

            if self.log_dir and tb_logging:
                tf.summary.scalar('Game score (inference)', data=self.last_game_score, step=self.infer_game_count)

        if self.log_dir and tb_logging:
            file_writer.close()

    def save_model(self):
        self.model.save(self.model_save_file)

    def load_model(self):
        return tf.keras.models.load_model(self.model_load_file)
