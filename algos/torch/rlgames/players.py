import time

import isaacgym
from rl_games.algos_torch.players import PpoPlayerContinuous

import torch as th


class PpoPlayerCustom(PpoPlayerContinuous):
    def __init__(self, params):
        self.player_observer = params['config']['features']['observer']
        super().__init__(params)
        self.player_observer.after_init(self)

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn

        self.player_observer.before_run()
        for trial in range(n_games):
            if games_played >= n_games:
                break

            obs = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obs, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = th.zeros(batch_size, dtype=th.float32)
            steps = th.zeros(batch_size, dtype=th.float32)
            self.end_idx = th.ones(batch_size, dtype=th.float32) * th.nan

            print_game_res = False

            self.player_observer.before_play()

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obs, masks, is_deterministic)
                else:
                    action = self.get_action(obs, is_deterministic)
                obs, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1
                self.player_observer.after_steps()

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = th.logical_and(done, (n > self.env.max_episode_length // 2).cpu()).sum()
                games_played += done_count

                if len(done_indices) > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()
                    self.end_idx[done_indices] = steps[done_indices]

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

            self.player_observer.after_play()

        self.player_observer.after_run()
        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

