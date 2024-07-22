import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    num_reset = 0

    def __init__(self, render_mode=None, player=1, file_name=None):
        super(TicTacToeEnv, self).__init__()

        self.render_mode = render_mode
        self.file_name = file_name
        self.player = player
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(10,),
        )
        self.reset()
        self.num_reset = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            (3, 3),
        )
        self.current_player = 1
        self.player *= -1
        self.done = False

        if self.render_mode is not None:
            self.render()

        self.num_reset += 1
        if self.num_reset == 250000:
            self.player = -1
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(
            10,
        )
        for i in range(9):
            row, col = divmod(i, 3)
            obs[i] = 1 if self.board[row, col] == 0 else 0
        obs[9] = 1 if self.current_player == 1 else 0
        # print("get_obs", obs)
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        row, col = divmod(action, 3)

        if self.board[row, col] != 0:
            return self._get_obs(), -10, True, False, {}

        self.board[row, col] = self.current_player
        # empty_places = np.count_nonzero(self.board == 0)
        if self._check_win():
            self.done = True
            reward = 1
            if self.current_player != self.player:
                reward *= -1
        elif np.all(self.board != 0):
            self.done = True
            reward = 0  # draw
        else:
            self.current_player *= -1
            reward = 0

        if self.render_mode is not None:
            self.render()

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        self._render_console()

    def close(self):
        print("-----------------")
        print("Closing environment ...")
        return super().close()

    def _render_console(self):
        if self.file_name is not None:
            with open(self.file_name, "a") as f:
                f.write(f"Current player: {'X' if self.current_player == 1 else 'O'}\n")
                for i in range(3):
                    for j in range(3):
                        if self.board[i, j] == 1:
                            f.write("X ")
                        elif self.board[i, j] == -1:
                            f.write("O ")
                        else:
                            f.write(". ")
                    f.write("\n")
                f.write("\n")
        # print(f"Current player: {'X' if self.current_player == 1 else 'O'}")
        # for i in range(3):
        #    for j in range(3):
        #        if self.board[i, j] == 1:
        #            print("X", end=" ")
        #        elif self.board[i, j] == -1:
        #            print("O", end=" ")
        #        else:
        #            print(".", end=" ")
        #    print()
        # print()

    def _check_win(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return True
        if (
            abs(np.sum(np.diag(self.board))) == 3
            or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3
        ):
            return True
        return False


gym.envs.registration.register(
    id="TicTacToe-v0", entry_point="cleanrl.ttt:TicTacToeEnv"
)

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np


# class TicTacToeEnv(gym.Env):
#    metadata = {"render_modes": ["human"]}

#    def __init__(self, render_mode=None, player=1):
#        super(TicTacToeEnv, self).__init__()

#        self.render_mode = render_mode
#        self.player = player
#        self.action_space = spaces.Discrete(9)
#        self.observation_space = spaces.Box(
#            low=0,
#            high=1,
#            shape=(19,),
#        )
#        self.reset()

#    def reset(self, seed=None, options=None):
#        super().reset(seed=seed)
#        self.board = np.zeros((3, 3))
#        self.current_player = 1
#        self.player *= -1
#        self.done = False

#        if self.render_mode is not None:
#            self.render()

#        return self._get_obs(), {}

#    def _get_obs(self):
#        obs = np.zeros(19, dtype=np.float32)
#        for i in range(9):
#            row, col = divmod(i, 3)
#            obs[i] = 1 if self.board[row, col] == 0 else 0
#        obs[9] = 1 if self.current_player == 1 else 0
#        obs[10:] = self._legal_moves()
#        return obs

#    def _legal_moves(self):
#        legal_moves = np.zeros(9, dtype=np.float32)
#        for i in range(9):
#            row, col = divmod(i, 3)
#            if self.board[row, col] == 0:
#                legal_moves[i] = 1
#        return legal_moves

#    def step(self, action):
#        if self.done:
#            return self._get_obs(), 0, True, False, {}

#        row, col = divmod(action, 3)

#        if self.board[row, col] != 0:
#            return self._get_obs(), -100, True, False, {}

#        self.board[row, col] = self.current_player
#        # empty_places = np.count_nonzero(self.board == 0)
#        if self._check_win():
#            self.done = True
#            reward = 10  # win from the perspective of the current player
#            if self.current_player != self.player:
#                reward = 1  # lose from the perspective of the current player (opponent wins) nonetheless we want to encourage the agent to win as fast as possible no matter on which side it is
#        elif np.all(self.board != 0):
#            self.done = True
#            reward = 0  # draw
#        else:
#            self.current_player *= -1
#            reward = 0

#        if self.render_mode is not None:
#            self.render()

#        return self._get_obs(), reward, self.done, False, {}

#    def render(self):
#        self._render_console()

#    def close(self):
#        print("-----------------")
#        print("Closing environment ...")
#        return super().close()

#    def _render_console(self):
#        print(f"Current player: {'X' if self.current_player == 1 else 'O'}")
#        for i in range(3):
#            for j in range(3):
#                if self.board[i, j] == 1:
#                    print("X", end=" ")
#                elif self.board[i, j] == -1:
#                    print("O", end=" ")
#                else:
#                    print(".", end=" ")
#            print()
#        print()

#    def _check_win(self):
#        for i in range(3):
#            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
#                return True
#        if (
#            abs(np.sum(np.diag(self.board))) == 3
#            or abs(np.sum(np.sum(np.fliplr(self.board)))) == 3
#        ):
#            return True
#        return False


# gym.envs.registration.register(
#    id="TicTacToe-v1", entry_point="cleanrl.ttt:TicTacToeEnv"
# )

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame


# class TicTacToeEnv(gym.Env):
#    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

#    def __init__(self, render_mode=None, player=1):
#        super(TicTacToeEnv, self).__init__()

#        self.render_mode = render_mode
#        self.player = player
#        self.action_space = spaces.Discrete(9)
#        self.observation_space = spaces.Box(
#            low=0, high=1, shape=(10,), dtype=np.float32
#        )
#        self.window = None
#        self.clock = None
#        self.cell_size = 100
#        self.window_size = self.cell_size * 3
#        self.reset()

#    def reset(self, seed=None, options=None):
#        super().reset(seed=seed)
#        self.board = np.zeros((3, 3))
#        self.current_player = 1
#        self.player *= -1
#        self.done = False

#        if self.render_mode is not None and self.render_mode == "human":
#            self._init_pygame()

#        return self._get_obs(), {}

#    def _get_obs(self):
#        obs = np.zeros(10, dtype=np.float32)
#        for i in range(9):
#            row, col = divmod(i, 3)
#            obs[i] = 1 if self.board[row, col] == 0 else 0
#        obs[9] = 1 if self.current_player == 1 else 0
#        return obs

#    def _legal_moves(self):
#        legal_moves = np.zeros(9, dtype=np.float32)
#        for i in range(9):
#            row, col = divmod(i, 3)
#            if self.board[row, col] == 0:
#                legal_moves[i] = 1
#        return legal_moves

#    def step(self, action):
#        if self.done:
#            return self._get_obs(), 0, True, False, {}

#        row, col = divmod(action, 3)

#        if self.board[row, col] != 0:
#            return self._get_obs(), -100, True, False, {}

#        self.board[row, col] = self.current_player
#        if self._check_win():
#            self.done = True
#            reward = 10  # win from the perspective of the current player
#            if self.current_player != self.player:
#                reward = (
#                    1  # lose from the perspective of the current player (opponent wins)
#                )
#        elif np.all(self.board != 0):
#            self.done = True
#            reward = 0  # draw
#        else:
#            self.current_player *= -1
#            reward = 0

#        if self.render_mode is not None:
#            self.render()

#        return self._get_obs(), reward, self.done, False, {}

#    def render(self):
#        if self.render_mode == "human":
#            self._render_pygame()
#        elif self.render_mode == "rgb_array":
#            return self._render_rgb_array()

#    def close(self):
#        # if self.window is not None:
#        # pygame.quit()
#        print("-----------------")
#        print("Closing environment ...")
#        return super().close()

#    def _init_pygame(self):
#        if self.window is None:
#            pygame.init()
#            self.window = pygame.display.set_mode((self.window_size, self.window_size))
#            pygame.display.set_caption("Tic Tac Toe")
#            self.clock = pygame.time.Clock()

#    def _render_pygame(self):
#        if self.window is None:
#            self._init_pygame()

#        self.window.fill((255, 255, 255))

#        # Draw grid
#        for x in range(1, 3):
#            pygame.draw.line(
#                self.window,
#                (0, 0, 0),
#                (0, x * self.cell_size),
#                (self.window_size, x * self.cell_size),
#                3,
#            )
#            pygame.draw.line(
#                self.window,
#                (0, 0, 0),
#                (x * self.cell_size, 0),
#                (x * self.cell_size, self.window_size),
#                3,
#            )

#        # Draw pieces
#        for i in range(3):
#            for j in range(3):
#                if self.board[i, j] == 1:
#                    pygame.draw.line(
#                        self.window,
#                        (0, 0, 0),
#                        (j * self.cell_size, i * self.cell_size),
#                        ((j + 1) * self.cell_size, (i + 1) * self.cell_size),
#                        3,
#                    )
#                    pygame.draw.line(
#                        self.window,
#                        (0, 0, 0),
#                        ((j + 1) * self.cell_size, i * self.cell_size),
#                        (j * self.cell_size, (i + 1) * self.cell_size),
#                        3,
#                    )
#                elif self.board[i, j] == -1:
#                    pygame.draw.circle(
#                        self.window,
#                        (0, 0, 0),
#                        (
#                            j * self.cell_size + self.cell_size // 2,
#                            i * self.cell_size + self.cell_size // 2,
#                        ),
#                        self.cell_size // 2,
#                        3,
#                    )

#        pygame.display.update()
#        self.clock.tick(self.metadata["render_fps"])

#    def _render_rgb_array(self):
#        if self.window is None:
#            self._init_pygame()

#        self._render_pygame()
#        return np.transpose(
#            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
#        )

#    def _check_win(self):
#        for i in range(3):
#            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
#                return True
#        if (
#            abs(np.sum(np.diag(self.board))) == 3
#            or abs(np.sum(np.sum(np.fliplr(self.board)))) == 3
#        ):
#            return True
#        return False


# gym.envs.registration.register(
#    id="TicTacToe-v1", entry_point="cleanrl.ttt:TicTacToeEnv"
# )

## Usage with RecordVideo
# def main():
#    env = gym.make('TicTacToe-v1', render_mode="human")
#    video_dir = "./video"

#    if not os.path.exists(video_dir):
#        os.makedirs(video_dir)

#    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)
#    obs, _ = env.reset()

#    done = False
#    while not done:
#        action = env.action_space.sample()
#        obs, reward, done, _, _ = env.step(action)
#        env.render()

#    env.close()

# if __name__ == "__main__":
#    main()
