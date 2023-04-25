"""
Grid World toy environment.
"""
import time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class Grid:
    def __init__(self,
                 x: int = None,
                 y: int = None,
                 dtype: int = 0,
                 reward: float = 0,
                 value: float = 0.0):
        """
        :param x: coordinate x
        :param y: coordinate y
        :param dtype: (0: empty, 1: obstacle or boundary)
        :param reward: instant reward for an agent entering this grid cell
        :param value: name of this grid
        """
        self.x = x
        self.y = y
        self.type = dtype
        self.reward = reward
        self.value = value
        self.name = None
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{5}, x:{0}, y:{1}, type:{2}, value:{4}".format(self.x,
                                                                    self.y,
                                                                    self.type,
                                                                    self.reward,
                                                                    self.value,
                                                                    self.name)


class GridMatrix:
    def __init__(self,
                 n_width: int,
                 n_height: int,
                 default_dtype: int = 0,
                 default_reward: float = 0.0,
                 default_value: float = 0.0):
        """
        Grid matrix simulate variable grid world environments by setting different properties.
        :param n_width: defines the number of cells horizontally
        :param n_height: vertically
        :param default_dtype: default cell type
        :param default_reward: default instant reward
        :param default_value: default value
        :return:
        """
        self.grids = None
        self.n_width = n_width
        self.n_height = n_height
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_dtype = default_dtype
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_dtype, self.default_reward, self.default_value))

    def get_grid(self, x, y=None):
        """
        Get a grid information.
        :param x: coordinate x or tuple type of (x,y)
        :param y: coordinate y
        :return: grid object
        """
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        # assert (0 <= xx < self.n_width and 0 <= yy < self.n_height, "coordinates should be in reasonable range")
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise "grid doesn't exist"

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise "grid doesn't exist"

    def set_type(self, x, y, dtype):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = dtype
        else:
            raise "grid doesn't exist"

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_dtype(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 u_size: int = 40,
                 n_width: int = 10,
                 n_height: int = 7,
                 default_reward: float = 0,
                 default_type: int = 0,
                 windy: bool = False):
        """
        Grid World Environment used to simulate variable grid worlds.
        :param n_width: width of the env calculated by number of cells
        :param n_height: height...
        :param u_size: size for each cell (pixels)
        :param default_reward:
        :param default_type:
        :param windy:
        """
        self.u_size = u_size
        self.n_width = n_width
        self.n_height = n_height
        self.width = u_size * n_width
        self.height = u_size * n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_dtype=self.default_type,
                                default_reward=self.default_reward,
                                default_value=0.0)
        self.reward = 0
        self.action = None
        self.windy = windy
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(4)
        # Observation space is a vector decided by low and high.
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        # Left-bottom corner is the position of (0,0).
        self.ends = [(7, 3)]  # goal cells position list
        self.start = (0, 3)  # start cell position, only one start position
        self.types = []  # special type of cells, (x,y,z) represents in position (x,y) the cell type is z
        self.rewards = []  # special reward for a cell
        self.refresh_setting()
        self.viewer = None
        self._seed()  # generate a random seed
        self.reset()

        # minor attr
        self.state = None
        self.agent = None
        self.agent_trans = None

    def _adjust_size(self):
        """
        Adjust the range of max-width and max-height within 800.
        """
        pass

    def _seed(self, seed=None):
        """
        Generate a random seed and return a np_random object used for random generation below.
        :param seed: seed
        :return: np_random object
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "{!r} ({:s}) invalid".format(action, type(action))

        self.action = action  # action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        # wind effect: additional movement when leaving current grid
        if self.windy:
            if new_x in [3, 4, 5, 8]:
                new_y += 1
            elif new_x in [6, 7]:
                new_y += 2

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        # boundary effect
        if new_x < 0:
            new_x = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_height:
            new_y = self.n_height - 1

        # wall effect, obstacles or boundary
        if self.grids.get_dtype(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.state = self._xy_to_state(new_x, new_y)
        self.reward = self.grids.get_reward(new_x, new_y)
        done = self._is_end_state(new_x, new_y)
        info = {'x': new_x, 'y': new_y}
        return self.state, self.reward, done, info

    def _state_to_xy(self, s):
        """
        Set status into a one-axis coordinate value.
        :return: x, y
        """
        x = s % self.n_width
        assert (s - x) % self.n_width == 0, "State info error."
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        """
        Set coordinate value into status.
        :param x: coordinate x or tuple type of (x,y)
        :param y: coordinate y
        :return: status
        """
        if isinstance(x, int):
            assert (isinstance(y, int)), "Incomplete position info."
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # unknown status

    def refresh_setting(self):
        """
        Take effect after changing the property of grids. (if needed)
        """
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "Incomplete coordinate values."
            xx, yy = x[0], x[1]

        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        u_size = self.u_size
        m = 2  # gaps between two cells

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # draw cells
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [
                        (x * u_size + m, y * u_size + m),
                        ((x + 1) * u_size - m, y * u_size + m),
                        ((x + 1) * u_size - m, (y + 1) * u_size - m),
                        (x * u_size + m, (y + 1) * u_size - m)
                    ]
                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        # rect.set_color(0.3, 0.5 + r, 0.3)
                        rect.set_color(0.5, 0.5 + r, 0.5)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)

                    # draw frameworks
                    v_outline = [
                        (x * u_size + m, y * u_size + m),
                        ((x + 1) * u_size - m, y * u_size + m),
                        ((x + 1) * u_size - m, (y + 1) * u_size - m),
                        (x * u_size + m, (y + 1) * u_size - m)
                    ]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # give end state cell a golden outline
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_dtype(x, y) == 1:
                        # give obstacle cell a gray outline
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass

            # draw agent
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        # update position of an agent
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


class MiniWorld(GridWorldEnv):
    """
    6*6 grid world.
    """

    def __init__(self):
        super().__init__(u_size=60,
                         n_width=6,
                         n_height=6,
                         default_reward=-0.1,
                         default_type=0,
                         windy=False)
        self.start = (0, 5)
        self.ends = [(5, 3), (3, 1)]
        self.types = [
            (2, 2, 1), (3, 2, 1), (3, 3, 1), (3, 4, 1),
        ]
        self.blocks = [(2, 2), (3, 2), (3, 3), (3, 4)]
        self.rewards = [(3, 1, 1), (5, 3, -1)]
        self.R = -0.1 * np.ones([self.n_width * self.n_height])
        for (x, y, r) in self.rewards:
            s = self._xy_to_state(x, y)
            self.R[s] = r
        self.action_meaning = ["^", ">", "v", "<"]
        self.max_step = 100
        self.refresh_setting()

    def update_r(self, values):
        """
        This function is used for rendering env when finally get a value table.
        """
        v_min = values.min()
        v_max = values.max()
        v_norm = (2 * (values - v_min) / (v_max - v_min) - 1) * 8.0
        for i in range(len(v_norm)):
            x, y = self._state_to_xy(i)
            self.grids.set_reward(x, y, v_norm[i])

    def show_values(self, values, sec=3):
        self.reset()
        self.update_r(values)
        self.render()
        time.sleep(sec)

    def show_policy(self, policy):
        result = [["" for _ in range(self.n_width)] for _ in range(self.n_height)]
        actions = self.action_meaning
        for s in range(self.n_height * self.n_width):
            row, col = self._state_to_xy(s)
            if (row, col) == self.start:
                result[row][col] = "SSSS"
            elif (row, col) in self.ends:
                result[row][col] = "EEEE"
            elif (row, col) in self.blocks:
                result[row][col] = "****"
            else:
                pi_str = ""
                max_p = max(policy[s])
                for a in range(len(actions)):
                    pi_str += actions[a] if policy[s][a] == max_p else 'o'
                result[row][col] = pi_str
        result = [[result[x][self.n_height - 1 - y] for x in range(self.n_width)] for y in range(self.n_height)]
        for row in result:
            for a in row:
                print(a, end=' ')
            print()


if __name__ == "__main__":
    env = GridWorldEnv()
    env.reset()
    n_state = env.observation_space
    n_action = env.action_space
    print(f"state space: {n_state}, action space:{n_action}")
    env.render()
    for _ in range(1000):
        env.render()

    print("env closed")
