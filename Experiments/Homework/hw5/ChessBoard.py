from Chess import *
from ClickBox import *


class ChessBoard(object):
    """
    棋盘类
    """

    def __init__(self, screen):
        """初始化"""
        self.screen = screen
        self.image = pygame.image.load("images/bg.png")
        self.topleft = (50, 50)
        self.chessboard_map = None  # 用来存储当前棋盘上的所有棋子对象
        self.create_chess()  # 调用创建棋盘的方法

    def show(self):
        # 显示棋盘
        self.screen.blit(self.image, self.topleft)

    def show_chess(self):
        """显示当前棋盘上的所有棋子"""
        # 显示棋盘上的所有棋子
        for line_chess in self.chessboard_map:
            for chess in line_chess:
                if chess:
                    chess.show()

    def show_chessboard_and_chess(self):
        """显示棋盘以及当前棋盘上所有的棋子"""
        self.show()
        self.show_chess()

    def create_chess(self):
        """创建默认棋盘上的棋子对象"""
        # 棋子
        self.chessboard_map = [
            ["b_c", "b_m", "b_x", "b_s", "b_j", "b_s", "b_x", "b_m", "b_c"],
            ["", "", "", "", "", "", "", "", ""],
            ["", "b_p", "", "", "", "", "", "b_p", ""],
            ["b_z", "", "b_z", "", "b_z", "", "b_z", "", "b_z"],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["r_z", "", "r_z", "", "r_z", "", "r_z", "", "r_z"],
            ["", "r_p", "", "", "", "", "", "r_p", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["r_c", "r_m", "r_x", "r_s", "r_j", "r_s", "r_x", "r_m", "r_c"],
        ]
        # self.chessboard_map = [
        #     ["", "", "", "", "b_j", "", "", "", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["r_z", "", "r_z", "", "r_z", "", "r_z", "", "r_z"],
        #     ["", "r_p", "", "", "", "", "", "r_p", ""],
        #     ["", "", "", "", "", "", "", "", ""],
        #     ["r_c", "r_m", "r_x", "r_s", "r_j", "r_s", "r_x", "r_m", "r_c"],
        # ]
        for row, line in enumerate(self.chessboard_map):
            for col, chess_name in enumerate(line):
                if chess_name:
                    # 将创建的棋子添加到属性map中
                    self.chessboard_map[row][col] = Chess(self.screen, chess_name, row, col)
                else:
                    self.chessboard_map[row][col] = None

    def get_chessboard_str_map(self):
        str_map = list()
        for row in range(10):
            line_map = list()
            for col in range(9):
                if self.chessboard_map[row][col]:
                    line_map.append(self.chessboard_map[row][col].team + '_' + self.chessboard_map[row][col].name)
                else:
                    line_map.append('')
            str_map.append(line_map)
        return str_map

    def set_chessboard_str_map(self, str_map):
        for row, line in enumerate(str_map):
            for col, chess_name in enumerate(line):
                if chess_name:
                    # 将创建的棋子添加到属性map中
                    self.chessboard_map[row][col] = Chess(self.screen, chess_name, row, col)
                else:
                    self.chessboard_map[row][col] = None

    def get_chess(self):
        """获取所有的棋盘上的棋子对象列表"""
        return [chess for line in self.chessboard_map for chess in line if chess]

    def get_put_down_position(self, clicked_chess):
        """获取当前被点击棋子可以落子的位置坐标"""
        put_down_chess_pos = list()
        # put_down_chess_pos.append((clicked_chess.row - 1, clicked_chess.col))
        # put_down_chess_pos.append((clicked_chess.row + 1, clicked_chess.col))
        # put_down_chess_pos.append((clicked_chess.row, clicked_chess.col - 1))
        # put_down_chess_pos.append((clicked_chess.row, clicked_chess.col + 1))
        team = clicked_chess.team
        row = clicked_chess.row
        col = clicked_chess.col
        map_ = self.chessboard_map

        if clicked_chess.name == "z":  # 卒
            if team == "r":  # 红方
                if row - 1 >= 0:  # 只能向上移动
                    if not map_[row - 1][col] or map_[row - 1][col].team != team:
                        put_down_chess_pos.append((row - 1, col))
            else:  # 黑方
                if row + 1 <= 9:  # 只能向下移动
                    if not map_[row + 1][col] or map_[row + 1][col].team != team:
                        put_down_chess_pos.append((row + 1, col))
            # 左右判断
            if (team == "r" and 0 <= row <= 4) or (team == "b" and 5 <= row <= 9):  # 左、右一步
                # 左
                if col - 1 >= 0 and (not map_[row][col - 1] or map_[row][col - 1].team != team):
                    put_down_chess_pos.append((row, col - 1))
                # 右
                if col + 1 <= 8 and (not map_[row][col + 1] or map_[row][col + 1].team != team):
                    put_down_chess_pos.append((row, col + 1))
        elif clicked_chess.name == "j":  # 将
            # 因为"将"是不能过河的，所以要计算出它们可以移动的行的范围
            row_start, row_stop = (0, 2) if team == "b" else (7, 9)
            # 有4个方向的判断
            if row - 1 >= row_start and (not map_[row - 1][col] or map_[row - 1][col].team != team):
                put_down_chess_pos.append((row - 1, col))
            if row + 1 <= row_stop and (not map_[row + 1][col] or map_[row + 1][col].team != team):
                put_down_chess_pos.append((row + 1, col))
            if col - 1 >= 3 and (not map_[row][col - 1] or map_[row][col - 1].team != team):
                put_down_chess_pos.append((row, col - 1))
            if col + 1 <= 5 and (not map_[row][col + 1] or map_[row][col + 1].team != team):
                put_down_chess_pos.append((row, col + 1))
        elif clicked_chess.name == "s":  # 士
            # 因为士是不能过河的，所以要计算出它们可以移动的行的范围
            row_start, row_stop = (0, 2) if team == "b" else (7, 9)
            if row - 1 >= row_start and col - 1 >= 3 and (not map_[row - 1][col - 1] or map_[row - 1][col - 1].team != team):
                put_down_chess_pos.append((row - 1, col - 1))
            if row - 1 >= row_start and col + 1 <= 5 and (not map_[row - 1][col + 1] or map_[row - 1][col + 1].team != team):
                put_down_chess_pos.append((row - 1, col + 1))
            if row + 1 <= row_stop and col - 1 >= 3 and (not map_[row + 1][col - 1] or map_[row + 1][col - 1].team != team):
                put_down_chess_pos.append((row + 1, col - 1))
            if row + 1 <= row_stop and col + 1 <= 5 and (not map_[row + 1][col + 1] or map_[row + 1][col + 1].team != team):
                put_down_chess_pos.append((row + 1, col + 1))
        elif clicked_chess.name == "x":  # 象
            # 因为象是不能过河的，所以要计算出它们可以移动的行的范围
            row_start, row_stop = (0, 4) if team == "b" else (5, 9)
            # 有4个方向的判断(没有越界，且没有蹩象腿)
            if row - 2 >= row_start and col - 2 >= 0 and not map_[row - 1][col - 1]:  # 左上
                if not map_[row - 2][col - 2] or map_[row - 2][col - 2].team != team:
                    put_down_chess_pos.append((row - 2, col - 2))
            if row - 2 >= row_start and col + 2 <= 8 and not map_[row - 1][col + 1]:  # 右上
                if not map_[row - 2][col + 2] or map_[row - 2][col + 2].team != team:
                    put_down_chess_pos.append((row - 2, col + 2))
            if row + 2 <= row_stop and col - 2 >= 0 and not map_[row + 1][col - 1]:  # 左下
                if not map_[row + 2][col - 2] or map_[row + 2][col - 2].team != team:
                    put_down_chess_pos.append((row + 2, col - 2))
            if row + 2 <= row_stop and col + 2 <= 8 and not map_[row + 1][col + 1]:  # 右下
                if not map_[row + 2][col + 2] or map_[row + 2][col + 2].team != team:
                    put_down_chess_pos.append((row + 2, col + 2))
        elif clicked_chess.name == "m":  # 马
            # 需要判断的是4个方向，每个方向对应2个位置
            # 上方
            if row - 1 >= 0 and not map_[row - 1][col]:  # 如果当前棋子没有被蹩马腿，那么再对这个方向的2个位置进行判断
                # 左上
                if row - 2 >= 0 and col - 1 >= 0 and (not map_[row - 2][col - 1] or map_[row - 2][col - 1].team != team):
                    put_down_chess_pos.append((row - 2, col - 1))
                # 右上
                if row - 2 >= 0 and col + 1 <= 8 and (not map_[row - 2][col + 1] or map_[row - 2][col + 1].team != team):
                    put_down_chess_pos.append((row - 2, col + 1))
            # 下方
            if row + 1 <= 9 and not map_[row + 1][col]:  # 如果当前棋子没有被蹩马腿，那么再对这个方向的2个位置进行判断
                # 左下
                if row + 2 <= 9 and col - 1 >= 0 and (not map_[row + 2][col - 1] or map_[row + 2][col - 1].team != team):
                    put_down_chess_pos.append((row + 2, col - 1))
                # 右下
                if row + 2 <= 9 and col + 1 <= 8 and (not map_[row + 2][col + 1] or map_[row + 2][col + 1].team != team):
                    put_down_chess_pos.append((row + 2, col + 1))
            # 左方
            if col - 1 >= 0 and not map_[row][col - 1]:  # 如果当前棋子没有被蹩马腿，那么再对这个方向的2个位置进行判断
                # 左上2（因为有左上了，暂且称为左上2吧）
                if row - 1 >= 0 and col - 2 >= 0 and (not map_[row - 1][col - 2] or map_[row - 1][col - 2].team != team):
                    put_down_chess_pos.append((row - 1, col - 2))
                # 左下2
                if row + 1 <= 9 and col - 2 >= 0 and (not map_[row + 1][col - 2] or map_[row + 1][col - 2].team != team):
                    put_down_chess_pos.append((row + 1, col - 2))
            # 右方
            if col + 1 <= 8 and not map_[row][col + 1]:  # 如果当前棋子没有被蹩马腿，那么再对这个方向的2个位置进行判断
                # 右上2（因为有右上了，暂且称为右上2吧）
                if row - 1 >= 0 and col + 2 <= 8 and (not map_[row - 1][col + 2] or map_[row - 1][col + 2].team != team):
                    put_down_chess_pos.append((row - 1, col + 2))
                # 右下2
                if row + 1 <= 9 and col + 2 <= 8 and (not map_[row + 1][col + 2] or map_[row + 1][col + 2].team != team):
                    put_down_chess_pos.append((row + 1, col + 2))
        elif clicked_chess.name == "c":  # 车
            # 一行
            left_stop = False
            right_stop = False
            for i in range(1, 9):
                # 左边位置没有越界且没有遇到任何一个棋子
                if not left_stop and col - i >= 0:
                    if not map_[row][col - i]:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row, col - i))
                    else:
                        left_stop = True
                        if map_[row][col - i].team != team:
                            # 如果当前位置有棋子，那么就判断是否能够吃掉它
                            put_down_chess_pos.append((row, col - i))
                # 右边位置没有越界且没有遇到任何一个棋子
                if not right_stop and col + i <= 8:
                    if not map_[row][col + i]:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row, col + i))
                    else:
                        right_stop = True
                        if map_[row][col + i].team != team:
                            # 如果当前位置有棋子，那么就判断是否能够吃掉它
                            put_down_chess_pos.append((row, col + i))

            # 一列
            up_stop = False
            down_stoop = False
            for i in range(1, 10):
                # 上边位置没有越界且没有遇到任何一个棋子
                if not up_stop and row - i >= 0:
                    if not map_[row - i][col]:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row - i, col))
                    else:
                        up_stop = True
                        if map_[row - i][col].team != team:
                            # 如果当前位置有棋子，那么就判断是否能够吃掉它
                            put_down_chess_pos.append((row - i, col))
                # 下边位置没有越界且没有遇到任何一个棋子
                if not down_stoop and row + i <= 9:
                    if not map_[row + i][col]:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row + i, col))
                    else:
                        down_stoop = True
                        if map_[row + i][col].team != team:
                            # 如果当前位置有棋子，那么就判断是否能够吃掉它
                            put_down_chess_pos.append((row + i, col))
        elif clicked_chess.name == "p":  # 炮
            # 一行
            direction_left_chess_num = 0
            direction_right_chess_num = 0
            for i in range(1, 9):
                # 计算当前行中，棋子左边与右边可以落子的位置
                # 左边位置没有越界
                if direction_left_chess_num >= 0 and col - i >= 0:
                    if not map_[row][col - i] and direction_left_chess_num == 0:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row, col - i))
                    elif map_[row][col - i]:
                        # 如果当前位置有棋子，那么就判断是否能够吃掉它
                        direction_left_chess_num += 1
                        if direction_left_chess_num == 2 and map_[row][col - i].team != team:
                            put_down_chess_pos.append((row, col - i))
                            direction_left_chess_num = -1  # 让其不能够在下次for循环时再次判断
                # 右边位置没有越界
                if direction_right_chess_num >= 0 and col + i <= 8:
                    if not map_[row][col + i] and direction_right_chess_num == 0:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row, col + i))
                    elif map_[row][col + i]:
                        # 如果当前位置有棋子，那么就判断是否能够吃掉它
                        direction_right_chess_num += 1
                        if direction_right_chess_num == 2 and map_[row][col + i].team != team:
                            put_down_chess_pos.append((row, col + i))
                            direction_right_chess_num = -1
            # 一列
            direction_up_chess_num = 0
            direction_down_chess_num = 0
            for i in range(1, 10):  # 这样就让i从1开始，而不是从0
                # 计算当前列中，棋子上边与下边可以落子的位置
                # 上边位置没有越界
                if direction_up_chess_num >= 0 and row - i >= 0:
                    if not map_[row - i][col] and direction_up_chess_num == 0:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row - i, col))
                    elif map_[row - i][col]:
                        # 如果当前位置有棋子，那么就判断是否能够吃掉它
                        direction_up_chess_num += 1
                        if direction_up_chess_num == 2 and map_[row - i][col].team != team:
                            put_down_chess_pos.append((row - i, col))
                            direction_up_chess_num = -1

                # 下边位置没有越界
                if direction_down_chess_num >= 0 and row + i <= 9:
                    if not map_[row + i][col] and direction_down_chess_num == 0:
                        # 如果没有棋子,则将当前位置组成一个元组，添加到列表
                        put_down_chess_pos.append((row + i, col))
                    elif map_[row + i][col]:
                        # 如果当前位置有棋子，那么就判断是否能够吃掉它
                        direction_down_chess_num += 1
                        if direction_down_chess_num == 2 and map_[row + i][col].team != team:
                            put_down_chess_pos.append((row + i, col))
                            direction_down_chess_num = -1

        # 剔除哪些被"将军"的位置
        put_down_chess_pos = self.judge_delete_position(put_down_chess_pos, clicked_chess)

        return put_down_chess_pos

    def judge_delete_position(self, all_position, clicked_chess):
        """
        删除被"将军"的位置
        """
        # 定义要删除的列表
        deleting_position = list()

        # 判断这些位置，是否会导致被"将军"，如果是则从列表中删除这个位置
        for row, col in all_position:
            # 1. 备份
            # 备份当前棋子位置
            old_row, old_col = clicked_chess.row, clicked_chess.col
            # 备份要落子的位置的棋子(如果没有，则为None)
            position_chess_backup = self.chessboard_map[row][col]
            # 2. 挪动位置
            # 移动位置
            self.chessboard_map[row][col] = self.chessboard_map[old_row][old_col]
            # 修改棋子的属性
            self.chessboard_map[row][col].update_position(row, col)
            # 清楚之前位置为None
            self.chessboard_map[old_row][old_col] = None
            # 3. 判断对方是否可以发起"将军"
            if self.judge_attack_general("b" if clicked_chess.team == "r" else "r"):
                deleting_position.append((row, col))
            # 4. 恢复到之前位置
            self.chessboard_map[old_row][old_col] = self.chessboard_map[row][col]
            self.chessboard_map[old_row][old_col].update_position(old_row, old_col)
            self.chessboard_map[row][col] = position_chess_backup

        # 5. 删除不能落子的位置
        all_position = list(set(all_position) - set(deleting_position))

        return all_position

    def move_chess(self, new_row, new_col):
        """
        将棋子移动到指定位置
        """
        # 得到要移动的棋子的位置
        old_row, old_col = ClickBox.singleton.row, ClickBox.singleton.col
        print("旧位置：", old_row, old_col, "新位置：", new_row, new_col)
        # 移动位置
        self.chessboard_map[new_row][new_col] = self.chessboard_map[old_row][old_col]
        # 修改棋子的属性
        self.chessboard_map[new_row][new_col].update_position(new_row, new_col)
        # 清楚之前位置为None
        self.chessboard_map[old_row][old_col] = None

    def get_general_position(self, general_player):
        """
        找到general_player标记的一方的将的位置
        """
        for row, line in enumerate(self.chessboard_map):
            for col, chess in enumerate(line):
                if chess and chess.team == general_player and chess.name == "j":
                    return chess.row, chess.col

    def judge_j_attack(self, attack_row, attack_col, general_row, general_col):
        """
        判断 两个将是否相对
        """
        if attack_col == general_col:
            # 在同一列
            min_row, max_row = (attack_row, general_row) if attack_row < general_row else (general_row, attack_row)

            chess_num = 0
            for i in range(min_row + 1, max_row):
                if self.chessboard_map[i][general_col]:
                    chess_num += 1
            if chess_num == 0:
                return True

    def judge_m_attack(self, attack_row, attack_col, general_row, general_col):
        """
        判断马是否攻击到"将"
        """
        if attack_row == general_row or attack_col == general_col:
            return False
        else:
            # "马走日"，利用这个特点会得出，如果此马能够攻击到"将"，那么两条边的平方和一定是5
            col_length = (attack_col - general_col) ** 2
            row_length = (attack_row - general_row) ** 2
            if col_length + row_length == 5:
                # 判断是否蹩马腿
                if col_length == 1:
                    if general_row < attack_row and not self.chessboard_map[attack_row - 1][attack_col]:
                        return True
                    elif general_row > attack_row and not self.chessboard_map[attack_row + 1][attack_col]:
                        return True
                elif col_length == 4:
                    if general_col < attack_col and not self.chessboard_map[attack_row][attack_col - 1]:
                        return True
                    elif general_col > attack_col and not self.chessboard_map[attack_row][attack_col + 1]:
                        return True

    def judge_c_and_p_attack(self, attack_chess_name, attack_row, attack_col, general_row, general_col):
        """
        判断"车"、"炮"能否攻击到对方"将"
        """
        check_chess_num = 1 if attack_chess_name == "p" else 0
        chess_num = 0
        if attack_row == general_row:
            # 在同一行
            min_col, max_col = (attack_col, general_col) if attack_col < general_col else (general_col, attack_col)
            for i in range(min_col + 1, max_col):
                if self.chessboard_map[attack_row][i]:
                    chess_num += 1
            if chess_num == check_chess_num:
                return True
        elif attack_col == general_col:
            # 在同一列
            min_row, max_row = (attack_row, general_row) if attack_row < general_row else (general_row, attack_row)
            for i in range(min_row + 1, max_row):
                if self.chessboard_map[i][general_col]:
                    chess_num += 1
            if chess_num == check_chess_num:
                return True

    @staticmethod
    def judge_z_attack(attack_team, attack_row, attack_col, general_row, general_col):
        """
        判断卒是否攻击到"将"
        """
        if attack_team == "r" and attack_row < general_row:
            return False
        elif attack_team == "b" and attack_row > general_row:
            return False
        elif (attack_row - general_row) ** 2 + (attack_col - general_col) ** 2 == 1:
            return True

    def judge_attack_general(self, attack_player):
        """
        判断 attact_player方是否 将对方的军
        """
        # 1. 找到对方"将"的位置
        general_player = "r" if attack_player == "b" else "b"
        general_position = self.get_general_position(general_player)

        # 2. 遍历我方所有的棋子
        for row, line in enumerate(self.chessboard_map):
            for col, chess in enumerate(line):
                if chess and chess.team == attack_player:
                    if chess.name == "z":  # 兵
                        # 传递5个参数（攻击方的标识，攻击方row，攻击方col，对方将row，对方将col）
                        if self.judge_z_attack(chess.team, chess.row, chess.col, *general_position):
                            return True
                    elif chess.name == "p":  # 炮
                        if self.judge_c_and_p_attack(chess.name, chess.row, chess.col, *general_position):
                            return True
                    elif chess.name == "c":  # 车
                        if self.judge_c_and_p_attack(chess.name, chess.row, chess.col, *general_position):
                            return True
                    elif chess.name == "m":  # 马
                        if self.judge_m_attack(chess.row, chess.col, *general_position):
                            return True
                    elif chess.name == "x":  # 象
                        pass
                    elif chess.name == "s":  # 士
                        pass
                    elif chess.name == "j":  # 将
                        if self.judge_j_attack(chess.row, chess.col, *general_position):
                            return True

    def judge_win(self, attack_player):
        """
        判断是否获胜
        """
        # 依次判断是否被攻击方的所有棋子，是否有阻挡攻击的可能
        for chess_line in self.chessboard_map:
            for chess in chess_line:
                if chess and chess.team != attack_player:
                    move_position_list = self.get_put_down_position(chess)
                    if move_position_list:  # 只要找到一个可以移动的位置，就表示没有失败，还是有机会的
                        return False
        return True
