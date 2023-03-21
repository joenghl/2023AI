import pygame
from ChessBoard import *


class BackChess(pygame.sprite.Sprite):
    def __init__(self, screen):
        self.screen = screen
        self.image = pygame.image.load("images/back.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (615, 280)
        self.history_map = list()

    def show(self):
        self.screen.blit(self.image, self.rect)

    def clicked_back(self, chessboard: ChessBoard, event):
        if event.type == pygame.MOUSEBUTTONUP and self.rect.collidepoint(event.pos):
            if len(self.history_map) <= 2:
                return False
            self.pop_history()
            self.pop_history()
            res = self.pop_history()
            chessboard.set_chessboard_str_map(res)
            self.add_history(res)
            return True

    def add_history(self, str_map):
        self.history_map.append(str_map)

    def pop_history(self):
        return self.history_map.pop()


class Game(object):
    """
    游戏类
    """

    def __init__(self, screen, chessboard):
        self.screen = screen
        self.player = "r"  # 默认走棋的为红方r
        self.player_tips_r_image = pygame.image.load("images/red.png")
        self.player_tips_r_image_topleft = (550, 500)
        self.player_tips_b_image = pygame.image.load("images/black.png")
        self.player_tips_b_image_topleft = (550, 100)
        self.show_attack = False
        self.attack_player = 'r'
        self.show_attack_count = 0
        self.show_attack_time = 100
        self.attack_img = pygame.image.load("images/pk.png")
        self.show_win = False
        self.win_img = pygame.image.load("images/win.png")
        self.win_player = None
        self.show_win_count = 0
        self.show_win_time = 300
        self.chessboard = chessboard

        self.AI_mode = True
        self.user_team = 'r'
        self.computer_team = 'b'
        self.back_button = BackChess(screen)

    def get_player(self):
        """
        获取当前走棋方
        """
        return self.player

    def exchange(self):
        """
        交换走棋方
        """
        self.player = "r" if self.player == "b" else "b"
        return self.get_player()

    def reset_game(self):
        """重置游戏"""
        # 所谓的重置游戏，就是将棋盘恢复到默认，走棋方默认的红方
        # 重建新的默认棋子
        self.chessboard.create_chess()
        self.back_button.history_map = list()
        self.back_button.add_history(self.chessboard.get_chessboard_str_map())
        # 设置走棋方为红方
        self.player = 'r'

    def show(self):
        # 如果一方获胜，那么显示"赢"
        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_win:
            self.show_win_count += 1
            if self.show_win_count == self.show_win_time:
                self.show_win_count = 0
                self.show_win = False
                self.reset_game()  # 游戏玩过一局之后，重置游戏

        if self.show_win:
            if self.win_player == "b":
                self.screen.blit(self.win_img, (550, 100))
            else:
                self.screen.blit(self.win_img, (550, 450))
            return

        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_attack:
            self.show_attack_count += 1
            if self.show_attack_count == self.show_attack_time:
                self.show_attack_count = 0
                self.show_attack = False

        if self.player == "r":
            self.screen.blit(self.player_tips_r_image, self.player_tips_r_image_topleft)
        else:
            self.screen.blit(self.player_tips_b_image, self.player_tips_b_image_topleft)

        if self.attack_player == "r":
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 400))
        else:
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 100))

        self.back_button.show()

    def set_attack(self):
        """
        标记"将军"效果
        """
        self.show_attack = True
        self.attack_player = self.player

    def set_win(self, win_player):
        """
        设置获胜方
        """
        self.show_win = True
        self.win_player = win_player
