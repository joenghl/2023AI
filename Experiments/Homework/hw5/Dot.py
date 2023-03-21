import pygame


class Dot(object):
    group = list()  # 这个类属性用来存储所有的“可落子对象”的引用

    def __init__(self, screen, row, col):
        """初始化"""
        self.image = pygame.image.load("images/dot2.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (60 + col * 57, 60 + row * 57)
        self.screen = screen
        self.row = row
        self.col = col

    def show(self):
        """显示一颗棋子"""
        self.screen.blit(self.image, self.rect.topleft)

    @classmethod
    def create_nums_dot(cls, screen, pos_list):
        """批量创建多个对象"""
        for temp in pos_list:
            cls.group.append(cls(screen, *temp))

    @classmethod
    def clean_last_position(cls):
        """
        清除所有可以落子对象
        """
        cls.group.clear()

    @classmethod
    def show_all(cls):
        for temp in cls.group:
            temp.show()

    @classmethod
    def click(cls):
        """
        点击棋子
        """
        for dot in cls.group:
            if pygame.mouse.get_pressed()[0] and dot.rect.collidepoint(pygame.mouse.get_pos()):
                print("被点击了「可落子」对象")
                return dot
