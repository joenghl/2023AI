import pygame


class ClickBox(pygame.sprite.Sprite):
    """
    标记类
    """
    singleton = None

    def __new__(cls, *args, **kwargs):
        """通过重写此方法，实现单例"""
        if cls.singleton is None:
            cls.singleton = super().__new__(cls)
        return cls.singleton

    def __init__(self, screen, row, col):
        super().__init__()
        self.screen = screen
        self.image = pygame.image.load("images/r_box.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (50 + col * 57, 50 + row * 57)
        self.row = row
        self.col = col

    @classmethod
    def show(cls):
        if cls.singleton:
            cls.singleton.screen.blit(cls.singleton.image, cls.singleton.rect)

    @classmethod
    def clean(cls):
        """
        清理上次的对象
        """
        cls.singleton = None
