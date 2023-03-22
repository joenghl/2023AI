"""
第3次作业, 请不要修改输入输出的数据类型和格式
"""


def ResolutionProp(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    return


def MGU(f1, f2):
    """
    :param f1: str
    :param f2: str
    :return: dict
    """
    return


def ResolutionFOL(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    return


if __name__ == '__main__':
    # 测试程序
    KB1 = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    result1 = ResolutionProp(KB1)
    for r in result1:
        print(r)

    print(MGU('P(xx,a)', 'P(b,yy)'))
    print(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))

    KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    result2 = ResolutionFOL(KB2)
    for r in result2:
        print(r)
