"""
第4次作业, 选择其中一种算法实现即可.
Puzzle问题的输入数据类型为二维嵌套list, 空位置用 `0`表示. 输出的解数据类型为 `list`, 是移动数字方块的次序.
"""

def A_star(puzzle):
    pass


def IDA_star(puzzle):
    pass


if __name__ == '__main__':
    # 可自己创建更多用例并分析算法性能
    puzzle1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    puzzle2 = [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]]
    sol1 = A_star(puzzle1)
    sol2 = A_star(puzzle2)
    print(sol1)
    print(sol2)
