def BinarySearch(nums, target):
    """
    :param nums: list[int]
    :param target: int
    :return: int
    """
    return

def MatrixAdd(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    return

def MatrixMul(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    return

def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    return


if __name__ == "__main__":
    print("输出", BinarySearch([-1, 0, 3, 5, 9, 12], 9), "答案", 4)
    print("输出", MatrixAdd([[1,0],[0,1]], [[1,2],[3,4]]), "答案", [[2, 2], [3, 5]])
    print("输出", MatrixMul([[1,0],[0,1]], [[1,2],[3,4]]), "答案", [[1, 2], [3, 4]])
    print("输出", ReverseKeyValue({'Alice':'001', 'Bob':'002'}), "答案", {'001':'Alice', '002':'Bob'})