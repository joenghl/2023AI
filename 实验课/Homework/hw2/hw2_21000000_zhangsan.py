class StuData:
    """
    本次作业中, 类方法的输入参数名可自定义, 但参数数据类型需保证测试程序正常运行
    请不要更改类方法的名
    """
    def __init__(self):
        self.data = []

    def AddData(self):
        pass

    def SortData(self):
        pass

    def ExportFile(self):
        pass


if __name__ == '__main__':
    # 测试程序
    s1 = StuData('student_data.txt')
    s1.AddData(name="Bob", stu_num="003", gender="M", age=20)
    s1.SortData('age')
    s1.ExportFile('new_stu_data.txt')