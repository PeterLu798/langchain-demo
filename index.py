class Test:

    def __or__(self, other):
        print(f"我右边有一个东西，它是：{other}")

    def __ror__(self, other):
        print(f"我左边有一个东西，它是：{other}")


if __name__ == '__main__':
    test = Test()

    test | 18000
    15000 | test
# 运行结果：
# 我右边有一个东西，它是：18000
# 我左边有一个东西，它是：15000
