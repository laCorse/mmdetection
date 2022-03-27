def demo_1():
    class Test:
        def __init__(self):
            self.value = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        def __getattr__(self, item):
            return self.value[item]

    test = Test()
    print(test.b)  # 1


if __name__ == '__main__':
    demo_num = 1
    eval(f'demo_{demo_num}()')
