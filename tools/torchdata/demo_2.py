def demo_1():
    # 函数式编程
    from torch.utils.data.datapipes.iter import IterableWrapper

    # map
    dp = IterableWrapper(range(4)).map((lambda x: x * 2))
    print(list(dp))  # [0, 2, 4, 6]

    dp = IterableWrapper(range(4))\
        .map((lambda x: (x, x)))\
        .map((lambda x: x * 2))
    print(list(dp))  # [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)]

    dp = IterableWrapper(range(4)) \
        .map((lambda x: (x, x))) \
        .map((lambda x: x * 2), input_col=1)
    print(list(dp))  # [(0, 0), (1, 2), (2, 4), (3, 6)]

    dp = IterableWrapper(range(4)) \
        .map((lambda x: (x, x))) \
        .map((lambda x: x * 2), input_col=1, output_col=0)
    print(list(dp))  # [(0, 0), (2, 1), (4, 2), (6, 3)]


if __name__ == '__main__':
    demo_num = 1
    eval(f'demo_{demo_num}()')
