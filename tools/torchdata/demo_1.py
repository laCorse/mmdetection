def demo_1():
    from torch.utils.data import IterDataPipe

    # Example IterDataPipe
    class ExampleIterPipe(IterDataPipe):
        def __init__(self, range=10):
            self.range = range

        def __iter__(self):
            for i in range(self.range):
                yield i

    dp = ExampleIterPipe(4)
    print(list(dp))  # [0, 1, 2, 3]


def demo_2():
    # 可以考虑直接用 wrapper,就不需要新建一个类了
    from torch.utils.data.datapipes.iter import IterableWrapper
    dp = IterableWrapper(range(4))
    print(list(dp))  # [0, 1, 2, 3]


def demo_3():
    # 函数式编程
    from torch.utils.data.datapipes.iter import IterableWrapper

    # map
    dp = IterableWrapper(range(4)).map((lambda x: x * 2))
    print(list(dp))  # [0, 2, 4, 6]

    # filter
    dp = IterableWrapper(range(4)).filter(lambda x: x % 2 == 0)
    print(list(dp))  # [0, 2]

    # shuffle, 默认 shuffle 尺寸是 10000
    dp = IterableWrapper(range(4)).shuffle()
    print(list(dp))  # [1, 0, 3, 2]

    # groupby
    dp = IterableWrapper(range(4)).groupby(lambda x: x % 2)
    print(list(dp))  # [[0, 2], [1, 3]]

    dp = IterableWrapper(range(4)).shuffle().groupby(lambda x: x % 2)
    print(list(dp))  # [[1, 3], [2, 0]]


def demo_4():
    from torch.utils.data.datapipes.iter import IterableWrapper
    # zip
    _dp = IterableWrapper(range(4)).shuffle()
    # 两个迭代器打包
    dp = IterableWrapper(range(4)).zip(_dp)
    print(list(dp))  # [(0, 2), (1, 1), (2, 3), (3, 0)]

    # 如果长度不一致，取最小的集合返回
    _dp = IterableWrapper(range(3)).shuffle()
    # 两个迭代器打包
    dp = IterableWrapper(range(4)).zip(_dp)
    print(list(dp))  # [(0, 2), (1, 1), (2, 0)]

    # fork 复制 n 份
    dp1, dp2, dp3 = IterableWrapper(range(4)).fork(3)
    print(list(dp1))  # [0, 1, 2, 3]
    print(list(dp2))  # [0, 1, 2, 3]
    print(list(dp1+dp2+dp3))  # [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    # demux 切分 Splits the input DataPipe into multiple child DataPipes
    dp1, dp2 = IterableWrapper(range(5)).shuffle().demux(2, lambda x: x % 2)
    print(list(dp1))  # [0, 4, 2]
    print(list(dp2))  # [3, 1]

    # mux 逐一合并
    dp1 = IterableWrapper(range(4))
    dp2 = IterableWrapper(range(3)).map(lambda x: x * 10)
    dp3 = IterableWrapper(range(2)).map(lambda x: x * 100)
    dp = dp1.mux(dp2, dp3)
    print(list(dp))  # [0, 0, 0, 1, 10, 100, 2, 20, 3]

    # concat 按照顺序拼接，要区别 mux
    dp = IterableWrapper(range(4))
    dp1 = IterableWrapper(range(3))
    dp = dp.concat(dp1)
    print(list(dp))  # [0, 1, 2, 3, 0, 1, 2]

    # sharding_filter 没理解
    # routed_decode 解码二进制流


def demo_5():
    from torch.utils.data.datapipes.iter import IterableWrapper
    # batch
    dp = IterableWrapper(range(5)).batch(2)
    print(list(dp))  # [[0, 1], [2, 3], [4]]

    dp = IterableWrapper(range(5)).batch(batch_size=2, drop_last=True)
    print(list(dp))  # [[0, 1], [2, 3]]

    # 打包后，再次打包
    dp = IterableWrapper(range(5)).batch(2).batch(2)
    print(list(dp))  # [[[0, 1], [2, 3]], [[4]]]

    # unbatch
    dp = IterableWrapper(range(5)).batch(2).shuffle().unbatch()
    print(list(dp))  # [4, 2, 3, 0, 1]

    # collate，默认就是转 tensor
    dp = IterableWrapper(range(5)).batch(2).collate()
    print(list(dp))  # [tensor([0, 1]), tensor([2, 3]), tensor([4])]


if __name__ == '__main__':
    demo_num = 5
    eval(f'demo_{demo_num}()')
