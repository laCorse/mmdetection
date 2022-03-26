def demo_1():
    class A:
        def __init__(self):
            self.a = [1, 2, 3]

        def __iter__(self):
            for a in self.a:
                yield a

    cls_a = A()
    for a in cls_a:
        print(a)  # 1 2 3


def demo_2():
    from collections.abc import Iterable, Iterator

    class A:
        def __init__(self):
            self.index = -1
            self.a = [1, 2, 3]

        # 必须要返回一个实现了 __next__ 方法的对象，否则后面无法 for 遍历
        # 因为本类自身实现了 __next__，所以通常都是返回 self 对象即可
        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            if self.index < len(self.a):
                return self.a[self.index]
            else:
                # 抛异常，for 内部会自动捕获，表示迭代完成
                raise StopIteration("遍历完了")

    cls_a = A()
    print(isinstance(cls_a, Iterable))  # True
    print(isinstance(cls_a, Iterator))  # True
    print(isinstance(iter(cls_a), Iterator))  # True

    for a in cls_a:
        print(a)  # 1 2 3


def demo_3():
    import functools

    class IterDataPipe:
        def __init__(self, data, fn):
            self.data = data
            self.fn = fn

        def __iter__(self):
            return self.fn(iter(self.data))

    def map(data, fn):
        for d in data:
            yield fn(d)

    def filter(data, fn):
        for d in data:
            condition = fn(d)
            if condition:
                yield d

    def batch(data, batch_size):
        buf = []
        for d in data:
            buf.append(d)
            if len(buf) >= batch_size:
                yield buf
                buf = []

    dp = IterDataPipe(range(10), functools.partial(map, fn=lambda x: x + 1))
    dp = IterDataPipe(dp, functools.partial(filter, fn=lambda x: x % 2))
    dp = IterDataPipe(dp, functools.partial(batch, batch_size=2))
    print(list(dp))  # [[1, 3], [5, 7]]


def demo_4():
    class IterDataPipe:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            for d in self.data:
                yield d

    class MapDataPipe:
        def __init__(self, data, fn):
            self.data = data
            self.fn = fn

        def __iter__(self):
            for d in self.data:
                yield self.fn(d)

    class FilterDataPipe:
        def __init__(self, data, fn):
            self.data = data
            self.fn = fn

        def __iter__(self):
            for d in self.data:
                condition = self.fn(d)
                if condition:
                    yield d

    class BatchDataPipe:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size

        def __iter__(self):
            buf = []
            for d in self.data:
                buf.append(d)
                if len(buf) >= self.batch_size:
                    yield buf
                    buf = []

    dp = IterDataPipe(range(10))
    dp = MapDataPipe(dp, lambda x: x + 1)
    dp = FilterDataPipe(dp, lambda x: x % 2)
    dp = BatchDataPipe(dp, batch_size=2)
    print(list(dp))  # [[1, 3], [5, 7]]


def demo_5():
    import time

    def calc_time(fn):
        def warpper(*args, **kwargs):
            time_start = time.time()
            res = fn(*args, **kwargs)
            cost_time = time.time() - time_start
            print(" %s运行时间：%0.3s 秒" % (fn.__name__, cost_time))
            return res

        return warpper

    @calc_time
    def fn():
        time.sleep(2.3)

    fn()


def demo_6():
    import time

    class CalcTime:
        def __init__(self):
             pass

        def __call__(self, fn):
            def calc_time(*args, **kwargs):
                time_start = time.time()
                res = fn(*args, **kwargs)
                cost_time = time.time() - time_start
                print(" %s运行时间：%0.3s 秒" % (fn.__name__, cost_time))
                return res
            return calc_time

    @CalcTime()
    def fn():
        time.sleep(2.3)

    fn()


if __name__ == '__main__':
    demo_num = 6
    eval(f'demo_{demo_num}()')
