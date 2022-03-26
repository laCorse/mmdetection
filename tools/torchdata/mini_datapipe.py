import functools


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError


class IterDataPipe(IterableDataset):
    functions = {}

    def __getattr__(self, attribute_name):
        function = functools.partial(IterDataPipe.functions[attribute_name], self)
        return function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        cls.functions[function_name] = function


# 方式一: 官方实现方式
# class functional_datapipe:
#     def __init__(self, name) -> None:
#         self.name = name
#
#     def __call__(self, cls):
#         IterDataPipe.register_datapipe_as_function(self.name, cls)
#         return cls

# 方式二：更简单的装饰器函数实现方式
def functional_datapipe(name):
    def warpper(cls):
        IterDataPipe.register_datapipe_as_function(name, cls)

    return warpper


@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe):
    def __init__(self, dp, fn):
        super().__init__()
        self.dp = dp
        self.fn = fn

    def __iter__(self):
        for d in self.dp:
            yield self.fn(d)


if __name__ == '__main__':
    class ExampleIterPipe(IterDataPipe):
        def __init__(self, range=10):
            self.range = range

        def __iter__(self):
            for i in range(self.range):
                yield i


    dp = ExampleIterPipe(4).map((lambda x: x * 2))
    print(list(dp))
