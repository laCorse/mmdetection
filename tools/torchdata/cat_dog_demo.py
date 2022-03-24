import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, _ForkerIterDataPipe


# torchdata 中有实现，直接 copy 过来即可
@functional_datapipe("unzip")
class UnZipperIterDataPipe(IterDataPipe):
    def __new__(
            cls,
            source_datapipe,
            sequence_length: int,
            buffer_size: int = 1000
    ):
        instance_ids = list(range(sequence_length))
        # The implementation basically uses Forker but only yields a specific element within the sequence
        container = _UnZipperIterDataPipe(source_datapipe, sequence_length, buffer_size)  # type: ignore[arg-type]
        return [_ChildDataPipe(container, i) for i in instance_ids]


class _UnZipperIterDataPipe(_ForkerIterDataPipe):
    def get_next_element_by_instance(self, instance_id: int):
        for return_val in super().get_next_element_by_instance(instance_id):
            yield return_val[instance_id]


data_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(10),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def demo_dataset():
    class CatDogImageDataset(Dataset):
        def __init__(self, annotations_file, transform=None, repeat=2):
            self.img_urls, self.labels = self._parse_csv(annotations_file)
            # 考虑到数据太少了，所以 repeat 倍数
            self.img_urls = self.img_urls * repeat
            self.labels = self.labels * repeat
            self.transform = transform

        def _parse_csv(self, annotations_file):
            with open(annotations_file, 'r') as f:
                datas = f.readlines()[2:]
            img_urls = []
            labels = []
            for data in datas:
                img_url, label_id = data.strip('\n').split(',')
                img_urls.append(img_url)
                labels.append(int(label_id))
            return img_urls, labels

        def __len__(self):
            return len(self.img_urls)

        def __getitem__(self, idx):
            img_url = self.img_urls[idx]
            # 考虑到如果真的走网络，实在是太慢了，大概率程序会挂，所以依然替换为本地
            local_path = img_url.replace('https://github.com/pytorch/data/blob/main/examples/vision/fakedata/', '')
            image = read_image(local_path)

            label = torch.tensor(self.labels[idx])

            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = CatDogImageDataset("cat_dog.csv", transform=data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for images, targets in train_dataloader:
        print(images.shape, targets)


def demo_datapipe():

    def build_data_pipe():
        class CSVParsePipe(IterDataPipe):
            def __init__(self, annotations_file):
                self.annotations_file = annotations_file

            def __iter__(self):
                with open(self.annotations_file, 'r') as f:
                    for line in f.readlines():
                        yield line

        # 1. 数据前处理
        dp = CSVParsePipe("cat_dog.csv") \
            .filter(lambda x: x.find('https') >= 0) \
            .map(lambda x: x.strip('\n').split(','))

        # 2.重复 2 遍
        dp = dp.concat(dp)

        # 3.分离为两条 pipe (如果不想分也是可以的，这里是为了引入 unzip 知识点)
        img_dp, label_dp = dp.unzip(sequence_length=2)

        # 4. 对 img_dp 和 label 单独处理
        img_dp = img_dp.map(
            lambda x: x.replace('https://github.com/pytorch/data/blob/main/examples/vision/fakedata/', '')) \
            .map(lambda x: read_image(x)) \
            .map(lambda x: data_transform(x))
        label_dp = label_dp.map(lambda x: (torch.tensor(int(x))))

        # 5. 合并，并打乱顺序返回
        data_pipe = img_dp.zip(label_dp).shuffle()
        # 目前 batch 没有多进程功能，所以 batch 操作依然由 dataloader 实现
        return data_pipe

    data_pipe = build_data_pipe()
    train_dataloader = DataLoader(data_pipe, batch_size=2, shuffle=False,
                                  worker_init_fn=torch.utils.data.backward_compatibility.worker_init_fn)
    for images, targets in train_dataloader:
        print(images.shape, targets)


if __name__ == '__main__':
    demo_dataset()
    print('===========================')
    demo_datapipe()
