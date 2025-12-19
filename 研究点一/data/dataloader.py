from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from data.pavia import NewPaviaDataset, NewPaviaCaveDataset
from data.paviac import NewPaviaCDataset, NewPaviaCCaveDataset
from data.base import MinibatchSampler
from data.grss2013 import NewGRSS2013Dataset
from data.salinas import NewSalinasDataset, NewSalinasCaveDataset
from data.indian import NewIndianDataset, NewIndianCaveDataset
from data.washington import NewWashingtonDataset, NewWashingtonCaveDataset
from data.houston18 import NewHouston18Dataset, NewHouston18CaveDataset
from data.mucad import NewMucadDataset, NewMucadCaveDataset
from data.agucha import NewAguchaDataset, NewAguchaCaveDataset
from data.cuprite import NewCupriteDataset, NewCupriteCaveDataset
from data.hanchuan import NewHanChuanDataset, NewHanChuanCaveDataset
from data.honghu import NewHongHuDataset, NewHongHuCaveDataset
from data.longkou import NewLongKouDataset, NewLongKouCaveDataset
from data.botswana import NewBotswanaDataset, NewBotswanaCaveDataset
from data.metriclearning import MetricLearningWHLKDataset


@registry.DATALOADER.register('NewBotswanaLoader')
class NewBotswanaLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewBotswanaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewBotswanaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewBotswanaCaveLoader')
class NewBotswanaCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewBotswanaCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewBotswanaCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('MetricLearningWHLKLoader')
class MetricLearningWHLKLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = MetricLearningWHLKDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(MetricLearningWHLKLoader, self).__init__(dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewLongKouLoader')
class NewLongKouLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewLongKouDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewLongKouLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewLongKouCaveLoader')
class NewLongKouCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewLongKouCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewLongKouCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))
@registry.DATALOADER.register('NewHongHuLoader')
class NewHongHuLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHongHuDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHongHuLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewHongHuCaveLoader')
class NewHongHuCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHongHuCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHongHuCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))
@registry.DATALOADER.register('NewHouston18Loader')
class NewHouston18Loader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHouston18Dataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHouston18Loader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewHouston18CaveLoader')
class NewHouston18CaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHouston18CaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHouston18CaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewWashingtonLoader')
class NewWashingtonLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewWashingtonDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewWashingtonLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewWashingtonCaveLoader')
class NewWashingtonCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewWashingtonCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewWashingtonCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewPaviaCLoader')
class NewPaviaCLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaCDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewPaviaCLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewPaviaCCaveLoader')
class NewPaviaCCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaCCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewPaviaCCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0, 
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewPaviaLoader')
class NewPaviaLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewPaviaCaveLoader')
class NewPaviaCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewPaviaCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0, 
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))


@registry.DATALOADER.register('NewHanChuanLoader')
class NewHanChuanLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHanChuanDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHanChuanLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewHanCHuanCaveLoader')
class NewHanCHuanCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHanChuanCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewHanCHuanCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0, 
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewMucadLoader')
class NewMucadLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewMucadDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewMucadLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewMucadRGBLoader')
class NewMucadRGBLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewMucadCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewMucadRGBLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0, 
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewIndianLoader')
class NewIndianLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewIndianDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewIndianLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewIndianCaveLoader')
class NewIndianCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewIndianCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewIndianCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewAguchaLoader')
class NewAguchaLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewAguchaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewAguchaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewAguchaCaveLoader')
class NewAguchaCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewAguchaCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewAguchaCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewcupriteLoader')
class NewcupriteLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewCupriteDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewcupriteLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewcupriteCaveLoader')
class NewcupriteCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewCupriteCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'], self.config['compactness'])
        sampler = MinibatchSampler(dataset)
        super(NewcupriteCaveLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))


@registry.DATALOADER.register('NewSalinasLoader')
class NewSalinasLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'], self.config['n_segments'])
        sampler = MinibatchSampler(dataset)
        super(NewSalinasLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))

@registry.DATALOADER.register('NewSalinasCaveLoader')
class NewSalinasCaveLoader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasCaveDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.num_train_samples_per_class, self.sub_minibatch, np_seed, self.config['slic'])
        sampler = MinibatchSampler(dataset)
        super(NewSalinasCaveLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=200,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))


@registry.DATALOADER.register('NewGRSS2013Loader')
class NewGRSS2013Loader(DataLoader):
    def __init__(self, config, np_seed):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewGRSS2013Dataset(self.image_path, self.gt_path, self.training, self.sub_minibatch, np_seed)
        sampler = MinibatchSampler(dataset)
        super(NewGRSS2013Loader, self).__init__(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                sampler=sampler,
                                                batch_sampler=None,
                                                num_workers=self.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                timeout=0,
                                                worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_path='',
            gt_path='',
            training=True,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=200
        ))
