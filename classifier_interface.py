# -*- coding:utf-8 -*-
# @last update time:2022/11/6 22:27
import torch
import torch.utils.data
import torchvision.datasets
from torch import nn, optim
from torchvision.transforms import transforms
from torchvision.models import AlexNet_Weights
import logging
# from typing import Callable, Any, Optional, Tuple
# bug:while typing model and torch model is torgether,it will cause ellipsis erro

# add current file path
import os
current_dir = os.path.split(os.path.realpath(__file__))[0]
self_data_path = current_dir + '/self_data'
train_path = self_data_path + '/train'
test_path = self_data_path + '/test'
predict_path = self_data_path + '/predict'
save_file_name = '/trained.pth'
load_file_name = save_file_name
save_path = current_dir + save_file_name
load_path = current_dir + load_file_name
alexnet = 'alexnet'
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
encoding_way = 'utf-8'


def create_self_data_dir(path=self_data_path):
    train = train_path
    test = test_path
    predict = predict_path

    os.makedirs(train, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    os.makedirs(predict, exist_ok=True)

create_self_data_dir(path=os.path.split(train_path)[0])

class ImageFolderAddPath(torchvision.datasets.ImageFolder):
    """
    add the 'root' filed on __init__() methond
    """

    # def __init__(self, root: str,
    #              transform: Optional[Callable] = None,
    #              target_transform: Optional[Callable] = None,
    #              loader: Callable[[str], Any] = ...,
    #              is_valid_file: Optional[Callable[[str], bool]] = None):
    #     super(ImageFolderAddPath, self).__init__(root, transform,
    #                                              target_transform,
    #                                              loader,
    #                                              is_valid_file)
    #     self.root = root

    """
    add the 'path' field  on __getitem__() method
    """

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class TrainModelPredict:
    def __init__(self,
                 train_path,
                 test_path,
                 predict_path,
                 save_path=None,
                 batch_size=1,
                 shuffle=True,
                 worker=1,
                 is_out=False,
                 device=device,
                 net_dict_path='net_dict.pth') -> None:
        self._is_out = is_out
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._worker = worker
        self._model = None
        self._save_path = save_path
        self._device = device
        self._log = Log().logger
        self._plt = PloyPic(is_darw_loss=True)
        self._loss_list = []
        self.criterion = nn.CrossEntropyLoss()
        self.multisteplr = None
        self.optimizer = None
        self.dict_save_path = net_dict_path

        self._train_data, self._index_classifier = self.load_data(train_path)
        self._test_data = self.load_data(test_path)[0]
        self._predict_data = self.load_data(predict_path)[0]

        self.init_model()

    def load_data(self, data_path, content: str = 'train data: '):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = ImageFolderAddPath(data_path, transform=transform)
        index_classifier = train_data.class_to_idx

        self.is_output(content + str(train_data.class_to_idx))

        train_data = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self._batch_size,
                                                 shuffle=self._shuffle,
                                                 num_workers=self._worker)

        return train_data, index_classifier

    def train_target_loss(self, 
                          loss_target=0.1, 
                          once_train_time=25,
                          is_draw_loss_ploy=True):
        train_time = 0
        self.train_once(train_time=once_train_time,
                        is_draw_loss_ploy=is_draw_loss_ploy)
        train_time += 1
        self._log.info('train time %s is over' % train_time)

        while self._loss_list[-1] > loss_target:
            self.train_once(train_time=once_train_time,
                            is_draw_loss_ploy=is_draw_loss_ploy)
            train_time += 1
            self._log.info('train time %s is over' % train_time)

    def train_target_accuracy(self, 
                              accuracy_target=0.8,
                              once_train_time=25, 
                              is_draw_loss_ploy=True):
        accuracy, trained_times = 0, 0
        while accuracy < accuracy_target:
            trained_times = self.train_once_test(once_train_time=once_train_time,
                                                is_draw_loss_ploy=is_draw_loss_ploy,
                                                trained_times=trained_times)

    def train_once_test(self, once_train_time, is_draw_loss_ploy, trained_times):
        self.train_once(train_time=once_train_time, 
                        is_draw_loss_ploy=is_draw_loss_ploy)
        trained_times += 1
        self.is_output('have trained %s' % trained_times)
        accuracy = self.test(model=self._model)
        self.is_output('%s trained times, accuracy is %s' %
                        (trained_times, accuracy))
        return trained_times

    def train_target_times_accuracy(self, 
                                   train_loop_time=25,
                                   once_train_time=25, 
                                   accuracy_target=0.8,
                                   is_draw_loss_ploy=True):
        accuracy, trained_times = 0, 0
        while accuracy < accuracy_target and trained_times < train_loop_time:
            trained_times = self.train_once_test(once_train_time=once_train_time,
                                is_draw_loss_ploy=is_draw_loss_ploy,
                                trained_times=trained_times)

    def train_once(self, train_time=25, is_draw_loss_ploy=True):
        for i in range(train_time):
            running_loss = 0.0
            for index, data in enumerate(self._train_data):
                img, labels, path = data
                img.to(self._device)
                labels.to(self._device)
                self.optimizer.zero_grad()
                output = self._model(img)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self.is_output('current is %s times,batch is %s,loss is %s'
                                % (i + 1, index, running_loss))

            if self.multisteplr is not None:
                self.multisteplr.step()

            self._loss_list.append(running_loss)
            if is_draw_loss_ploy:
                self._plt.draw_loss_plot(range(len(self._loss_list)), self._loss_list)

        self.is_output('train_finished!')

    def init_model(self):
        self._log.info('default is alexnet net')

        if self._model is None:
            weight = AlexNet_Weights.DEFAULT
            self._model = torchvision.models.alexnet(weights=weight)
            self._model.to(self._device)
            
        for params in self._model.parameters():
            params.requires_grad = False
        self._model.classifier = nn.Sequential(
            # redifine the classification layer,the default param is requires_grad=True
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, len(self._index_classifier)),
        )

        self.optimizer = optim.SGD(
            self._model.classifier.parameters(), lr=0.001, momentum=0.9)  # only update layers' param
        # self.multisteplr = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=[2, 3], gamma=0.65)  # lr decline strategy

    def load_pre_net_dict(self,path=None):
        if path is None:
            if self.dict_save_path is not None and os.path.exists(self.dict_save_path):
                net_dict = torch.load(self.dict_save_path)
            else:
                return
        if self._model is None:
            self.init_model()
        self._model.load_state_dict(net_dict)

    def save(self):
        if self._save_path is not None:
            torch.save(self._model.state_dict(), self._save_path)
            self._log.info('has saved the model net')

    def test(self, test_data=None, model=None):
        if model is None:
            model = self.load_module()

        classifier = {v: k for k, v in self._index_classifier.items()}

        correct, total = 0, 0
        if test_data is None:
            test_data_ = self._test_data
        else:
            test_data_ = test_data
        model.eval()
        with torch.no_grad():
            for one in test_data_:
                img, labels, path = one
                img.to(self._device)
                labels.to(self._device)
            
                output = model(img)

                _, predicted = torch.max(output.data, 1)

                self.is_output('%s file after test is %s,value is %s' %
                            (path, classifier[predicted.item()], output))
                total += labels.size(0)
                correct += (predicted == labels).sum()

        accuracy = (100 * correct / total)
        self.is_output('accuracy on test set : %s ' % accuracy)
        return accuracy

    def load_module(self):
        if self._model is None:
            if self._save_path is not None and os.path.exists(self._save_path):
                self._model = self._model.load_state_dict(torch.load(self._save_path))
                self._model.to(self._device)
            else:
                raise FileNotFoundError('not find saved net,no model is used!')
        return self._model

    def is_output(self, content: str, is_out=None):
        if is_out is None:
            is_out = self._is_out
        if is_out:
            self._log.info(content)

    def predict(self):
        if self._save_path is not None and os.path.exists(self._save_path):
            model = self.load_module()
        elif self._model is not None:
            model = self._model
        else:
            raise Exception('no model is found by save_path and model params')

        self.test(model=model, test_data=self._predict_data)

class PloyPic:
    def __init__(self, is_darw_loss=True) -> None:
        import matplotlib
        matplotlib.use('TKAgg')
        import matplotlib.pyplot as plt
        self._plt = plt
        self._fig, self._axs = plt.subplots()

        if is_darw_loss:
            self.interactive_mode(active=True)

    def interactive_mode(self, active=True):
        if active and not self._plt.isinteractive():
            self._plt.ion()
        elif active is False and self._plt.isinteractive():
            self._plt.ioff()
    
    def draw_loss_plot(self, x, y, title=None):
        self._axs.plot(x, y)
        if title is not None:
            self._fig.suptitle(title)
        self._plt.pause(1)

class Log:
    current_dir = current_dir
    def __init__(self, 
                 log_name=None, 
                 encoding=encoding_way, 
                 file_path='running.log',
                 clear_file_size=1024*1024*50,
                 format='[%(asctime)s %(name)s %(filename)s %(lineno)s-%(funcName)20s() %(levelname)s]:%(message)s') -> None:
        if log_name is None:
            log_name = os.path.split(os.path.abspath(__file__))[-1]
        self.file_path = os.path.join(self.current_dir, file_path)
        self.format = format
        logging.basicConfig(filename=self.file_path,
                            format=self.format,
                            encoding=encoding,
                            level=logging.NOTSET)
        self.logger = logging.getLogger(log_name)
        self.add_console_stream()

        self.clear_file_size = clear_file_size
        
        if self.clear_file_size is not None or self.clear_file_size != 0:
           self.check_size()
        

    def check_size(self):
        if os.path.exists(self.file_path):
             if os.path.getsize(self.file_path) >= self.clear_file_size:
                os.remove(self.file_path)

    def add_console_stream(self):
        self.console = logging.StreamHandler()
        self.console.setFormatter(logging.Formatter(self.format))
        self.logger.addHandler(self.console)


if __name__ == '__main__':
    model_str = alexnet
    ai_class = TrainModelPredict(train_path=train_path,
                                 test_path=test_path,
                                 predict_path=predict_path,
                                 save_path=save_path,
                                 is_out=True)
    ai_class.train_target_times_accuracy(train_loop_time=2, 
                                         once_train_time=5,
                                         accuracy_target=0.8,
                                         is_draw_loss_ploy=True)
    # ai_class.predict()
