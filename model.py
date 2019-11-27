"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

class Model(nn.Module):
    def __init__(self, num_class, base_model='resnet18'):
        super(Model, self).__init__()

        print(("""
Initializing model:
    base model:         {}.
    num_class:          {}.
        """.format(base_model, num_class)))

        self._prepare_base_model(base_model)
        self._prepare_recognizer(num_class)

    def _prepare_base_model(self, base_model):

        self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
        self._input_size = 224

    def _prepare_recognizer(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

    def forward(self, input):
        ''' Execute one CNN on the frames
        Args:
            input (Tensor): frames. size [batch_size, c, h, w]

        Returns:
            base_out (Tensor) : [batch_size, num_class]
        '''
        base_out = self.base_model(input)
        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        scales = [1, .875, .75, .66]
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=False)])
