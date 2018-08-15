from collections import OrderedDict
import numpy as np


class GreyAugment:
    """
    Greyscale value augmentation.
    Randomly adjust contrast/brightness, and apply random gamma correction.
    """

    def __init__(
            self,
            mode='3D',
            skip_ratio=0.3,
            CONTRAST_FACTOR=0.3,
            BRIGHTNESS_FACTOR=0.3):
        """
        Initialize parameters.
        Args:
            mode: 2D, 3D, or mix
            skip_ratio:
        """
        assert mode=='3D' or mode=='2D' or mode=='mix'
        self.mode  = mode
        self.ratio = skip_ratio
        self.CONTRAST_FACTOR   = CONTRAST_FACTOR
        self.BRIGHTNESS_FACTOR = BRIGHTNESS_FACTOR

    def prepare(self, spec, **kwargs):
        return dict(spec)

    def augment(self, sample, **kwargs):
        #print '\n[GreyAugment]'  # DEBUG
        ret = sample
        if np.random.rand() > self.ratio:
            if self.mode == 'mix':
                mode = '3D' if np.random.rand() > 0.5 else '2D'
            else:
                mode = self.mode
            ret = eval('self.augment{}(sample, **kwargs)'.format(mode))
        return ret

    def augment2D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        #print '2D greyscale augmentation'  # DEBUG

        # Greyscale augmentation.
        imgs = kwargs['imgs']
        for key in imgs:
            for z in xrange(sample[key].shape[-3]):
                img = sample[key][...,z,:,:]
                img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
                img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
                img = np.clip(img, 0, 1)
                img **= 2.0**(np.random.rand()*2 - 1)
                sample[key][...,z,:,:] = img

        return sample

    def augment3D(self, sample, **kwargs):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        #print '3D greyscale augmentation'  # DEBUG

        # Greyscale augmentation.
        imgs = kwargs['imgs']
        for key in imgs:
            sample[key] *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            sample[key] += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            sample[key] = np.clip(sample[key], 0, 1)
            sample[key] **= 2.0**(np.random.rand()*2 - 1)

        return sample

