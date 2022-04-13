import glob
import logging
import os.path

from fuxictr.pytorch.data_generator import DataGenerator


class GeneratorBuilder(object):
    def __init__(self, data_dir:str="", batch_size=1024,test_batch_size=2048,**kwargs):
        self.train_data = os.path.join(data_dir,"train.h5")
        self.valid_data = os.path.join(data_dir, "valid.h5")
        self.test_data = os.path.join(data_dir, "test.h5")
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

    def get_generator(self, stage="train"):
        batch_size = self.batch_size
        shuffle = True
        if stage=="train":
            blocks = glob.glob(self.train_data)
        elif stage == "valid":
            blocks = glob.glob(self.valid_data)
        else:
            blocks = glob.glob(self.test_data)
            batch_size = self.test_batch_size
            shuffle = False
        gen = DataGenerator(blocks, batch_size=batch_size, shuffle=shuffle)
        logging.info("samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                     .format(gen.num_samples, gen.num_positives, gen.num_negatives,
                             100. * gen.num_positives / gen.num_samples, gen.num_blocks))
        return gen


