from ebsnrec.dataset_creator import DatasetCreator
from tqdm.auto import tqdm
import logging
tqdm.pandas()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s'
                        )
    dataset_creator = DatasetCreator("C:\\dataset\\meetup\\meetup_sg.db",'.\\data',tz_offset=8,begin_time='2017-01-01',end_time='2018-11-01')
    dataset_creator.create_dataset()


