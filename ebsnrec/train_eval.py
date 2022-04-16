import logging
import os.path

from tqdm.auto import tqdm

from ebsnrec.dataset_creator import DatasetCreator
from ebsnrec.dataset_builder import DatasetBuilder
from ebsnrec.generator_builder import GeneratorBuilder
from ebsnrec.model_factory import ModelFactory
from ebsnrec.util import check_and_mkdir, set_log, parse_spec_cols

tqdm.pandas()


def evaluation_once(common_config: dict, creator_config: dict, feature_config=None, model_config: dict = None,
                    evaluation_config: dict = None, log_config:dict=None):

    # base info

    dataset_id = creator_config.get("dataset_id", "meetup_sg")
    model_id = model_config.get("model_id", "DNN")

    # make dirs

    data_root_dir = common_config.get("data_root_dir", ".\\data")
    raw_dataset_dir = common_config.get("raw_dataset_dir", "C:\\dataset\\meetup")
    dataset_dir = os.path.join(data_root_dir, dataset_id, "dataset")
    processed_data_dir = os.path.join(data_root_dir, dataset_id, model_id, "processed_data")
    model_dir = os.path.join(data_root_dir, dataset_id, model_id, "model")
    log_data_dir = os.path.join(data_root_dir, dataset_id, model_id, "log")
    for dir_name in [data_root_dir, dataset_dir, processed_data_dir, model_dir, log_data_dir]:
        check_and_mkdir(dir_name)

    parse_spec_cols(feature_config["spec_cols"])

    set_log(**log_config, log_data_dir=log_data_dir)

    # create dataset from raw_dataset

    raw_dataset_file = os.path.join(raw_dataset_dir, creator_config["dataset_file_name"])
    dataset_creator = DatasetCreator(raw_dataset_file=raw_dataset_file, sava_dir=dataset_dir, **creator_config)
    dataset_creator.create_dataset()

    # create processed_dataset from dataset

    data_builder = DatasetBuilder(data_dir=dataset_dir, save_dir=processed_data_dir, **feature_config)
    data_builder.build_data()

    # create train and eval generator

    generator_builder = GeneratorBuilder(data_dir=processed_data_dir, spec_cols=feature_config["spec_cols"], **evaluation_config)
    train_gen = generator_builder.get_generator(stage="train")
    valid_gen = generator_builder.get_generator(stage="valid")

    # get model
    model_config.update(evaluation_config)
    model_config["verbose"] = common_config["verbose"]

    model_factory = ModelFactory(data_dir=processed_data_dir,model_root=model_dir,spec_cols=feature_config["spec_cols"], **model_config)
    model = model_factory.get_model()
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        epochs=evaluation_config["epochs"],
                        verbose=common_config["verbose"])

    model.load_weights(model.checkpoint)
    model.evaluate_generator(valid_gen)

    logging.info('***** validation results *****')
    test_gen = generator_builder.get_generator(stage="test")
    model.evaluate_generator(test_gen)