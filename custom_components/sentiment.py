import os
import re
import pandas as pd
import compress_fasttext
import numpy as np
import typing
import json
from pathlib import Path
from ruamel.yaml import YAML
from typing import Any, Optional, Text, Dict, List, Type
import pymorphy2
import pickle
import logging
morph = pymorphy2.MorphAnalyzer()

from sklearn.linear_model import LogisticRegression

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES
from rasa.nlu.training_data.message import Message
from rasa.nlu.training_data.training_data import TrainingData
from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES, TOKENS_NAMES
from conf.stopwords import STOPWORDS_LIST
from custom_components.custom_fasttext import FastTextFeaturizer
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
FEATURIZER_CLASS_ALIAS = "alias"
FEATURE_TYPE_SENTENCE = "sentence"
FEATURE_TYPE_SEQUENCE = "sequence"

ruamel_yaml = YAML(typ='safe')

config_data = ruamel_yaml.load(Path('config.yml'))
use_normal_word_form = config_data.get('use_normal_word_form', True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class SentimentClassifier(Component):

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [FastTextFeaturizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        if "cache_dir" not in component_config.keys():
            raise ValueError("You need to set `cache_dir` for the SentimentAnalyzer.")
        if "file" not in component_config.keys():
            raise ValueError("You need to set `file` for the SentimentAnalyzer.")
        path = os.path.join(component_config["cache_dir"], component_config["file"])
        if not os.path.exists(component_config["cache_dir"]):
            raise FileNotFoundError(
                f"It seems that the cache dir {component_config['cache_dir']} does not exists. Please check config."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"It seems that file {path} does not exists. Please check config."
            )
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        self.sent_map = {'positive': 'позитивное', 'very positive': 'очень позитивное', 'neutral': 'нейтральное',
                    'negative': 'негативное', 'very negative': 'очень негативное'}


    def process(self, message: Message, **kwargs: Any) -> None:
        dense_matrix = message.get("text_dense_features")
        cls_vector = dense_matrix[-1].reshape(1, -1)
        prediction = self.sent_map[str(self.model.predict(cls_vector)[0])]
        entities = message.get("entities", [])
        entities.append({
            "entity": "sentimental",
            "start": 0,
            "end": 1,
            "value": f"{prediction}",
            "extractor": "SentimentClassifier"
        })
        message.set("entities", entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component

        return cls(meta)
