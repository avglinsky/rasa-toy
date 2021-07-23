import os
import typing
from typing import Any, Optional, Text, Dict, List, Type

import compress_fasttext

import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)

from pathlib import Path
from ruamel.yaml import YAML
import pymorphy2

from custom_components.stopwords import STOPWORDS_LIST

ruamel_yaml = YAML(typ='safe')

config_data = ruamel_yaml.load(Path('config.yml'))
domain_data = ruamel_yaml.load(Path('domain.yml'))
use_normal_word_form = config_data.get('use_normal_word_form', True)
delete_stopwords = config_data.get('delete_stopwords', False)

morph = pymorphy2.MorphAnalyzer()


if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata




class FastTextFeaturizer(DenseFeaturizer):
    """This component adds fasttext features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["compress_fasttext"]

    defaults = {"file": None, "cache_dir": None}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        if "cache_dir" not in component_config.keys():
            raise ValueError("You need to set `cache_dir` for the FasttextFeaturizer.")
        if "file" not in component_config.keys():
            raise ValueError("You need to set `file` for the FasttextFeaturizer.")
        path = os.path.join(component_config["cache_dir"], component_config["file"])

        if not os.path.exists(component_config["cache_dir"]):
            raise FileNotFoundError(
                f"It seems that the cache dir {component_config['cache_dir']} does not exists. Please check config."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"It seems that file {path} does not exists. Please check config."
            )

        self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_fasttext_features(example, attribute)

    def set_fasttext_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = message.get(TOKENS_NAMES[attribute])

        if not tokens:
            return None

        # We need to reshape here such that the shape is equivalent to that of sparsely
        # generated features. Without it, it'd be a 1D tensor. We need 2D (n_utterance, n_dim).

        if use_normal_word_form:
            modified = [morph.parse(token.text.lower())[0].normal_form for token in tokens[:-1]]
        else:
            modified = [token.text.lower() for token in tokens[:-1]]
        if delete_stopwords:
            modified = [token for token in modified if token not in STOPWORDS_LIST] or modified
        word_vectors = np.array([self.model[t.text.lower()] for t in tokens])
        text_vector = np.sum(word_vectors, axis=0).reshape(1, -1)


        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_fasttext_features(message)

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