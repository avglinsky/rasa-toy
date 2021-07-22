import os
import re
import json
from pathlib import Path
from ruamel.yaml import YAML
import glob
import logging
import datetime
from collections import Counter
import typing
from typing import Any, Optional, Text, Dict, List, Type

from custom_components.stopwords import STOPWORDS_LIST

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import compress_fasttext
import numpy as np
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


import rasa.core.utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component, UnsupportedLanguageError
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.model import Metadata
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.tokenizers.tokenizer import Tokenizer, Token
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES,
    FEATURIZER_CLASS_ALIAS,
    NO_LENGTH_RESTRICTION,
    NUMBER_OF_SUB_TOKENS,
    TOKENS_NAMES,
    LANGUAGE_MODEL_DOCS,
)
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
)
from rasa.utils import train_utils

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

ruamel_yaml = YAML(typ='safe')

config_data = ruamel_yaml.load(Path('config.yml'))
domain_data = ruamel_yaml.load(Path('domain.yml'))
use_normal_word_form = config_data.get('use_normal_word_form', True)
delete_stopwords = config_data.get('delete_stopwords', False)



class FastTextFeaturizer(DenseFeaturizer):
    """Добавление fasttext эмбеддингов."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["compress_fasttext"]


    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"file": None, "cache_dir": None}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

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

    @staticmethod
    def _combine_with_existing_dense_features(
            message: Message,
            additional_features: Any,
            feature_name: Text,
    ) -> Any:
        if message.get(feature_name) is not None:

            if len(message.get(feature_name)) != len(additional_features):
                raise ValueError(
                    f"Cannot concatenate dense features as sequence dimension does not "
                    f"match: {len(message.get(feature_name))} != "
                    f"{len(additional_features)}. Message: '{message.text}'."
                )

            return np.concatenate(
                (message.get(feature_name), additional_features), axis=-1
            )
        else:
            return additional_features

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        
        '''

        intent_2_ft_vectors = {}
        for intent in training_data.intents:
            intent_examples = [ex for ex in training_data.training_examples if ex.get('intent') == intent]
            cls_vector = np.zeros((1, 300))
            for example in intent_examples:
                tokens = example.get('tokens')
                if use_normal_word_form:
                    modified = [morph.parse(token.text.lower())[0].normal_form for token in tokens[:-1]]
                else:
                    modified = [token.text.lower() for token in tokens[:-1]]
                if delete_stopwords:
                    modified = [token for token in modified if token not in STOPWORDS_LIST] or modified
                cls_vector += np.sum([self.model[token] for token in modified], axis=0).reshape(1, -1)
            cls_vector = cls_vector / len(intent_examples)
            intent_2_ft_vectors[intent] = cls_vector.tolist()


        with open(os.path.join(".", "generated_files", "intent_2_ft_vectors.json"), "w", encoding="utf-8") as file:
            json.dump(intent_2_ft_vectors, file, ensure_ascii=False)



        # generating intents files dict

        # if os.path.exists(os.path.join('conf', 'resource', 'additional_vocab_full.md')):
        #     additional_vocab_path = os.path.join('conf', 'resource', 'additional_vocab_full.md')
        # else:
        #     additional_vocab_path = os.path.join('conf', 'resource', 'additional_vocab.md')

        intents_folder_path = os.path.join('data', 'nlu')
        intents_paths = glob.glob(os.path.join(intents_folder_path, '*.md'))
        # additional_vocab_path = additional_vocab_path
        # intents_paths.append(additional_vocab_path)
        exception_intents = []

        intents_paths = [intent_path for intent_path in intents_paths if intent_path not in exception_intents]

        intent_vocab = []
        for intent_file_path in intents_paths:
            words = []
            with open(intent_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('-'):
                        words.extend(
                            [word.lower() for word in re.findall(r'[А-Яа-я-]+', line[2:])]
                        )
            intent_vocab.extend(words)

        with open(f'generated_files{os.sep}intent_vocab_data.json', 'w', encoding='utf-8') as intent_vocab_data_file:
            intent_vocab_data = {
                'intent_vocab': list(set(intent_vocab)),
                'intent_word_2_freq': Counter(intent_vocab)
            }
            json.dump(intent_vocab_data, intent_vocab_data_file, ensure_ascii=False)


        intent_2_bow = {}
        intent_2_training_phrases = {}
        for intent in training_data.intents:
            intent_examples = [ex for ex in training_data.training_examples if ex.get('intent') == intent]
            bow = []
            tmp_list = []
            for example in intent_examples:
                tokens = example.get('tokens')
                tokens = [token.text.lower() for token in tokens]
                tmp_list.append(" ".join(tokens[1:-1]))
                noun_adj_tokens = []

                for token in tokens[1:-1]:
                    token = "".join([symbol for symbol in token if symbol not in "!\"#$%&\'()*+,;<=>?@[\\]^`{|}_~"])
                    if token:
                        parsed = morph.parse(token)[0]
                        if 'LATN' in parsed.tag:
                            noun_adj_tokens.append(token)
                        elif (parsed.normal_form not in STOPWORDS_LIST and token not in STOPWORDS_LIST):
                            if ((bool(re.fullmatch('[а-яА-Я-]+', token)) and
                                 (parsed.tag.POS in ['NOUN', 'ADJF', 'ADJS', 'INTJ', 'VERB', 'INFN', 'PRTF', 'PRTS']
                                  or parsed.methods_stack[0][0].__class__ == pymorphy2.units.unkn.UnknAnalyzer))
                                or (token.isdigit() and 3 <= len(token) <= 6 and (
                                            int(token) < datetime.datetime.now().year - 5 or
                                            int(token) > datetime.datetime.now().year))) \
                                    and not "Geox" in parsed.tag:
                                noun_adj_tokens.append(parsed.normal_form)
                    else:
                        continue

                bow.extend(noun_adj_tokens)
            intent_2_training_phrases[intent] = tmp_list
            counter = {}
            for el in bow:
                if el in counter.keys():
                    counter[el] += 1
                else:
                    counter[el] = 1
            additional_counter = {}
            counter = dict(sorted(counter.items(), key=lambda item: item[1]))

            for key in counter.keys():
                coef = counter[key] / len(intent_examples)
                counter[key] = coef

                if '-' in key:
                    subwords = key.split('-')
                    subwords.append(key.replace('-', ''))
                    for subword in subwords:
                        if subword not in counter.keys():
                            additional_counter[subword] = coef


            for word in ['заявка', 'заявление', 'запрос', 'ходатайство', 'просьба', 'прошение', 'петиция', 'требование', 'заказ']:
                if word in counter.keys():
                    additional_counter.update({'заявка': 0.25,
                                    'заявление': 0.25,
                                    'запрос': 0.25,
                                    'ходатайство': 0.25,
                                    'просьба': 0.25,
                                    'прошение': 0.25,
                                    'петиция': 0.25,
                                    'требование': 0.25,
                                    'заказ': 0.25})

            counter.update(additional_counter)


            intent_2_bow[intent] = counter

        with open(os.path.join(".", "generated_files", "intent_2_bow.json"), "w", encoding="utf-8") as file:
            json.dump(intent_2_bow, file, ensure_ascii=False)

        with open(os.path.join(".", "generated_files", "intent_2_training_phrases.json"), "w", encoding="utf-8") as file:
            json.dump(intent_2_training_phrases, file, ensure_ascii=False)

        organizations = set()
        for key in intent_2_bow.keys():
            if len(key) > 3 and key[:3] in 'кор див орг':
                organizations.add(key.split('_')[0])
        faqs = list(domain_data.get('faq_answers', {}).keys())
        excluded = [f'{corp_token}_Приветствие']
        faqs_excluded = faqs + excluded
        org_2_scenarios = {}


        all_orgs = []
        for key in intent_2_bow.keys():
            org = key.split('_')[0]
            if org in organizations and key not in faqs_excluded:
                if org in org_2_scenarios.keys():
                    org_2_scenarios[org].append(key)
                else:
                    org_2_scenarios[org] = [key]
        for key in org_2_scenarios.keys():
            org_2_scenarios[key] = sorted(org_2_scenarios[key])

        with open(os.path.join(".", "generated_files", "org_2_scenarios.json"), "w", encoding="utf-8") as file:
            json.dump(org_2_scenarios, file, ensure_ascii=False)


        scenario_start_action = set()
        pattern = re.compile("(?<=- )\w+")
        stories_paths = glob.glob(os.path.join(".", "data", "core", "*.md"))
        for story_path in stories_paths:
            with open(story_path, 'r', encoding='utf-8') as file:
                content = file.read().split("##")
                content = [el for el in content if el]
                for el in content:
                    if "*" in el:
                        el = el.split("*")[1]
                        action = pattern.search(el).group()
                        if action:
                            scenario_start_action.add(action)

        with open(os.path.join(".", "generated_files", "scenario_start_action.py"), "w", encoding="utf-8") as file:
            file.write(f"SCENARIO_START_ACTION = {list(scenario_start_action)}")
            
        '''


        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_fasttext_features(example, attribute)


    def set_fasttext_features(self, message: Message, attribute: Text = TEXT) -> None:

        tokens = message.get(TOKENS_NAMES[attribute])

        methods = dir(message)
        token0_methods = dir(tokens[0])
        token_last_methods = dir(tokens[-1])
        token_text = tokens[0].text
        token_last_text = tokens[-1].text

        if not tokens:
            return None

        # We need to reshape here such that the shape is equivalent to that of sparsely
        # generated features. Without it, it'd be a 1D tensor. We need 2D (n_utterance, n_dim).
        if use_normal_word_form:
            modified = [morph.parse(token.text.lower())[0].normal_form for token in tokens]
        else:
            modified = [token.text.lower() for token in tokens]
        if delete_stopwords:
            modified = [token for token in modified if token not in STOPWORDS_LIST] or modified
        cls_vector = np.sum([self.model[token] for token in modified], axis=0).reshape(1, -1)
        word_vectors = np.array([self.model[t.text.lower()] for t in tokens])


        features = np.concatenate([word_vectors, cls_vector])

        features = self._combine_with_existing_dense_features(
            message, additional_features=features, feature_name=DENSE_FEATURIZABLE_ATTRIBUTES[attribute]
        )
        message.set(DENSE_FEATURIZABLE_ATTRIBUTES[attribute], features)


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