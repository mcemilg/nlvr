import configparser
import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertModel


def build_bert(config_file, model_file=None):
    """Reads config, builds and returns the Bert model."""
    _config = configparser.ConfigParser()
    _config.read(config_file)
    config = _config["BERT"]

    model_config = BertConfig(int(config["vocab_size"]),
                              hidden_size=int(config["hidden_size"]),
                              num_hidden_layers=int(config["num_hidden_layers"]),
                              num_attention_heads=int(config["num_attention_heads"]),
                              intermediate_size=int(config["intermediate_size"]),
                              hidden_act=config["hidden_act"],
                              hidden_dropout_prob=float(config["hidden_dropout_prob"]),
                              attention_probs_dropout_prob=float(config["attention_probs_dropout_prob"]),
                              max_position_embeddings=int(config["max_position_embeddings"]),
                              type_vocab_size=int(config["type_vocab_size"]),
                              initializer_range=float(config["initializer_range"]))

    bert_model = BertModel(model_config)
    if model_file:
        state_dict = torch.load(model_file, map_location="cpu")
        bert_model.load_state_dict(state_dict)
    return bert_model
