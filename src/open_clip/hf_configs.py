# HF architecture dict:
arch_dict = {
  # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
  "roberta": {
    "config_names": {
      "context_length": "max_position_embeddings",
      "vocab_size": "vocab_size",
      "width": "hidden_size",
      "heads": "num_attention_heads",
      "layers": "num_hidden_layers",
    },
    "pooler": "mean_pooler",
  },
  # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
  "xlm-roberta": {
    "config_names": {
      "context_length": "max_position_embeddings",
      "vocab_size": "vocab_size",
      "width": "hidden_size",
      "heads": "num_attention_heads",
      "layers": "num_hidden_layers",
    },
    "pooler": "mean_pooler",
  },
  # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
  "mt5": {
    "config_names": {
        "context_length": "relative_attention_max_distance", # very unsure of this, couldn't find anything better
        "vocab_size": "vocab_size",
        "width": "d_model",
        "heads": "num_heads",
        "layers": "num_layers",
    },
    "pooler": "mean_pooler",
  },
}
