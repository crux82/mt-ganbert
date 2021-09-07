from transformers import *
from module.san_model import SanModel
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlm": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
    "m": (BertConfig, BertModel, BertTokenizer),
    "musixmatch": (CamembertConfig, CamembertModel, CamembertTokenizer),
    "t5": (T5Config, T5EncoderModel, T5Tokenizer)
}
