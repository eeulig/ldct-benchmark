from enum import Enum


class Methods(Enum):
    """Enum of available Methods.

    Can be used as `method` argument in [ldctbench.hub.load_model.load_model][]. Alternatively, their string representations may be used.

    Attributes
    ----------
    CNN10: str
        Has string representation `"cnn10"`
    REDCNN: str
        Has string representation `"redcnn"`
    WGANVGG: str
        Has string representation `"wganvgg"`
    RESNET: str
        Has string representation `"resnet"`
    QAE: str
        Has string representation `"qae"`
    DUGAN: str
        Has string representation `"dugan"`
    TRANSCT: str
        Has string representation `"transct"`
    BILATERAL: str
        Has string representation `"bilateral"`
    """

    CNN10 = "cnn10"
    REDCNN = "redcnn"
    WGANVGG = "wganvgg"
    RESNET = "resnet"
    QAE = "qae"
    DUGAN = "dugan"
    TRANSCT = "transct"
    BILATERAL = "bilateral"
