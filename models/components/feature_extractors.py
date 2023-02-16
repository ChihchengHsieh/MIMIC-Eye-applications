# The extractors here, take into different kinds of modalities, then ouput a 3-D tensor for task perfromer.
# Those 3d tensors from different extractors can then be fused together 

from torch import nn

class GeenralFeatureExtractor(nn.Module):
    def __init__(self, name) -> None:
        self.name = name
        super().__init__()

class ImageFeatureExtractor(GeenralFeatureExtractor):
    '''
    Extracting features maps from a given image, x.
    '''
    def __init__(self, backbone) -> None:
        super().__init__("extractor-image")
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x['images'])

class ClinicalFeatureExtractor(GeenralFeatureExtractor):
    def __init__(self) -> None:
        super().__init__("extractor-tabular")

    def forward(self, x):
        pass


class SequentialFeatureExtractor(GeenralFeatureExtractor):
    def __init__(self, encoder_type="", **kwargs) -> None:
        super().__init__("extractor-sequential")
        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    