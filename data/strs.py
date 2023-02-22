class TaskStrs:
    LESION_DETECTION = 'lesion-detection'
    NEGBIO_CLASSIFICATION = "negbio-classification"
    CHEXPERT_CLASSIFICATION = "chexpert-classification"
    FIXATION_GENERATION = "fixation-generation"

class SourceStrs:
    XRAYS = "xrays"
    CLINICAL = "clinical"


class FusionStrs:
    ElEMENTWISE_SUM = "element-wise sum"
    HADAMARD_PRODUCT = "hadamard product"
    NO_ACTION = "no-action"
    CONCAT = "concat"