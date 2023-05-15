class TaskStrs:
    LESION_DETECTION = 'lesion-detection'
    NEGBIO_CLASSIFICATION = "negbio-classification"
    CHEXPERT_CLASSIFICATION = "chexpert-classification"
    FIXATION_GENERATION = "fixation-generation"
    
    AGE_REGRESSION = "age-regression"
    TEMPERATURE_REGRESSION = "temperature-regression"
    HEARTRATE_REGRESSION = "heartrate-regression"
    RESPRATE_REGRESSION = "resprate-regression"
    O2SAT_REGRESSION = "o2sat-regression"
    SBP_REGRESSION = "sbp-regression"
    DBP_REGRESSION = "dpb-regression"
    ACUITY_REGRESSION = "acuity-regression"
    GENDER_CLASSIFICATION = "gender-classification"

class SourceStrs:
    XRAYS = "xrays"
    CLINICAL = "clinical"
    FIXATIONS = "fixations"


class FusionStrs:
    ElEMENTWISE_SUM = "element-wise sum"
    HADAMARD_PRODUCT = "hadamard product"
    NO_ACTION = "no-action"
    CONCAT = "concat"
    CONCAT_WITH_TOKENMIXER = "concat_with_token_mixer"
    CONCAT_WITH_NORM_ACT_TOKENMIXER = "concat_with_norma_act_token_mixer"