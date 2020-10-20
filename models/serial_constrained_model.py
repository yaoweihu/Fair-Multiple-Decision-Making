from models.separate_constrained_model import SeparateConstrainedModel


class SerialConstrainedModel(SeparateConstrainedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)