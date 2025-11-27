import os

from ...utils import import_modules


TEXT_VISION_CONNECTOR_FACTORY = {}

def TextVisionConnectorFactory(text_vision_connector_name):
    model = None
    for name in TEXT_VISION_CONNECTOR_FACTORY.keys():
        if name.lower() in text_vision_connector_name.lower():
            model = TEXT_VISION_CONNECTOR_FACTORY[name]
    assert model, f"{text_vision_connector_name} is not registered"
    return model


def register_text_vision_connector(name):
    def register_text_vision_connector_cls(cls):
        if name in TEXT_VISION_CONNECTOR_FACTORY:
            return TEXT_VISION_CONNECTOR_FACTORY[name]
        TEXT_VISION_CONNECTOR_FACTORY[name] = cls
        return cls
    return register_text_vision_connector_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.text_vision_connector")
