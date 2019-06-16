from algo.common.input import observation_input
from algo.common.tf_util import adjust_shape

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):

        self.name = name

    def get(self):

        raise NotImplementedError

    def make_feed_dict(self, data):
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):

        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt


