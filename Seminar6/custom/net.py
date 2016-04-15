################################################# You MIGHT need these imports.
import cPickle

from fast_rcnn.config import cfg

class Net(object):
  """A class for holding a symbolic representation of the neural network.
  Instances of this class are going to be used both in the solver and 
  in the tester. 
  """

  def __init__(self, snapshot_path=None):
    """Constructs a symbolic graph for a neural network.

    Arguments:
      snapshot_path (str): path to the pretrained network
    """
    pass

  def save(self, filename):
    """Saves model weights."""
    pass

  @property
  def input(self):
    """Returns symbolic inputs of the model."""
    pass

  @property
  def prediction(self):
    """Returns symbolic variable containing the model predictions."""
    pass

  @property
  def params(self):
    """Returns shared variables containing the model weights."""
    pass

  @property
  def param_values(self):
    """Returns a list of the model weights (values)."""
    pass
