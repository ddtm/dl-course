################################################# You MIGHT need these imports.
from fast_rcnn.config import cfg
from net import Net
from roi_data_layer.layer import RoIDataLayer

class Solver(object):
  def __init__(self):
    # Holds current iteration number. 
    self.iter = 0

    # How frequently we should print the training info.
    self.display_freq = 1

    # Holds the path prefix for snapshots.
    self.snapshot_prefix = 'snapshot'

    ###################################################### Your code goes here.

  # This might be a useful static method to have.
  @staticmethod
  def build_step_fn(net):
    """Takes a symbolic network and compiles a function for weights updates."""
    pass

  def get_training_batch(self):
    """Uses ROIDataLayer to fetch a training batch.

    Returns:
      input_data (ndarray): input data suitable for R-CNN processing
      labels (ndarray): batch labels (of type int32)
    """

    ###################################################### Your code goes here.

    return input_data, labels

  def step(self):
    """Conducts a single step of SGD."""
    
    ###################################################### Your code goes here.
    # Among other things, assign the current loss value to self.loss.

    self.iter += 1
    if self.iter % self.display_freq == 0:
      print 'Iteration {:<5} Train loss: {}'.format(self.iter, self.loss)

  def save(self, filename):
    """Saves model weights."""
    pass
