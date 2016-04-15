################################################### You MIGHT need this import.
from net import Net

class Tester(object):
  def __init__(self, snapshot_path):
    # The original Girshick's code requires this field to exist.
    self.name = ''

    ###################################################### Your code goes here.
    # Load your network into, say, self.net.

  def forward(self, data, rois):
    """Performs a forward pass through the neural network.

    Arguments:
      data (ndarray): tensor containing the whole scenes (images)
      rois (ndarray): tensor containg ROIs; rois[:, 0] are indices of scenes 
                      in data, the rest are (left, top, bottom, right) 
                      coordinates

    Returns:
      output (dict): a dictionary with a single key 'cls_prob' holding
                     probability distributions produced by the network
    """

    ###################################################### Your code goes here.
    # You should have the following line:
    # output = {'cls_prob': net_output}.

    return output
