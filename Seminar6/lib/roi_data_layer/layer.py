# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Adapted for Theano usage by Yaroslav Ganin
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self):
        self.top = []

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self):
        """Setup the RoIDataLayer."""

        top = self.top

        self._num_classes = 21

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top.append(np.zeros((cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE), dtype=np.single))
        self._name_to_top_map['data'] = idx
        idx += 1

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top.append(np.zeros((1, 5), dtype=np.single))
        self._name_to_top_map['rois'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top.append(np.zeros((1,), dtype=np.single))
        self._name_to_top_map['labels'] = idx
        idx += 1

        if cfg.TRAIN.BBOX_REG:
            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top.append(np.zeros((1, self._num_classes * 4), dtype=np.single))
            self._name_to_top_map['bbox_targets'] = idx
            idx += 1

            # bbox_inside_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top.append(np.zeros((1, self._num_classes * 4), dtype=np.single))
            self._name_to_top_map['bbox_inside_weights'] = idx
            idx += 1

            top.append(np.zeros((1, self._num_classes * 4), dtype=np.single))
            self._name_to_top_map['bbox_outside_weights'] = idx
            idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        top = self.top

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].resize(blob.shape)
            # Copy data into net's input blobs
            top[top_ind][...] = blob.astype(np.float32, copy=False)
