This directory holds (*after you download them*):
- Symlinks to datasets

In order to train and test with PASCAL VOC, you will need to establish symlinks.
From the `data` directory (`cd data`):

```
# For VOC 2007
ln -s /your/path/to/VOC2007/VOCdevkit VOCdevkit2007

# For VOC 2012
ln -s /your/path/to/VOC2012/VOCdevkit VOCdevkit2012
```

```
# data/cache holds various outputs created by the datasets package
ln -s /data/fast_rcnn_shared/cache

# move the selective search data to a shared location and symlink to them
# (only applicable to Fast R-CNN training)
ln -s /data/fast_rcnn_shared/selective_search_data

ln -s /data/VOC2007/VOCdevkit VOCdevkit2007
ln -s /data/VOC2012/VOCdevkit VOCdevkit2012
```
