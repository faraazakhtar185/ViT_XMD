# ViT_XMD
Vision Transformer on XMD dataset. Original paper can be found here: https://ieeexplore.ieee.org/document/7926666
The code is implemented in the same way the original paper does it—a ViT is trained on labels that have 17 columns, each one with a value of 1 or 0, depending on whether or not we see the property in the image. This trained transformer is further used to extract a feature vector right before classification. This feature vector is used to train 98 binary SVM classifiers to predict which of the 98 classes are present in each image. 
