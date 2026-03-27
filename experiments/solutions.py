import torch
#####################################
# 1. Rotation-invariant classifier
#####################################
"""
Align training distribution with test distribution.
Generate augmented training data so our model is more robust to variations in orientation
"""



#####################################
# 2. Rotation Correction
#####################################
"""
Align test distribution with training distribution.
Undo rotations in test set by building a regression model to predict rotation angle of an image
and using the predicted angle to rotate the image back into its original orientation before feeding it
into the classifier
"""
#####################################
# 3. Test Time Augmentation
#####################################