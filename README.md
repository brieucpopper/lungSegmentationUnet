# Lung Segmentation with U-Net
*Started Nov 2023 — team project @ Telecom SudParis, with Theo Danielou and Thibault Kiewsky*

![unet](https://github.com/brieucpopper/lungSegmentationUnet/assets/102361078/6c83ff85-a6b7-4f09-8528-e205dafd101f)

U-Net implementation in TensorFlow/Keras for lung segmentation on chest X-rays, including data pre-processing, loss tracking, and training on a GPU cluster over SSH.

Both binary and 3-class (left/right lung) segmentation.

## Results

![example](https://github.com/brieucpopper/lungSegmentationUnet/blob/main/IMAGE_3.png)
Input image, ground truth, predicted mask.

3-class example (left vs right lung):
![3class](https://github.com/brieucpopper/lungSegmentationUnet/blob/main/IMAGE_37.png)

## Notes

Python files are in the repo root but could use reorganizing. Weights and training data are not included.
