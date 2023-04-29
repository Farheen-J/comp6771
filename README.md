# comp6771 - Image Deblurring

## <u>Dataset</u>

* Dataset is downloaded by Google Colab using link: https://www.kaggle.com/kwentar/blur-dataset
* To address the issue of spatially variant blurring caused by motion and defocus, we did not use the original blurred images provided in the dataset. Instead, we applied Gaussian blurring to the images using the `blurring.py` script located in the `scripts` folder. This enabled us to generate new blurred images that were more consistent and suitable for use in the deblurring process.

## <u>Files in script folder</u>

* Execute jupyter notebook [comp6771.ipynb](comp6771.ipynb)
* Below are the scripts executed by the notebook, in the following order:
  * [blurring.py](scripts%2Fblurring.py)
  * [deblurring.py](scripts%2Fdeblurring.py)
  * [test.py](scripts%2Ftest.py)
