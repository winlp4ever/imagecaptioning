### Quick Set-up

**System:** `Ubuntu 18.04`
(Tested and work on Windows 10!)

To begin with, we need to install a Python3 environment. We recommend `anaconda`. If you already have `anaconda`, you can follow the instructions below:

1. Firstly, create a virtual env with command:

```python
conda create -n nic python=3.6 pillow numpy nltk tensorboardx
```

Activate this env with `conda activate nic`.

Proceed to the [PyTorch website]() to download `pytorch` and `torchvision`, or you can just type the following command:

* If you have GPU(s) and `CUDA` totally operational on your machine:


```python
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
(replace `9.0` by your version of `cuda`, all options are `8.0`, `9.0` and `10.0`)

* Otherwise, consider to install `PyTorch` for `cpu`:


```python
conda install pytorch-cpu torchvision-cpu -c pytorch
```

2. Next step, download our pretrained model checkpoint (for the network to load pretrained weights) and vocab with command `python download_ckpt.py`.

Now you're good to go. You can run the following command to check the auto caption

```python
python play_with.py --fn test.jpg
```

Change `test.jpg` to any other image filename that you put in `images` dir (If you want to test an image, it must be located in `images` dir)


### Retrain from scratch

If you want to retrain from scratch, you have to first download the whole [`COCO`](http://cocodataset.org/#download) dataset (version 2014: 3 files `2014 Train images [83K/13GB]`, `2014 Val images [41K/6GB]` and `2014 Train/Val annotations [241MB]`) and extract it in the `data` dir. Your `data` dir must be of the following structure at the end:

```
data/
  |--annotations/
    |--(all json captions files)
  |--train2014/
    |--(all train images)
  |--val2014/
    |--(all val images)
```

You can download the datasets manually, or use the script we have written with command `python download_dataset.py`. Remember, the dataset files are quite big (20 Gb in total), so be patient, the download will take some time to finish.

After that, you are all good. Just type the command `python train.py`. If you have troubles, try reduce the batch-size (default 128) to 64 or smaller number with flag `--batch-size`, for ex, `python train.py --batch-size 64`

(Consider using a machine with GPU to train, otherwise it will take very long time to converge)

Run the evaluation with `bleu` score as evaluation metric with command `python eval.py` (default BLEU-1)

Test your model with `python play_with.py` like told before.
