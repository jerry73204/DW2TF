# DW2TF: Darknet Weights to TensorFlow

This is a simple convector which converts Darknet weights file (`.weights`) to Tensorflow weights file (`.ckpt`).

The repository is a fork from [jinyu121/DW2TF](https://github.com/jinyu121/DW2TF). It tries to fix several compatibility issues with Yolov3.

## Usage

```
python3 main.py \
    --cfg data/yolov3.cfg \
    --weights data/yolov3.weights \
    --output data \
    --prefix yolov3_ \
    --gpu 0
```

## Todo

- More layer types

## Thanks

- [darkflow](https://github.com/thtrieu/darkflow)
