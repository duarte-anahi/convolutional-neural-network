# Convolutional model: design, training and diagnosis

**English** · [Español](README.es.md)

A convolutional neural network built with Keras over an image dataset of 8 classes. The point of the project is not the accuracy but the **diagnosis**: why the model does not generalize, and why shrinking it did not help.

![Training curves: both versions](docs/comparison.png)

## Architecture

| Block | Filters | Output |
|---|---|---|
| 1 | 32 | 100×100×32 |
| 2 | 64 | 50×50×64 |
| 3 | 128 | 25×25×128 |
| 4 | 128 | 12×12×128 |

Four `Conv2D` + `MaxPooling2D` blocks, then `Flatten`, a 128-unit dense layer and an 8-way softmax: **~2.6M parameters**. Trained with Adam and `sparse_categorical_crossentropy` for 15 epochs.

**Data:** 1,567 images resized to 200×200, split 80/20 (1,254 training, 313 validation).

## The experiment

A second version keeps **32 filters in every block** — ~619K parameters, 75% fewer — to test whether the overfitting came from excess capacity.

| Model | Parameters | Train accuracy | Validation accuracy | Validation loss |
|---|---|---|---|---|
| V1 · growing filters | ~2.6M | 0.992 | 0.597 | 2.63 |
| V2 · constant filters | ~619K | 0.994 | 0.585 | 2.57 |

## Diagnosis

Both versions memorize the training set (~99%) while validation accuracy stalls near 59%, and validation loss starts climbing at epoch 6 while training loss keeps falling — the textbook shape of overfitting.

Cutting 75% of the parameters changed nothing, so the bottleneck is the **dataset** (size and variety), not the architecture.

**What this project taught me:** when the validation loss rises while the training loss falls, the answer is not another layer — it is more and better data.

**Next steps:** data augmentation and transfer learning from a pretrained model.

## Running it

The notebook is written for **Google Colab**. The dataset lives in Google Drive; section 0 of the notebook explains how to add it to your own Drive before running.

## Tech stack

Python · TensorFlow · Keras · NumPy · Matplotlib
