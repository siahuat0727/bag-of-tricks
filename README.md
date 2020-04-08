# Bag of tricks

An easy-to-reuse code for [bag-of-tricks](https://arxiv.org/abs/1812.01187) in PyTorch.

## Reuse hints

Reuse the codes in `utils/trick` and follow `main.py` to modify your code.

For example,

```bash
$ git clone https://github.com/siahuat0727/bag-of-tricks
$ cp -r bag-of-tricks/utils path/to/your/repo
```

then slightly modify your code.

Just a simple reminder here. For detail, please have a look at the simple and clean demo code `main.py` instead of the hints below.

### 1. Cosine learning rate

Use `torch.optim.lr_scheduler.CosineAnnealingLR` and `.step()` every training batch.

### 2. Warmup learning rate

It's recommended to use `torch.optim.lr_scheduler` for the remaining lr schedule (after warmup), and pass your original `scheduler` as the parameter `after_scheduler` in `GradualWarmupScheduler`.
Remember to initialize your `optimizer` with the real initial lr (not the lr after warmup).

### 3. Label smoothing

`CrossEntropyLossMaybeSmooth` is inherited from `CrossEntropyLoss`.
There is usually no need to calculate the loss when evaluating.
If you need to do this, you can call `.train() ` to select the smooth version before training and call `.eval()` to select the non-smooth version before evaluating.

### 4. Mixup

Aside from the code, the only thing to mention is that the accuracy shown in the training step is usually lower than the actual training accuracy, since the input is a mixture of two images.

## Results - CIFAR10

**ResNet18**

cosine|warmup|label smoothing|mixup|accuracy (90 epochs)|accuracy (300 epochs)
--|--|--|--|--|--
| | | | |92.08|93.22
|✓| | | |92.54|93.10
| |✓| | |92.60|93.37
| | |✓| |92.74|93.50
| | | |✓|93.05|94.51
|✓|✓|✓|✓|**93.63**|**94.65**

To reproduce exactly the same result:

```bash
$ python main.py --gpu
$ python main.py --gpu --cosine_lr
$ python main.py --gpu --warmup 5
$ python main.py --gpu --smooth 0.1
$ python main.py --gpu --mixup 0.3
$ python main.py --gpu --warmup 5 --smooth 0.1 --mixup 0.3 --cosine_lr
```
