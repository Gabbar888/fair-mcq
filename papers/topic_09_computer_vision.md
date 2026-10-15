# Mock Practice — Topic 9: Computer Vision Fundamentals
## 20 questions | 33 minutes | MCQ, single correct

Choose the single best option for each question. OpenCV refers to the `cv2` Python bindings, version 4.x.

---

**Q1.** A video-analytics service stores raw 1920 × 1080 RGB frames as 8-bit unsigned integers. For inference, frames are resized to 224 × 224, converted to `float32`, and stacked into batches of 32 in channels-first layout.

Which option gives the memory for one raw frame and for one inference batch, ignoring any framework overhead?

A. About 2.1 million bytes per frame and 6.4 million per batch, one per pixel.

B. About 6.2 million bytes per frame and 4.8 million per batch, one per value.

C. About 6.2 million bytes per frame and about 19.3 million bytes per batch.

D. About 24.9 million bytes per frame and 19.3 million per batch, four per value.

---

**Q2.** A preprocessing script loads an image whose top-left pixel is stored on disk as pure colour values B = 200, G = 100, R = 50.

```python
import cv2

img = cv2.imread("frame_0001.png")          # top-left pixel as described
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
print(rgb[0, 0], gray[0, 0])
```

OpenCV's grayscale conversion uses 0.299·R + 0.587·G + 0.114·B. Which option matches the printed output?

A. `[ 50 100 200]` and `124`, the value the second conversion produces.

B. `[ 50 100 200]` and `96`, the luminance of the original pixel.

C. `[200 100  50]` and `96`, as imread already returns RGB order.

D. `[200 100  50]` and `124`, as cvtColor reorders channels in place.

---

**Q3.** An engineer compares OpenCV filtering with SciPy convolution on a 5 × 5 image using a vertical-edge Sobel kernel.

```python
import numpy as np, cv2
from scipy.signal import convolve2d

I = np.array([[1, 2, 0, 1, 3],
              [0, 1, 3, 2, 1],
              [2, 1, 0, 1, 0],
              [1, 3, 2, 0, 2],
              [0, 1, 1, 2, 1]], dtype=np.float32)
K = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]], dtype=np.float32)

a = cv2.filter2D(I, -1, K)[2, 2]
b = convolve2d(I, K, mode="valid")[1, 1]
print(a, b)
```

Both indices refer to the output centred on the image's central pixel. Which option matches the printed values?

A. -2.0 and 2.0, since filter2D flips the kernel before sliding it.

B. 2.0 and 2.0, as both libraries compute the same discrete convolution.

C. -2.0 and -2.0, since the kernel is symmetric under a rotation.

D. 2.0 and -2.0, the two results differing only in sign.

---

**Q4.** A quality-inspection pipeline runs `cv2.Canny(gray, 50, 150)` on images of machined parts. To suppress clutter, an engineer changes the call to `cv2.Canny(gray, 50, 220)`, leaving the image and the lower threshold unchanged.

Compared with the original edge map, which change is expected?

A. New faint edges appear, since a higher upper threshold widens the hysteresis band.

B. Chains with no pixel above 220 disappear, even where their weak pixels exceed 50.

C. Only pixels between 50 and 220 vanish, while each chain's strong core stays.

D. Connected chains stay unchanged, since hysteresis judges survival by the lower threshold.

---

**Q5.** A team uses a ViT-Base model with 224 × 224 RGB inputs, 16 × 16 non-overlapping patches, a hidden size of 768, a learnable class token, and a linear patch projection with bias.

Which option gives the transformer's input sequence length and the parameter count of the patch-projection layer?

A. 196 tokens and 590,592 parameters, as the class token comes later.

B. 197 tokens, and 590,592 parameters in the patch-projection layer itself.

C. 197 tokens and 589,824 parameters, as patch projections carry no bias.

D. 197 tokens and 151,296 parameters, one 768-vector for each sequence position.

---

**Q6.** AlexNet's first convolutional layer applies 96 kernels of size 11 × 11 with stride 4 and no padding to a 227 × 227 × 3 input.

Which option gives the spatial size of its output and the layer's parameter count, including biases?

A. 54 × 54 and 34,944 parameters, using the paper's 224 input.

B. 55 × 55 and 34,848 parameters, as biases fold into normalization.

C. 55 × 55 and 34,944 parameters, with the 96 biases included.

D. 55 × 55 and 290,400 parameters, one per output activation.

---

**Q7.** A pedestrian detector produces four boxes (x1, y1, x2, y2) with confidence scores. The single ground-truth box is (50, 50, 150, 150). Greedy non-maximum suppression is applied with an IoU threshold of 0.5, and surviving boxes are then matched to the ground truth at IoU ≥ 0.5.

| Box | Coordinates | Score |
|---|---|---|
| P1 | (60, 60, 160, 160) | 0.92 |
| P2 | (55, 40, 150, 140) | 0.88 |
| P3 | (100, 100, 200, 200) | 0.75 |
| P4 | (300, 300, 380, 380) | 0.70 |

Which option is correct?

A. P1, P3 and P4 survive, giving one true positive and two false positives.

B. P2, P3 and P4 survive, as NMS keeps the best-aligned box.

C. P1, P2 and P4 survive, as P1 and P2 overlap below 0.5.

D. P1, P3 and P4 survive, giving two true positives through P3.

---

**Q8.** A team trains a shelf-product detector. Each sample is an image plus a list of bounding boxes in pixel coordinates.

```python
import random
import torchvision.transforms.functional as TF

class ShelfDataset:
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __getitem__(self, i):
        img, boxes, labels = self.items[i]           # boxes: [x1, y1, x2, y2]
        if self.train:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                img = TF.hflip(img)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        return img, boxes, labels
```

Validation mAP stalls well below training expectations. Which line is the main problem?

A. The brightness jitter, since photometric changes shift the box coordinates too.

B. The ImageNet normalization, since it is also applied at validation time.

C. The `to_tensor` call, since it rescales box coordinates into unit range.

D. The horizontal flip applied inside the training branch of `__getitem__`.

---

**Q9.** The following transfer-learning script raises a runtime error.

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, 10)                        # line 5

for p in model.parameters():                         # line 7
    p.requires_grad = False

opt = torch.optim.SGD(model.parameters(), lr=0.01)   # line 10
x, y = torch.randn(8, 3, 224, 224), torch.randint(0, 10, (8,))
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()                                      # line 13
opt.step()
```

Which line is responsible for the error, and what is the cause?

A. Line 10, since SGD rejects parameter lists where no tensor requires gradients.

B. Line 5, since the new head expects 2048 features from the ResNet backbone.

C. Line 7, which also freezes the new head, so line 13 has no gradient path.

D. Line 13, since cross-entropy needs one-hot targets, not integer class indices.

---

**Q10.** A 4 × 4 feature map is pooled.

```python
import numpy as np

X = np.array([[1, 3, 2, 4],
              [5, 6, 1, 2],
              [7, 2, 9, 0],
              [1, 8, 3, 4]], dtype=float)

mp = X.reshape(2, 2, 2, 2).max(axis=(1, 3))    # 2x2 window, stride 2
ap = X.reshape(2, 2, 2, 2).mean(axis=(1, 3))
print(mp.tolist(), ap[0, 1])
```

Which option matches the printed output?

A. `[[6, 4], [8, 9]]` and `2.25`, the mean of the top-right window.

B. `[[6, 4], [8, 9]]` and `3.75`, the mean of the top-left window.

C. `[[9, 9], [9, 9]]` and `2.25`, as windows overlap at stride 1.

D. `[[6, 4], [8, 9]]` and `4.0`, reading index (0, 1) as bottom-right.

---

**Q11.** The following inspection script raises a runtime error.

```python
import cv2
import matplotlib.pyplot as plt

img  = cv2.imread("pcb_scan.png")                  # loads successfully
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

box  = cv2.blur(gray, (4, 4))                      # line 7
med  = cv2.medianBlur(gray, 5)                     # line 8
gau  = cv2.GaussianBlur(gray, (6, 6), 1.5)         # line 9
edges = cv2.Canny(gau, 50, 150)

plt.imshow(img)                                    # line 12
plt.title("input")
plt.show()
```

Which line raises the error?

A. Line 7, since box filters need an odd kernel for a defined centre.

B. Line 8, since median filtering here accepts only three-channel input.

C. Line 12, since matplotlib cannot render arrays stored in BGR order.

D. Line 9, the Gaussian smoothing call that feeds the Canny step.

---

**Q12.** A 3 × 3 grayscale neighbourhood is filtered with the standard Sobel kernels Gx = [[−1, 0, 1], [−2, 0, 2], [−1, 0, 1]] and Gy = [[−1, −2, −1], [0, 0, 0], [1, 2, 1]], applied as correlation.

| | col 0 | col 1 | col 2 |
|---|---|---|---|
| row 0 | 10 | 10 | 80 |
| row 1 | 10 | 50 | 90 |
| row 2 | 20 | 90 | 100 |

Which option gives the L2 gradient magnitude and the gradient direction, measured from the x-axis, at the centre pixel?

A. Magnitude 500 and direction about 31.5°, summing absolute components.

B. Magnitude about 364 and direction about 31.5° measured from the x-axis.

C. Magnitude about 364 and 58.5°, as atan2 takes Gx first.

D. Magnitude about 310 and direction 0°, as only Gx responds.

---

**Q13.** A thumbnailing script halves a 640 × 480 colour image and then makes a separate anisotropic copy.

```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
half = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
squash = cv2.resize(img, None, fx=0.5, fy=0.25)
print(half.shape, squash.shape)
```

Which option matches the printed shapes?

A. `(320, 240, 3)` and `(120, 320, 3)`, from the two resize calls.

B. `(240, 320, 3)` and `(120, 320, 3)`, as dsize follows NumPy order.

C. `(320, 240, 3)` and `(320, 120, 3)`, as fx scales the rows.

D. `(240, 320, 3)` and `(320, 120, 3)`, with both in row order.

---

**Q14.** An edge-detection step for road-marking extraction currently reads:

```python
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blur, 60, 160)
```

The blur kernel is changed to `(9, 9)` with everything else unchanged. Which effect on the edge map is expected?

A. More edges appear, since blurring spreads each gradient across neighbouring pixels.

B. No change, since Canny already applies its own internal Gaussian with matching sigma.

C. Fine texture edges fade, while long, high-contrast boundaries largely persist.

D. Edges become thinner, since smoothing removes the double responses suppression misses.

---

**Q15.** A ViT-B/16 pretrained at 224 × 224 with a class token is fine-tuned at 384 × 384, keeping the patch size at 16.

Which option correctly describes the new input and its cost?

A. 197 tokens, since position embeddings fix length, and unchanged attention cost.

B. 577 tokens with fresh random position embeddings, and about 2.9× attention cost.

C. 576 tokens with interpolated position embeddings, and about 8.6× attention cost.

D. 577 tokens, with interpolated position embeddings, and attention costing about 8.6× more.

---

**Q16.** A YOLOv3-style detector predicts three anchor boxes per grid cell at three scales (13 × 13, 26 × 26 and 52 × 52 for a 416 × 416 input), with 80 classes. Each anchor predicts four box offsets, one objectness score, and class scores.

Which option gives the number of output channels at the 13 × 13 scale and the total number of predicted boxes per image?

A. 85 channels and 3,549 boxes, as each cell predicts one box.

B. 255 channels and 10,647 boxes in total across all three scales.

C. 255 channels and 3,549 boxes, as anchors share each cell's prediction.

D. 252 channels and 10,647 boxes, as objectness folds into class scores.

---

**Q17.** A document-scanning step binarizes pixel intensities.

```python
import cv2
import numpy as np

a = np.array([[12, 127, 128],
              [200, 90, 255]], dtype=np.uint8)
_, b = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
_, z = cv2.threshold(a, 127, 255, cv2.THRESH_TOZERO)
print(b[0].tolist(), z[1].tolist())
```

Which option matches the output?

A. `[0, 255, 255]` and `[200, 0, 255]`, as the threshold test is inclusive.

B. `[0, 0, 255]` and `[127, 90, 127]`, as TOZERO caps values at 127.

C. `[0, 0, 255]` and `[200, 0, 255]`, for the two threshold modes.

D. `[0, 255, 255]` and `[255, 0, 255]`, as TOZERO binarizes survivors.

---

**Q18.** A defect classifier processes 128 × 128 single-channel images.

```python
import torch.nn as nn

features = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
)
classifier = nn.Linear(IN_FEATURES, 2)
```

What must `IN_FEATURES` be?

A. 14,400, from a 15 × 15 map with 64 channels.

B. 16,384, as dilation with padding 2 grows the map.

C. 12,544, since each convolution trims two pixels per dimension.

D. 57,600, since unpadded stride-2 convolutions halve without flooring.

---

**Q19.** A fruit-sorting line flags red items by thresholding in HSV.

```python
import cv2
import numpy as np

px = np.uint8([[[0, 0, 255]],       # p1
               [[0, 128, 255]],     # p2
               [[255, 0, 170]],     # p3
               [[20, 20, 200]]])    # p4   (all BGR)
hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
m2 = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
mask = cv2.bitwise_or(m1, m2)
print(np.flatnonzero(mask.ravel()))
```

Which pixels are flagged?

A. p1, p2 and p4, as orange's 30° hue falls within 0–10.

B. p1 only, since p4's darker value shifts its hue outside both ranges.

C. p1, p3 and p4, as magenta's hue wraps into 170–179.

D. p1 and p4, the two pixels whose stored hue is 0.

---

**Q20.** A colleague summarizes what made AlexNet (2012) succeed on ImageNet. Which list of design choices is historically accurate for that network?

A. Batch normalization after each convolution, which permitted much higher learning rates.

B. ReLU activations, overlapping max-pooling, and dropout within its fully connected layers.

C. Uniform stacks of 3 × 3 convolutions for cheap large receptive fields.

D. Residual shortcut connections, letting gradients bypass each of its eight layers.

---

# ANSWER KEY

| Q | Ans | Domain |
|---|-----|--------|
| 1 | C | CV |
| 2 | A | CV |
| 3 | D | CV |
| 4 | B | CV |
| 5 | B | CV |
| 6 | C | CV |
| 7 | A | CV |
| 8 | D | CV |
| 9 | C | CV |
| 10 | A | CV |
| 11 | D | CV |
| 12 | B | CV |
| 13 | A | CV |
| 14 | C | CV |
| 15 | D | CV |
| 16 | B | CV |
| 17 | C | CV |
| 18 | A | CV |
| 19 | D | CV |
| 20 | B | CV |

# EXPLANATIONS

**Q1 — C.** Frame: 1920 × 1080 × 3 × 1 byte = 6,220,800. Batch: 32 × 3 × 224 × 224 × 4 bytes = 19,267,584. A counts a single channel. B forgets the float32 conversion (4,816,896 bytes). D treats the raw frame as float32.

**Q2 — A.** imread returns BGR, so the BGR→RGB conversion gives [50, 100, 200]. The second call wrongly applies the BGR2GRAY formula to RGB data, treating 200 as R: 0.299 × 200 + 0.587 × 100 + 0.114 × 50 = 124.2 → 124. The true luminance is 96, which B reports. C and D misread the channel order of imread or cvtColor.

**Q3 — D.** cv2.filter2D computes correlation without flipping the kernel, giving 2.0 at the centre. convolve2d performs true convolution, which rotates the kernel 180°; for this antisymmetric kernel that negates the result to −2.0. A has the flip backwards. B assumes identical operations. C is wrong because this kernel is antisymmetric, not symmetric, under rotation.

**Q4 — B.** In hysteresis, a pixel above high is a strong seed, and weak pixels (between low and high) survive only if connected to a seed. Raising high removes seeds, so any chain whose maximum lies between 150 and 220 disappears entirely. Chains that still contain a pixel above 220 keep their connected weak pixels. A gets the direction wrong. C ignores connectivity. D ignores that seeds come from the upper threshold.

**Q5 — B.** (224/16)² = 196 patches plus one class token gives 197 tokens. Projection = 16 × 16 × 3 × 768 + 768 = 590,592. A omits the class token from the sequence. C drops the bias. D gives the positional-embedding table size (197 × 768).

**Q6 — C.** (227 − 11)/4 + 1 = 55. Parameters = 96 × (11 × 11 × 3) + 96 = 34,944. A uses 224, which does not divide cleanly ((224 − 11)/4 + 1 = 54.25). B omits the biases. D counts output activations per channel (55 × 55 × 96 = 290,400), not parameters.

**Q7 — A.** In score order, P1 is kept. IoU(P1, P2) = 0.585 > 0.5, so P2 is suppressed even though it matches the ground truth better (0.78 vs 0.68). P3 (IoU with P1 = 0.22) and P4 survive. Against the ground truth, only P1 has IoU ≥ 0.5, giving 1 TP and 2 FP. B misdescribes NMS, which ranks by score, not ground-truth fit. C misjudges the P1–P2 overlap. D counts P3, whose IoU with the ground truth is 0.14.

**Q8 — D.** The flip mirrors the image, but the boxes are returned unchanged, so half of the training samples carry boxes on the wrong side. That label noise caps detection quality. A is wrong because photometric jitter does not move geometry. B is correct practice. C is wrong because to_tensor rescales pixel values, not box coordinates.

**Q9 — C.** The new head is created before the freezing loop, so its parameters are frozen too. The forward output has no grad_fn, and loss.backward() raises "element 0 of tensors does not require grad and does not have a grad_fn". The fix is to freeze first, then replace fc. A is wrong: PyTorch optimizers accept such lists. B is wrong: resnet18's head input is 512. D is wrong: CrossEntropyLoss takes class indices.

**Q10 — A.** The 2 × 2 windows give maxima [[6, 4], [8, 9]]. Window (0, 1) contains 2, 4, 1 and 2, with mean 2.25. B gives window (0, 0) (mean 3.75). C applies overlapping 3 × 3 windows. D reads (0, 1) as the bottom-right window, whose mean is 4.0.

**Q11 — D.** GaussianBlur requires odd, positive kernel sizes, and (6, 6) fails OpenCV's assertion. A is wrong: cv2.blur accepts even box sizes. B is wrong: medianBlur with odd ksize 5 on a single-channel uint8 image is valid. C is a real but non-blocking bug: matplotlib renders BGR data with red and blue swapped and raises nothing.

**Q12 — B.** Gx = 310 and Gy = 190. Magnitude = sqrt(310² + 190²) = 363.6 and direction = atan2(190, 310) = 31.5°. A uses the L1 norm (|Gx| + |Gy| = 500). C swaps the atan2 arguments. D ignores Gy.

**Q13 — A.** cv2.resize takes dsize as (width, height). Passing (shape[0]//2, shape[1]//2) = (240, 320) requests width 240 and height 320, so the array shape is (320, 240, 3). With fx = 0.5 on width and fy = 0.25 on height, the second copy is (120, 320, 3). B assumes NumPy order. C and D attach fx to rows.

**Q14 — C.** A larger Gaussian lowers gradient magnitudes of fine, high-frequency texture below the fixed thresholds. Strong, extended boundaries keep enough gradient to remain, though corners round slightly. A gets the direction wrong. B is false: cv2.Canny computes Sobel gradients on the input as given and does not blur internally. D is wrong: non-maximum suppression already thins edges.

**Q15 — D.** (384/16)² = 576 patches plus the class token gives 577 tokens. The 14 × 14 grid of pretrained position embeddings is interpolated to 24 × 24. Self-attention cost scales with sequence length squared: (577/197)² ≈ 8.6. A wrongly assumes a fixed length. B discards the pretrained position embeddings and scales cost linearly. C omits the class token.

**Q16 — B.** Channels = 3 × (4 + 1 + 80) = 255. Boxes = 3 × (169 + 676 + 2704) = 10,647. A gives one anchor per cell at all scales. C counts only cells across scales without multiplying by anchors. D drops objectness.

**Q17 — C.** THRESH_BINARY sets values strictly greater than 127 to 255, so 127 → 0 and the first row is [0, 0, 255]. THRESH_TOZERO keeps values above the threshold and zeros the rest, so the second row is [200, 0, 255]. A and D treat the test as inclusive. B describes THRESH_TRUNC.

**Q18 — A.** 128 → conv(7, s2, p3) → 64 → maxpool(3, s2, p1) → 32 → dilated conv (effective kernel 5, p2) → 32 → conv(3, s2, p0) → floor(29/2) + 1 = 15. Flatten = 64 × 15 × 15 = 14,400. B treats the dilated convolution as an ordinary 3 × 3 with padding 2 (34 → 16 → 16,384). C and D do not match any valid trace.

**Q19 — D.** OpenCV stores 8-bit hue as 0–179 (degrees halved). The computed hues are p1 = 0, p2 = 15, p3 = 140 and p4 = 0 (saturation 229, value 200), so p1 and p4 pass. A uses the 0–360 scale; p2's hue of 15 lies outside 0–10. B is wrong: value does not change hue. C is wrong: magenta's hue of 140 is outside 170–179.

**Q20 — B.** AlexNet used ReLU, overlapping 3 × 3 stride-2 max-pooling, dropout in the fully connected layers, heavy augmentation, LRN and a two-GPU split. A describes batch normalization, introduced in 2015. C describes VGG (2014). D describes ResNet (2015).

---
