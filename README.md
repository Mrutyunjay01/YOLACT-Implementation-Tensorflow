# YOLACT-Implementation-Tensorflow
This repository is for implementation of YOLACT paper by Tensorlfow from scratch. Open to pull requests.
>>
I present a simple, fully-convolutional model for real-time (> 30 fps) instance segmentation that achieves competitive
results on MS COCO evaluated on a single Titan Xp, which is significantly faster than any previous state-of-the-art approach.
Moreover, we obtain this result after training on only one GPU. We accomplish this by breaking instance segmentation into two parallel
subtasks: (1) generating a set of prototype masks and (2) predicting per-instance mask coefficients. Then we produce instance masks
by linearly combining the prototypes with the mask coefficients. We find that because this process doesn’t depend on repooling, this
approach produces very high-quality masks and exhibits temporal stability for free. Furthermore, we analyze the emergent behavior of
our prototypes and show they learn to localize instances on their own in a translation variant manner, despite being fully-convolutional.
We also propose Fast NMS, a drop-in 12 ms faster replacement for standard NMS that only has a marginal performance penalty.
Finally, by incorporating deformable convolutions into the backbone network, optimizing the prediction head with better anchor scales
and aspect ratios, and adding a novel fast mask re-scoring branch, our YOLACT++ model can achieve 34:1 mAP on MS COCO at 33:5
fps, which is fairly close to the state-of-the-art approaches while still running at real-time.
>> Ref : arXiv:1912.06218v1 [cs.CV] 3 Dec 2019

- Repo References:
[https://github.com/leohsuofnthu/Tensorflow-YOLACT]
[https://github.com/dbolya/yolact]
