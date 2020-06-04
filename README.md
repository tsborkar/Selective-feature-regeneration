#  Defending Against Universal Attacks Through Selective Feature Regeneration (CVPR 2020)
<img src="https://github.com/tsborkar/Selective-feature-regeneration/blob/master/fig/5625-teaser.gif" width="350" height="350"> [<img src="fig/teaser_frame0.PNG" height="350" width="540">](https://youtu.be/wMWhb7xqubg)


## Introduction
<p align="justify">
Deep neural network (DNN) predictions have been shown to be vulnerable to carefully crafted adversarial perturbations. Specifically, image-agnostic (universal adversarial) perturbations added to any image can fool a target network into making erroneous predictions. Departing from existing defense strategies that work mostly in the image domain, we present a novel defense which operates in the DNN feature domain and effectively defends against such universal perturbations. Our approach identifies pre-trained convolutional features that are most vulnerable to adversarial noise and deploys trainable feature regeneration units which transform these DNN filter activations into resilient features that are robust to universal perturbations. Regenerating only the top 50% adversarially susceptible activations in at most 6 DNN layers and leaving all remaining DNN activations unchanged, we outperform existing defense strategies across different network architectures by more than 10% in restored accuracy. We show that without any additional modification, our defense trained on ImageNet with one type of universal attack examples effectively defends against other types of unseen universal attacks. 

A complete description of our CVPR 2020 work can be found in the pre-print on [ArXiv](https://arxiv.org/abs/1906.03444) as well as at the [Project Page](https://www.cs.princeton.edu/~fheide/SelectiveFeatureRegeneration/). </p>

<img align="center" src="fig/intro_fig.png" height="400">

<p align="justify"><em><b>
Defending Against Adversarial Attacks by Selective Feature Regeneration: Convolutional filter activations in the baseline DNN (top) are first sorted in order of vulnerability to adversarial noise using their respective filter weight norms (see Manuscript). For each considered layer, we use a feature regeneration unit, consisting of a residual block with a single skip connection (4 layers), to regenerate only the most adversarially susceptible activations into resilient features that restore the lost accuracy of the baseline DNN, while leaving the remaining filter activations unchanged. We train these units on both clean and perturbed images in every mini-batch using the same target loss as the baseline DNN such that all parameters of the baseline DNN are left unchanged during training. </b></em></p>

## Citation

If you use our code, models or need to refer to our results, please use the following:

```
@inproceedings{selectivefeatadvdef,
 author = {Tejas Borkar and Felix Heide and Lina Karam},
 booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})},
 title = {Defending Against Universal Attacks Through Selective Feature Regeneration},
 year = {2020}
}
```

## Key Results on ILSVRC2012 Validation Set

### Restoration accuracy for [Universal Adversarial Peturbations](https://arxiv.org/abs/1610.08401) (UAP)

|   Methods        |  [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)        |    [VGG-F](https://arxiv.org/abs/1405.3531)  |   [GoogLeNet](https://arxiv.org/abs/1409.4842)  |  [VGG-16](https://arxiv.org/abs/1409.1556)   |  [Res152](https://arxiv.org/abs/1512.03385)  |
| :-----------:  | :--------------: | :---------: |   :----------: |   :---------:  | :---------: |
|  Baseline      |   0.596          | 0.628       |      0.691     |       0.681    |   0.670     |
|  *Ours*        |   **0.976**      | **0.967**   |      **0.970** |     **0.963**  |  **0.982**  | 

Please refer to Table 2. in our paper for additional details.


### Restoration accuracy for unseen stronger UAP attack perturbations against CaffeNet

|  Method    |  Attack Strength = 15    |   Attack Strength = 20    |   Attack Strength = 25      |
| :---------:|  :--------:  |:---------:|:----------:|
| Baseline   |     0.543    |    0.525  |     0.519   |
| *Ours*     |     **0.952**    |    **0.896**  |     **0.854**   |

Our defense is trained on attack examples with an attack strength of 10. Please refer to Table 4. in our paper for additional details. 

### Restoration accuracy for other types of unseen universal attacks 
<img src="fig/feat_regen.PNG">

###### *Effectiveness of feature regeneration units at masking adversarial perturbations in DNN feature maps for images perturbed by universal perturbations ([UAP](https://arxiv.org/abs/1610.08401), [NAG](https://arxiv.org/abs/1712.03390), [GAP](https://arxiv.org/abs/1712.02328) and [sPGD](https://arxiv.org/abs/1812.03705)). Perturbation-free feature map (clean), different adversarially perturbed feature maps (Row 1) and corresponding feature maps regenerated by feature regeneration units (Row 2) are obtained for a single filter channel in conv1 1 layer of VGG-16, along with an enlarged view of a small region in the feature map (yellow box). Feature regeneration units are only trained on UAP attack examples but are very effective at suppressing adversarial artifacts generated by unseen attacks (e.g., NAG, GAP and sPGD).*


#### CaffeNet

| Method |  [FFF](https://arxiv.org/abs/1707.05572) | [NAG](https://arxiv.org/abs/1712.03390) | [S. Fool](https://arxiv.org/abs/1709.03582) |
|:-------:| :------: | :-------:| :---------: |
| Baseline  | 0.645  |  0.670   |    0.815    |
| *Ours*    | **0.941** |  **0.840** |  **0.914** |

#### ResNet 152

| Method |  [GAP](https://arxiv.org/abs/1712.02328) | [G-UAP](https://arxiv.org/abs/1801.08092) | [sPGD](https://arxiv.org/abs/1812.03705) |
|:-------:| :------: | :-------:| :---------: |
| Baseline  | 0.640  | 0.726  | 0.671  |
| *Ours*    | **0.922** | **0.914** |  **0.976** |

Our defense is trained only one UAP attack examples. Please refer to Table 5. in our paper for additional details.

