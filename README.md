# Attentional Adaptive Fusion For Real-Time Monte Carlo Path Tracing Denoising
### Welcome to our the official code repo of our research paper, (insert paper title)

## Dataset
We utilise the BMFR dataset in this paper which can be found at: https://etsin.fairdata.fi/dataset/0ab24b68-4658-4259-9f1d-3150be898c63/data (both parts are required)

## Prerequisites
(The software along with versions we used)

  - pytorch v2.6.0
  - cuda v12.4
  - pyexr v0.5.0
  - scikit-image v0.26.0
  - tqdm v4.67.3
  - psutil v7.2.2

## Instructions

Extract both parts of the BMFR dataset into a directory called 'dataset' within the project root directory (within the dataset directory you should see directories named after all the BMFR scenes)
The scenes are:
  - classroom
  - living-room
  - san-miguel
  - sponza
  - sponza-glossy
  - sponza-moving-light

After the dataset is prepared run:
```bash
python data_preprocess.py
```
