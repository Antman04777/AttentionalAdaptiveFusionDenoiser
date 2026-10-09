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

<details>
After the dataset is prepared run:
```bash
python data_preprocess.py
```
In order to generate the temporally accumulated irradiance maps and depth maps (this is a one and done execution)
</details>

The checkpoints of our model for each scene lie within the checkpoints directory, in order to test them, simply, specify the scene you would like to denoise in the test_parameters.json file, next to the 'test_scene' key

An example of this would look like:
```json
{
    "experiment": "Rep+AFF+Aux+NoWorld",
    "test_scene": "san-miguel"
}
```
(Since the checkpoints are provided in this project, you do not need to change the experiment name from the example, unless you have trained your own experiment)

Next, simply enable your virtual environment with the necessary installed pip packages and run:
```bash
python test.py --parameters test_parameters.json
```

Once the inference is done, you may view the results within the 'results' directory under the project root, navigate to the directory named after the experiment provided in the test_parameters.json and then the directory named after the scene which you ran inference on (eg. san-miguel).

The images are saved as .exr files, it is encouraged that you use an exr viewer such as mrViewer (https://sourceforge.net/projects/mrviewer/)

Should you wish to train your own experiment, simply adjust the parameters within the train_parameters.json

And example of this would look like:
```json
{
    "experiment": "new-experiment",
    "current_testing_scene_name": "san-miguel",
    "learning_rate": 0.001,
    "loss_function": "smape",
    "epochs": 400,
    "epochs_validate": 25,
    "batch_size": 16,
    "previously_elapsed_epochs": 0
}
```

Next, enable your virtual environment with the necessary installed pip packages and run:
```bash
python train.py --parameters train_parameters.json
```

If you would like to resume training (useful for training on clusters where walltime may be exceeded), run:
```bash
python train.py --parameters train_parameters.json --resume
```
Run this using the parameters present in train_parameters.json prior to your stopped training (do not modify this json file if you wish to resume training)
