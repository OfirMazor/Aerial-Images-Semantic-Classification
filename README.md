# Aerial-Images-Semantic-Classification

<p>
  <a href="https://colab.research.google.com/drive/1Ulv57Z-VsE_G5drH7aj473uRGyBFWCuk?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>



## Introduction
This repo is a personal practice for semantic classification of imagesusing Unet.







## Data
The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts  more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images and the test set is made up of 200 private images.

more details are in [Dataset page](https://www.tugraz.at/institute/icg/research/team-fraundorfer/software-media/dronedataset).
Sample of the dataset images can be found in the [Data](https://github.com/OfirMazor/Aerial-Images-Semantic-Classification/blob/main/Data/) folder.

Masks color mapping:
| Name        | R   | G   | B   | Color                                                                                        |
| ----------- | --- | --- | --- | -------------------------------------------------------------------------------------------- |
| unlabeled   | 0   | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/unlabeled.png" /></p>   |
| paved-area  | 128 | 64  | 128 | <p align="center"><img width = "30" height= "20" src="./label_colors/paved-area.png" /></p>  |
| dirt        | 130 | 76  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dirt.png" /></p>        |
| grass       | 0   | 102 | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/grass.png" /></p>       |
| gravel      | 112 | 103 | 87  | <p align="center"><img width = "30" height= "20" src="./label_colors/gravel.png" /></p>      |
| water       | 28  | 42  | 168 | <p align="center"><img width = "30" height= "20" src="./label_colors/water.png" /></p>       |
| rocks       | 48  | 41  | 30  | <p align="center"><img width = "30" height= "20" src="./label_colors/rocks.png" /></p>       |
| pool        | 0   | 50  | 89  | <p align="center"><img width = "30" height= "20" src="./label_colors/pool.png" /></p>        |
| vegetation  | 107 | 142 | 35  | <p align="center"><img width = "30" height= "20" src="./label_colors/vegetation.png" /></p>  |
| roof        | 70  | 70  | 70  | <p align="center"><img width = "30" height= "20" src="./label_colors/roof.png" /></p>        |
| wall        | 102 | 102 | 156 | <p align="center"><img width = "30" height= "20" src="./label_colors/wall.png" /></p>        |
| window      | 254 | 228 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/window.png" /></p>      |
| door        | 254 | 148 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/door.png" /></p>        |
| fence       | 190 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence.png" /></p>       |
| fence-pole  | 153 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence-pole.png" /></p>  |
| person      | 255 | 22  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/person.png" /></p>      |
| dog         | 102 | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dog.png" /></p>         |
| car         | 9   | 143 | 150 | <p align="center"><img width = "30" height= "20" src="./label_colors/car.png" /></p>         |
| bicycle     | 119 | 11  | 32  | <p align="center"><img width = "30" height= "20" src="./label_colors/bicycle.png" /></p>     |
| tree        | 51  | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/tree.png" /></p>        |
| bald-tree   | 190 | 250 | 190 | <p align="center"><img width = "30" height= "20" src="./label_colors/bald-tree.png" /></p>   |
| ar-marker   | 112 | 150 | 146 | <p align="center"><img width = "30" height= "20" src="./label_colors/ar-marker.png" /></p>   |
| obstacle    | 2   | 135 | 115 | <p align="center"><img width = "30" height= "20" src="./label_colors/obstacle.png" /></p>    |
| conflicting | 255 | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/conflicting.png" /></p> |




## Getting started
*Steps are executed in [Colab notebook](https://colab.research.google.com/drive/1Ulv57Z-VsE_G5drH7aj473uRGyBFWCuk?usp=sharing)

1) Clone the repository.

2) Install requirements describes in [requirements.txt](https://github.com/OfirMazor/Aerial-Images-Semantic-Classification/blob/main/requirements.txt).

3) Use the Preprocessing.py script to convert RGB masks to single channel mask image.
   * This step is not requierd if you already own a converted version of masks. Once converted all rgb masks - the step is no longer require.
