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


<!-- | Name        | R   | G   | B   | Color                                                                                           |
| ----------- | --- | --- | --- | ----------------------------------------------------------------------------------------------- |
| unlabeled   | 0   | 0   | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,0,0)" /></svg>       |
| paved-area  | 128 | 64  | 128 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(128,64,128)" /></svg>  |
| dirt        | 130 | 76  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(130,76,0)" /></svg>    |
| grass       | 0   | 102 | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,102,0)" /></svg>     |
| gravel      | 112 | 103 | 87  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(112,103,87)" /></svg>  |
| water       | 28  | 42  | 168 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(28,42,168)" /></svg>   |
| rocks       | 48  | 41  | 30  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(48,41,30)" /></svg>    |
| pool        | 0   | 50  | 89  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(0,50,89)" /></svg>     |
| vegetation  | 107 | 142 | 35  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(107,142,35)" /></svg>  |
| roof        | 70  | 70  | 70  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(70,70,70)" /></svg>    |
| wall        | 102 | 102 | 156 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(102,102,156)" /></svg> |
| window      | 254 | 228 | 12  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(254,228,12)" /></svg>  |
| door        | 254 | 148 | 12  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(254,148,12)" /></svg>  |
| fence       | 190 | 153 | 153 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(190,153,153)" /></svg> |
| fence-pole  | 153 | 153 | 153 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(153,153,153)" /></svg> |
| person      | 255 | 22  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(255,22,0)" /></svg>    |
| dog         | 102 | 51  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(102,51,0)" /></svg>    |
| car         | 9   | 143 | 150 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(9,143,150)" /></svg>   |
| bicycle     | 119 | 11  | 32  | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(119,11,32)" /></svg>   |
| tree        | 51  | 51  | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(51,51,0)" /></svg>     |
| bald-tree   | 190 | 250 | 190 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(190,250,190)" /></svg> |
| ar-marker   | 112 | 150 | 146 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(112,150,146)" /></svg> |
| obstacle    | 2   | 135 | 115 | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(2,135,115)" /></svg>   |
| conflicting | 255 | 0   | 0   | <svg width="30" height="20"><rect width="30" height="20" style="fill:rgb(255,0,0)" /></svg>     | -->



## Getting started
*Steps are executed in [Colab notebook](https://colab.research.google.com/drive/1Ulv57Z-VsE_G5drH7aj473uRGyBFWCuk?usp=sharing)

1) Clone the repository.

2) Install requirements describes in [requirements.txt](https://github.com/OfirMazor/Aerial-Images-Semantic-Classification/blob/main/requirements.txt).

3) Use the Preprocessing.py script to convert RGB masks to single channel mask image.
   * This step is not requierd if you already own a converted version of masks. Once converted all rgb masks - the step is no longer require.
