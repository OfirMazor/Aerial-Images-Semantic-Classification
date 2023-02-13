# Aerial-Images-Semantic-Classification

<p>
  <a href="https://colab.research.google.com/drive/1Ulv57Z-VsE_G5drH7aj473uRGyBFWCuk?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>





## Introduction
----------------
Semantic Classification of Aerial Images from Drone using Unet






## Data
--------
The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts  more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images and the test set is made up of 200 private images.

more details are in [Dataset page](https://www.tugraz.at/institute/icg/research/team-fraundorfer/software-media/dronedataset)






## Getting started
-------------------
*Steps are executed in [Colab notebook]("https://colab.research.google.com/drive/1Ulv57Z-VsE_G5drH7aj473uRGyBFWCuk?usp=sharing)

1) Clone the repository .

2) Install requirements describes in [requirements.txt](https://github.com/OfirMazor/Aerial-Images-Semantic-Classification/blob/main/requirements.txt).

3) Use the Preprocessing.py script to convert RGB masks to single channel mask image.
   * This step is not requierd if you already own a converted version of masks. Once converted all rgb masks - the step is no longer require.
