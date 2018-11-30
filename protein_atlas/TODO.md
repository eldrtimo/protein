# To Do
## Data Labeling
The labels are contained in the "Target" column of "train.csv"; they are strings.

This is a multiclass dataset; the labels must read into integer arrays and then
one-hot-encodeded.

## Image Processing/Visualization
Important to visualize training examples to understand what the network is doing.

This will be accomplished with functions in `skimage.exposure`.

Complicating this task is the fact that each image has four channels,
corresponding to different protein structures:

0. red: Microtubules
1. green: Antibody
2. blue: Nucleus
3. yellow: Endoplasmic Reticulum

## Multilabel Classification Problem
Given a set of images with the channels as stated, find which subset of labels
below apply to the images:
 0. "Nucleoplasm"
 1. "Nuclear membrane"
 2. "Nucleoli"
 3. "Nucleoli fibrillar center"
 4. "Nuclear speckles"
 5. "Nuclear bodies"
 6. "Endoplasmic reticulum"
 7. "Golgi apparatus"
 8. "Peroxisomes"
 9. "Endosomes"
10. "Lysosomes"
11. "Intermediate filaments"
12. "Actin filaments"
13. "Focal adhesion sites"
14. "Microtubules"
15. "Microtubule ends"
16. "Cytokinetic bridge"
17. "Mitotic spindle"
18. "Microtubule organizing center"
19. "Centrosome"
20. "Lipid droplets"
21. "Plasma membrane"
22. "Cell junctions"
23. "Mitochondria"
24. "Aggresome"
25. "Cytosol"
26. "Cytoplasmic bodies"
27. "Rods & rings"


### Visualizing Hyperspectral Images
This looks like a good resource:

https://personalpages.manchester.ac.uk/staff/d.h.foster/Tutorial_HSI2RGB/Tutorial_HSI2RGB.html

## Neural Network
