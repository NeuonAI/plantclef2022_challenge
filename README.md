# Neuon AI in PlantCLEF 2022
This repository contains our team's training lists and scripts used in [PlantCLEF2022](https://www.aicrowd.com/challenges/lifeclef-2022-plant). 
 Given a training dataset of 4 million images and 80,000 species, the task of the challenge was to identify the correct plant species from 26,868 multi-image plant observations. We trained several deep learning models based on the Inception-v4 and Inception-ResNet-v2 architectures and submitted 9 runs. Our best submission achieved a Macro Averaged Mean Reciprocal Rank score of 0.608. The official results can be found [here](https://www.imageclef.org/PlantCLEF2022).

## Methodology
### Single CNN
This network resembles a conventional Inception-v4 and Inception-ResNet-v2 neural network. Similarly, it consists of convolutional layers, pooling layers, dropout layers and fully-connected layers, which return the softmax probabilities of its prediction. The multi-task classification is adopted in this network by utilising the five taxonomy labels: Class, Order, Family, Genus, and Species.

![Figure 1](https://github.com/NeuonAI/plantclef2022_challenge/blob/08dbc0e44fed35a0f713608ea4abc34a4b505258/single_cnn.png "Single CNN")


### Triplet Network
This network resembles the single CNN mentioned above. However, instead of using its fully-connected layer for its predictions, it is used to compute the plants' image embedding representation. In addition, a batch normalisation layer is added, followed by L2-normalisation, and finally, a triplet loss layer to train the optimum embedding representation of the plants. Furthermore, instead of its original 1536 features in the fully-connected layer, we reduced its final feature vector to 500. Due to resource limitation, we did not adopt the multi-classification approach in this training. Only the Species taxonomy label is utilised.

![Figure 2](https://github.com/NeuonAI/plantclef2022_challenge/blob/08dbc0e44fed35a0f713608ea4abc34a4b505258/triplet_network.png "Triplet Network")
