# Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI

Robert Hönig, [Javier Rando](https://javirando.com/), [Nicholas Carlini](https://nicholas.carlini.com), [Florian Tramèr](https://floriantramer.com/) | SPY Lab (ETH Zurich) and Google Deepmind*

Official repository for the paper [**Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI**]().

# Data

We are releasing the dataset used to train our models and the generated images included in the user study. The dataset only contains data from historical artists. You can access the entire dataset at: https://drive.google.com/file/d/11u6ITxkrAvWLYnry__9rm1QIOQbSKasT. The dataset is organized as follows:

* `training_images`: contains the images used to train our models
    * `original`: original artwork from historical artists, as exported from WikiArt
    * `protected`: contains folders for each protection method (`antidb`, `mist`, `glaze`), with the original artwork after applying the perturbations
    * `protected+preprocessed`: contains the same images as `protected`, but with 3 robust mimicry methods applied for each protection (`diffpure`, `noisy_upscaling`, `impress++`). Note: Impress++ only includes gaussian noising and reverse optimization preprocessing steps, as the remaining steps are applied during sampling.
* `generated_images`: contains the generated images from the resulting models, with one image for each protection. The `no-protections` folder contains images generated from models finetuned directly on the original artwork, using the same structure as `generated_images` (folders for each protection and robust mimicry method) to match the different seeds used for each combination.
