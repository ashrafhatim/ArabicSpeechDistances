# Implementation of Wav2Vec2 Distances intended for the Arabic speech inspired by the DeepSpeech Distances proposed in High Fidelity Speech Synthesis with Adversarial Networks [paper](https://arxiv.org/abs/1909.11646).

This repo provides a code for estimation of Wav2Vec2 Distances, new evaluation metrics for neural Arabic speech synthesis.

## Details
The computation involves estimating Fréchet between high-level features of the reference and the examined samples extracted from hidden representation of the pre-trained Wav2Vec2ForCTC speech recognition model from the HuggingFace transformets library.

## References
1. Mikołaj Bińkowski, Jeff Donahue, Sander Dieleman, Aidan Clark, Erich Elsen, Norman Casagrande, Luis C. Cobo, Karen Simonyan, [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/abs/1909.11646), ICLR 2020.
2. Authors' [Implementation](https://github.com/mbinkowski/DeepSpeechDistances) of DeepSpeech Distances.
3. Fréchet Inception Distance (FID) for Pytorch [Implementation](https://github.com/hukkelas/pytorch-frechet-inception-distance).

