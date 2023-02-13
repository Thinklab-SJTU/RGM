# RGM

This repo provides code for 'Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching', which is accepted by ICLR 2023.

Here are the instructions for the key codes:

dqn_model_r.py : the code for the RL agent and its neural networks.

environment_qap_ag.py : the code for the RL environment.

run_rgm_pytorch_willow.py : the main code for RGM, run this code can train or test RGM.

Example arguments for run_rgm_pytorch_willow.py (the pretrained model required is given in the model/ folder):

python run_rgm_pytorch_willow.py --train 0 --cuda 1 --support 1 --gamma 0.9 --bs 64 --rs 100000 --lr 1e-5 --sync 40 --ls 1000 --ms 1 --ed 500000 --eb 1.1 --es 1.0 --ef 0.02 --units 64 --hs 64 --t 3 --ad 0.1 --af 0.01 --ab 5 --b 10 --d 0 --cls Car --load_from_cls Car --outlier 2 --hard_mask 0 --normalize 1 --inlier_count 1 --inlier 10

For the meaning of the arguments, please see in the code of run_rgm_pytorch_willow.py and refer to our paper.

We provide the pretrained model for the example arguments, if you want to try other arguments, please train RGM from scratch by setting "--train 1".
