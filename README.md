# RL
Small Project for Reinforcement Learning

## 1. Flappy Bird
The flappy bird here is very simple. The algorithm used is simply distrbuting the final rewards along the flying-path. It is not Q-learning (which will require iterative updating q(s,a) with adjecent q(s,a), r, and q(s',a').

* run the flapppy bird
  * mode 'Run', which is using pretrained weights
  > python main_bird.py --mode 'Run'
  * mode 'Train', which is training new weights, and save updated weights periodically. 
  > python main_bird.py --mode 'Train'
  
## 2. Rocket Landing
  The code is forked from [IISource](https://github.com/llSourcell/Landing-a-SpaceX-Falcon-Heavy-Rocket) and [EmberceArc](https://github.com/EmbersArc/PPO). With a little modification.
  Use guidence:
  * before run the code, you need install several tools. 
  * these tools are: cmake, swig, box2d, zlib1g-dev, ffmpeg
  * and then pip install gym, and pip install gym\[all]
  
  
