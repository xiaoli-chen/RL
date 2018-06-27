# ![GIF](https://thumbs.gfycat.com/CoarseEmbellishedIsopod-max-14mb.gif)
[Click here for higher quality video](https://gfycat.com/CoarseEmbellishedIsopod)

RL algorithm and working model: https://github.com/EmbersArc/PPO

# OpenAI Gym with custom SpaceX Rocket Lander environment for box2d  

[/envs/box2d/rocket_lander.py](https://github.com/EmbersArc/gym/blob/master/envs/box2d/rocket_lander.py) is the only important file, along with some small changes to init files, see https://github.com/openai/gym#environments

The objective of this environment is to land a rocket on a ship. The environment is highly customizable and takes discrete or continuous inputs.

### STATE VARIABLES  
The state consists of the following variables:
  * x position  
  * y position  
  * angle  
  * first leg ground contact indicator  
  * second leg ground contact indicator  
  * throttle  
  * engine gimbal  
  
If VEL_STATE is set to true, the velocities are included:  
  * x velocity  
  * y velocity  
  * angular velocity  
  
All state variables are normalized for improved training.
    
### CONTROL INPUTS  
Discrete control inputs are:  
  * gimbal left  
  * gimbal right  
  * throttle up  
  * throttle down  
  * use first control thruster  
  * use second control thruster  
  * no action  
    
Continuous control inputs are:  
  * gimbal (left/right)  
  * throttle (up/down)  
  * control thruster (left/right)  
