[//]: # (Image References)

[video1]: ./videos/JustP.mp4 "Video"
[video2]: ./videos/PD.mp4 "Video"
[video3]: ./videos/PDI.mp4 "Video"
[video4]: ./videos/full.mp4 "Video"




### First, I will discuss the effect that each of P, I and D had on the performance:

I chose to limit the value of CTE provided by the simulator to [-1.3, 1.3]. This is roughly the width of the track.

I also chose to average out the suggested steering angle over the last 5 frames to get smoother change in angles.


## P:

With just P, the car goes around in a wavy motion even on a straight track - with the waves getting bigger as we progress - like I expected it to.

Here's a video of the simulation with just the P controller: 
[P Video](https://youtu.be/64DMrbk8-zU)

## PD: 

For the most part, the car drives just fine, but when it does have to do the corrections, the waviness is more pronounced. It dies out eventually and the car goes back to driving smoothly.

Here's a video of the simulation with P AND D controller: 
[PD Video](https://youtu.be/62o5yCd7P2Y)


## PDI:

With the introduction of the I controller, I see that the corrections (when they need to be done) are a lot more smoother. They form less pronounced waves as compared to the case of P and D controller. 

It does introduce a weird problem where the car needs corrections slightly more often - like the wheels will turn abruptly even when the CTE is quite low.

Here's a video of the simulation with P, I AND D controller: 
[PDI Video](https://youtu.be/DSVv7rh_LA4)


## Full track:
Here's a video of the whole track with P, I AND D controller: 
[Full track Video](https://youtu.be/Xl0FH4SlFS8)



### Approach used for hyperparameter tuning: 

I used Twiddle for hyperparameter tuning. Most of the code for twiddle is implemented in method UpdateError(...).

The code for computing the steering angle is implemented in method Run(...).

I chose a consistent growth factor/pentalty of 1.1 and 0.9 respectively for all controllers.

I used the design where I have a P/I/D that goes into the actual equation for calculating the steering angle and then a dP/dI/dD that helps P/I/D grow or shrink based on the decrease or increase in CTE over consecutive frames.

For the P controller, I chose to limit the value of P and dP between [-0.4, 0.4].
For the D controller, I chose to limit the value of D and dD between [0, 40].
For the I controller, I chose to limit the value of I and dI between [0, 0.1].

In addition, since the cumulative CTE can grow pretty large if I just keep on adding it, I decided to use a moving average calculated as the sum: 0.95*sum(CTE so far) + current_cte.

For each frame's CTE that comes in, the hyperparameters are updated and a new suggested steering estimated based on them after clipping theor values to the respective limits I've decided for each.