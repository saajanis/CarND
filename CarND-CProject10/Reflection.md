[//]: # (Image References)

[video1]: ./videos/JustP.mp4 "Video"
[video2]: ./videos/PD.mp4 "Video"
[video3]: ./videos/PDI.mp4 "Video"
[video4]: ./videos/full.mp4 "Video"




### Model description:

This project uses the MPC model consisting of the following major components:

## State:

A in initial state consists of:
- The X position of the car
- The Y position of the car
- The orientation of the car
- The tangential velocity of the car
- The cross-track error
- The error in the orientation of the car


## Actuators:

We feed two actuators - the steering angle and throttle back to the simulator


## Update equations:

The only interesting thing (read different from the quizzes) about the update equations in MPC is that now , for computing cte in the update equation, we use the refernce value of y as given by a polynomial (fitted to reference points) of degree 3 instead of 1.

Correspondingly, the reference psi used to conpute epsi also becomes a derivative of a 3rd degree polynomial.

Both of these are coded in the FG_eval class in MPC.cpp


## Frame:

Note that in this implementation, everything is considered w.r.t. the car frame, where the direction of the heading is the X-axis and Y-axis points to the left of the car.


## Waypoints preprocessing:

The refernce points given by the simulator are converted from world map into the car frame a variation of the homogeneous transform from earlier lessons. The following code in main.cpp help with that:

"""
double xoffset = ptsx[i] - px;
double yoffset = ptsy[i] - py;
ptsx_car.push_back(xoffset * cos(psi) + yoffset * sin(psi));
ptsy_car.push_back(yoffset * cos(psi) - xoffset * sin(psi));
"""


The points are fitted to an polynomial of degree 3 to get the reference trajectory using the polyfit(...) method in main.cpp. 

In the initial state, the X, Y, and psi are all zero (in car frame) and cte and epsi is calculated w.r.t. the reference trajectory considering values X=0 and Y=0 for where the car should ideally be in the car frame. The following equations in main.cpp help with that:

"""
double cte = polyeval(coeffs, 0); // deviation in Y of car from Y=0 at X=0
double epsi = 0 - atan(coeffs[1]); // deviation in orientation of car from psi=0 at X=0
"""


## Timestep, duration:

I used 12 timesteps of 100ms duration generate the predictions by MPC.

These values were mostly found by trial and error. For the most part, the reason I chose dt=0.1 was so I can use the second prediction instead of the first to account for a latency of 100ms. A value of 12 for N gave just enough predictions to fit almost perfectly the close-by points on the reference trajectory on average for most of the track.  A smaller N leads to wrong predictions and a larger N leads to the prediction trajectory having a hard time trying to bend to the curve of the reference trajectory at some points in the track.

A total of N-2 actuations are predicted to minimize the cost of modeling the reference trajectory and the second of those predicted actuations is fed back to the simulator.


## Latency:

Since the timestep is 100ms, just picking the second prediction (instead of the first like in the lesson) should give us the prediction for actuation 100ms later.


## Cost:

It is mostly the same as the ones in the quiz except that I use a product of the steering angle and velocity instead of just steering angle to penalize higher steering angles at high velocities.

Some weights are increased by multiplies where needed. 