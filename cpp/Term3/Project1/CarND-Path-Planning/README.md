# Generating paths
My implementation is heavily based on the project walk-through in the lessons.
In this write up, I want to capture the interesting parts of what I did over and above that to make things work (apart from stating some essential trajectory generation algorithms - a repetition from what was taught in the walkthrough).

[Here’s a video](https://www.youtube.com/watch?v=Cg6vVRJ0h0M&feature=youtu.be) of my car driving 5 miles without an incident.

   
## Building a reference trajectory
### The general algorithm (picked from the walkthrough) is:
* Pick two points very close to the car to begin with. If the car doesn’t have any points from the past path, use the car’s current s and d and extrapolate in the opposite direction of the heading of the car to obtain a point at a distance s such that s could be traveled at the car’s current speed in 20ms.
* Pick three more points in the middle of the target lane that are s = {30, 60, 90} (when keeping lane) and s = {40, 60, 90} m further ahead (when changing lanes) and in the middle of the target lane while maintaining lanes. For lane change, the discussion follows for how d is chosen in these far away points in a different section.
The different values of far out s’s (in keeping lanes vs changing) help avoid the wobble while keeping the lane vs changing lanes swiftly enough.
* Convert these points to the car frame to make calculations easier (and avoid feeding vertical points to the Spline library).
* Feed these points to the spline library to learn a polynomial trajectory.
* Pick an N such that: (distance the car can travel at the current speed in 20ms) * N = Target distance.
* The target distance (in the direction of the heading of the car) is set at 30 m for when the car is not changing lanes, 15 m otherwise. The corresponding y’s are obtained using spline. This gives us a target for the final position after N movements of the car over the trajectory. The rest is about picking intermediate (x, y) points.
* Pick N x’s equidistant from each other up to the target distance x and use spline to get their corresponding y’s using spline.
* These (x, y) pairs provide a good proxy for  the map coordinates of the trajectory that the car should travel on.
* Convert these points back to the map frame before feeding them to the simulator because that’s what it expects.


## Picking a reference velocity
* If the car is stuck behind a slow vehicle, apart from attempting a lane change, it will incrementally lower the velocity.
* The increments are set at +/- 0.7 m/s per cycle (a cycle is a back and forth between the simulator and my controller). So when the car is behind a slower vehicle, I reduce the velocity until I can increase it back to the max velocity again.
* The max velocity is set at 49.5MPH and the min velocity is set at 30MPH. The value of 30 corresponds to the slowest car I saw on the track - so there was no reason to go slower than the slowest car in this setup.


## Changing Lanes
Some interesting points:
* Although I pick the middle of the target lane as reference points fed to Spline while keeping lane - I pick slightly nuanced points while executing a lane change.
* Of the 3 far away points (apart from the first two that are very close to the car) that I pick, the first one is on the edge of the current lane in the direction of the target lane (d = current lane’s middle + 1.5 m). The second is at an offset of 2.5m from the current lane’s middle point and the third one is in the middle of the target lane - at a s of {40, 60 and 90} respectively.
* If the car in front of the ego car is slower than 50MPH, I attempt a lane change in the left lane and then to the right, whichever fits the criteria (later) in that order.
* **Ensuring when it’s safe to change lanes (lane change criteria):**  
While changing lanes, I ensure that:
    * **Safe gap:** The cars that are ahead and behind me are at least 40 m away in terms of s.
    * **Safe speed** relative to the cars in other lane: The difference between my ego car and the cars in the target lane must not be more than 3m/s. The car behind me must not be faster than 3 m/s and the car ahead of me must not be slower than 3 m/s to reduce the probability of future collisions.
    * **Reducing erraticity:** Sometimes, the car would be in the middle of a lane change but decide that the previous lane was in fact better - thus aborting the lane change and trying to re-enter the previous lane. This led to the triggering “Outside of lane” warning.  
    To avoid this, I check that the car hasn’t *started* execution of lane change in the last 200 cycles (back and forth between the simulator and my controller).