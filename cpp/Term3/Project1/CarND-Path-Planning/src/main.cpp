#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;


int TARGET_LANE = 1;
double REF_VEL = 0;
double VEL_CHANGE = 0.7; // 0.224
double MIN_DIST_TO_NEXT = 40.0;
double FAR_OUT_S_ADDER[] = {30, 60, 90}; // {30, 60, 90}
double FAR_OUT_S_ADDER_LANE_CHANGE[] = {40, 60, 90};
// TODO: Change to min frames before attempting another lane change
int MIN_FRAMES_TO_LANE_REFRESH = 200;
int frame_count = 0;
// Define the actual (x,y) points we will use for the planner
vector<double> next_x_vals;
vector<double> next_y_vals;



// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

            // Define the actual (x,y) points we will use for the planner
            next_x_vals.clear();
            next_y_vals.clear();

            // Generating a path that starts tangent to the car's heading, and extends out up to 90 mts away in Frenet
            vector<double> pts_x;
            vector<double> pts_y;
            double pos_x;
            double pos_y;
            double angle;
            int prev_size = previous_path_x.size();


            // ADJUST REF VEL TO THE CAR(S) BEING FOLLOWED
            bool next_car_too_close = false;
            for (int i = 0; i < sensor_fusion.size(); i++) {
                // whether car is in my lane
                float d = sensor_fusion[i][6];

                if (d < (2 + 4 * TARGET_LANE + 2.5) && d > (2 + 4 * TARGET_LANE - 2.5)) {
                    double vx = sensor_fusion[i][3];
                    double vy = sensor_fusion[i][4];
                    double check_speed = sqrt(vx * vx + vy * vy);
                    double check_car_s = sensor_fusion[i][5];

                    // find final state of the other car after the time period captured by previous_path_*
                    check_car_s += ((double) prev_size * 0.02 * check_speed);
                    // final state of my car after previous_path_* is done, if it exists (current s otherwise)
                    double car_end_s = (prev_size > 0) ? end_path_s : car_s;

                    // check if cars ahead of us have a projected value of s too close to my current pos (?)
                    if ((check_car_s > car_end_s) && (check_car_s - car_end_s) < MIN_DIST_TO_NEXT) {
                        // TODO: Do some logic here.
                        // - Lower the ref velocity so we don't crash into this car in front of us.
                        // - We can also flag to try to change lanes.
                        //REF_VEL = 29.5; //mph
                        next_car_too_close = true;
                    }
                }
        }


        bool eligible_for_lane_change = false;
        if (next_car_too_close && frame_count > MIN_FRAMES_TO_LANE_REFRESH) {
            eligible_for_lane_change = true;
        } else {
            frame_count++;
        }
        // STEP DOWN VELOCITY AND CHANGE LANES
        if (next_car_too_close) {
            if (REF_VEL > 30.0) {
                REF_VEL -= VEL_CHANGE;
            }

            bool is_car_in_left_lane = false;
            bool is_car_in_right_lane = false;
            if (eligible_for_lane_change) {
                // CHANGE LANES
                // TODO: Check if lane is free to go into
                // - check if a car is in that lane within an s range of my car
                // - check for both left and right until success (too_close = false)
                // - compute cost for which lane will be the best in the next 5 secs or so
                // - can pick diff anchor points in the other lane and compute the jerk and reject trajectories / pick
                // the best among them
                for (int i = 0; i < sensor_fusion.size(); i++) {
                    double other_car_s = sensor_fusion[i][5];
                    float other_car_d = sensor_fusion[i][6];
                    double vx = sensor_fusion[i][3];
                    double vy = sensor_fusion[i][4];
                    double other_car_speed = sqrt(vx * vx + vy * vy);

                    float d_diff = other_car_d - car_d;
                    float s_diff = car_s - other_car_s;
                    float s_diff_abs = abs(car_s - other_car_s);
                    bool other_car_too_slow = ((car_speed / 2.24) - other_car_speed) > 3; // m/s
                    bool other_car_too_fast = (other_car_speed - (car_speed / 2.24)) > 3; // m/s

                    if (d_diff > 0 && abs(d_diff) > 2.0 && abs(d_diff) < 6.0) {
                        if (s_diff_abs < 60.0 ||
                            (s_diff < 0 && other_car_too_slow) ||
                            (s_diff > 0 && other_car_too_fast)) {
                            is_car_in_right_lane = true;
                        }
                    }
                    if (d_diff < 0 && abs(d_diff) > 2.0 && abs(d_diff) < 6.0) {
                        if (s_diff_abs < 60.0 ||
                                (s_diff < 0 && other_car_too_slow) ||
                                (s_diff > 0 && other_car_too_fast)) {
                            is_car_in_left_lane = true;
                        }
                    }
                }

                int old_lane = TARGET_LANE;
                // try turning left
                if (TARGET_LANE != 0 && !is_car_in_left_lane) {
                    TARGET_LANE = max(0, TARGET_LANE - 1);
                } else if (TARGET_LANE != 2 && !is_car_in_right_lane) { // try turning right
                    TARGET_LANE = min(2, TARGET_LANE + 1);
                } else {
                    // TODO: Consider Planning for a lane change here (by messing with vel)

                }

                if (TARGET_LANE != old_lane) {
                    frame_count = 0;
                }
            }
        } else if (REF_VEL < 49.5) { // INCREMENT VEL TO SPEED LIMIT OTHERWISE
            REF_VEL += VEL_CHANGE;
        }



        // REFERENCE x, y, yaw states
        // either we will reference the starting point as where the car is or at the previous paths' end point
        double ref_x = car_x;
        double ref_y = car_y;
        double ref_yaw = deg2rad(car_yaw);

        // ADD TWO POINTS - EITHER PREVIOUS PATH'S OR WHIP UP TWO BASED ON CAR'S REF
        // if prev_path is mostly empty, start building from the car's localization data (tangent to heading)
        if (prev_size < 2) {
            // use two points that make the path tangent to the car
            double estimated_prev_car_x = car_x - cos(ref_yaw);
            double estimated_prev_car_y = car_y - sin(ref_yaw);

            pts_x.push_back(estimated_prev_car_x);
            pts_x.push_back(car_x);

            pts_y.push_back(estimated_prev_car_y);
            pts_y.push_back(car_y);
        }
            // else use last two points in the prev_path (tangent to heading)
        else {
            ref_x = previous_path_x[prev_size - 1];
            ref_y = previous_path_y[prev_size - 1];

            double prev_car_x_2 = previous_path_x[prev_size - 2];
            double prev_car_y_2 = previous_path_y[prev_size - 2];
            ref_yaw = atan2(ref_y - prev_car_y_2, ref_x - prev_car_x_2);

            pts_x.push_back(prev_car_x_2);
            pts_x.push_back(ref_x);

            pts_y.push_back(prev_car_y_2);
            pts_y.push_back(ref_y);
        }

        double target_lane_d = 2 + 4 * TARGET_LANE;
        bool is_changing_lane = abs(car_d - target_lane_d) > 0.5;

        // ADD THREE FAR OUT POINTS TO BUILD A SPLINE TOWARDS THE TARGET_LANE
        // In Frenet add evenly spaced points ahead of the starting reference
        vector<double> next_wp0 = getXY(car_s + (is_changing_lane ? FAR_OUT_S_ADDER_LANE_CHANGE[0] : FAR_OUT_S_ADDER[0]),
                                        is_changing_lane ?
                                        target_lane_d > car_d ? car_d + 1.5 : car_d - 1.5 :
                                        target_lane_d,
                                        map_waypoints_s, map_waypoints_x, map_waypoints_y);
        vector<double> next_wp1 = getXY(car_s + (is_changing_lane ? FAR_OUT_S_ADDER_LANE_CHANGE[1] : FAR_OUT_S_ADDER[1]),
                                        is_changing_lane ?
                                        target_lane_d > car_d ? car_d + 2.5 : car_d - 2.8 :
                                        target_lane_d,
                                        map_waypoints_s, map_waypoints_x, map_waypoints_y);
        vector<double> next_wp2 = getXY(car_s + (is_changing_lane ? FAR_OUT_S_ADDER_LANE_CHANGE[2] : FAR_OUT_S_ADDER[2]),
                                        target_lane_d, map_waypoints_s,
                                        map_waypoints_x, map_waypoints_y);

        pts_x.push_back(next_wp0[0]);
        pts_x.push_back(next_wp1[0]);
        pts_x.push_back(next_wp2[0]);

        pts_y.push_back(next_wp0[1]);
        pts_y.push_back(next_wp1[1]);
        pts_y.push_back(next_wp2[1]);

        for (int i = 0; i < pts_x.size(); i++) {
            // MOVING TO CAR FRAME IS IMP - because if the xs represent vertical points in the world coords,
            // multiple cs will just map to the same y val - a problem.
            // Still don't understand why this won't be a problem in the car frame since the car can be oriented
            // vertically wrt to the world too.
            // shift pts so that car is at the reference angle of 0 degrees
            double shift_x = pts_x[i] - ref_x;
            double shift_y = pts_y[i] - ref_y;

            pts_x[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
            pts_y[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
        }

        // create a spline
        // The car can drive in its lane with just the frenet coords - the spline helps with smoothing the motion
        tk::spline s;
        // set (x,y) points to the spline
        s.set_points(pts_x, pts_y);


        // BUILD POINTS WE'LL SEND TO THE CAR
        // start with previous path points from the last time
        for (int i = 0; i < previous_path_x.size(); i++) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
        }

        // -- ADD POINTS AFTER SAMPLING FROM THE SPLINE
        // Calculate how to break up spline points so that we travel at our desired reference velocity
        double target_x = is_changing_lane ? 30.0 : 15.0;
        double target_y = s(target_x);
        double target_dist = sqrt((target_x) * (target_x) + (target_y) * (target_y));

        double x_add_on = 0;

        // Fill up the rest of our path planner after filling it with previous points, here we will always output
        // 50 points.
        for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
            double N = (target_dist / (0.02 * REF_VEL / 2.24));

            double x_point = x_add_on + (target_x) / N;
            double y_point = s(x_point);
            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // rotate back to normal after rotatin it earlier
            // from local coordinates to global coordinates
            x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
            y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));
            // shift
            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
        }





          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds

//            // Straight in Map Coordinates
//            double dist_inc = 0.5;
//            for(int i = 0; i < 50; i++)
//            {
//                next_x_vals.push_back(car_x+(dist_inc*i)*cos(deg2rad(car_yaw)));
//                next_y_vals.push_back(car_y+(dist_inc*i)*sin(deg2rad(car_yaw)));
//            }
//
//
//            // Go in a circle
//            double pos_x;
//            double pos_y;
//            double angle;
//            int path_size = previous_path_x.size();
//
//            for(int i = 0; i < path_size; i++)
//            {
//                next_x_vals.push_back(previous_path_x[i]);
//                next_y_vals.push_back(previous_path_y[i]);
//            }
//
//            if(path_size == 0)
//            {
//                pos_x = car_x;
//                pos_y = car_y;
//                angle = deg2rad(car_yaw);
//            }
//            else
//            {
//                pos_x = previous_path_x[path_size-1];
//                pos_y = previous_path_y[path_size-1];
//
//                double pos_x2 = previous_path_x[path_size-2];
//                double pos_y2 = previous_path_y[path_size-2];
//                angle = atan2(pos_y-pos_y2,pos_x-pos_x2);
//                cout << angle << " " << deg2rad(car_yaw) << "\n";
//            }
//
//            double dist_inc = 0.5;
//            for(int i = 0; i < 50-path_size; i++)
//            {
//                next_x_vals.push_back(pos_x+(dist_inc)*cos(angle+(i+1)*(pi()/100)));
//                next_y_vals.push_back(pos_y+(dist_inc)*sin(angle+(i+1)*(pi()/100)));
//                pos_x += (dist_inc)*cos(angle+(i+1)*(pi()/100));
//                pos_y += (dist_inc)*sin(angle+(i+1)*(pi()/100));
//            }

            // Straight in Frenet Coordinates (follow the road)
//            double dist_inc = 0.4;
//            for(int i = 0; i < 50; i++)
//            {
//                double next_s = car_s + (i + 1) * dist_inc;
//                double next_d = 6;
//                vector<double> nextXY = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
//                next_x_vals.push_back(nextXY[0]);
//                next_y_vals.push_back(nextXY[1]);
//            }

            // TODO(saajan): Consider making point path spacing proportional to car_speed value

            // TODO(saajan): Generate future trajectory
            // - Pick a final x,y (derive from waypoints?)
            // - The waypoint has an x,y - add the dir of right * 2,6,10 for 1,2,3 lane to get the x,y of it's middle
            // - take prev_path up till getXY(end_s, end_d) + final lane (x,y) - pass it to spline to learn a poly
            // - use SPLINE(x) to generate Ys for a set of Xs where Xs are chosen based on how soon the shift has to be
            // executed (each point represents 20ms - use car_speed too).
            // SPLINE.set_points(vector<double>(), vector<double>());

            // TODO(saajan): Change lanes
            // - I think we can just enumerate KL, TL, TR and check for their feasibility and cost (if it leads to a better state) - then pick the best
            // Factors in computing cost:
            // - Optimality - Any time a car in the front has a speed below the speed limit, it's a reason to change lanes.
            // - Feasibility - Make sure the lane is safe and would lead to a more optimal speed (check if points collide in time - use sensor_fusion)
            // - Safety - leads to us being further away from the traffic
            // - Comfort - lane change is not jerky:
            // Check for max/min: long speed, long and lat acc, and
            // max_curvature = tan(steering_max)/L (for every point pair) to make sure sideways jerk is in control
            // Just use a different trajectory generator (quintic poly) for lane change traj. Use sDot and sDoubleDot of other cars(?)
            //







//            // PRINT THE VALUES BEING SENT
//            cout << "next_x_vals: " << "\n";
//            for (int i=0; i < next_x_vals.size(); i++) {
//                cout << to_string(next_x_vals[i]) + ", ";
//            }
//            cout << "\n";
//            cout << "next_y_vals: " << "\n";
//            for (int i=0; i < next_y_vals.size(); i++) {
//                cout << to_string(next_y_vals[i]) + ", ";
//            }
//            cout << "\n\n\n";







          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
