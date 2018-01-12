#include "PID.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <list>
#include <numeric>
#include <iomanip>

using namespace std;

const int NUM_FRAMES_FOR_AVERAGE = 5;
std::list<double> PAST_STEERING_ANGLES = {};
const double MAX_CTE = 1.3;
const double MAX_P_VAL = 0.4;
const double MAX_D_VAL = 30;
const double MAX_I_VAL = 0.1;

const double REWARD_FACTOR[] = {1.1, 1.1, 1.1};
const double PENALTY_FACTOR[] = {0.9, 0.9, 0.9};


/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

/*
 * Initialize PID.
 */
void PID::Init(double Kp, double Kd, double Ki, double maxVal) {
    this->P[0] = Kp;
    this->P[1] = Kd;
    this->P[2] = Ki;

    this->dP[0] = 1.0;
    this->dP[1] = 1.0;
    this->dP[2] = 0.002;

    this->maxVal = maxVal;

    int_cte = 0;
    prev_cte = 0.0;
    prev_update_cte = 10000000000.0;
    prev_steering_val = 0.0;
    cumulative_steering_val = 0.0;
    err = 0.0;

    last_component_index = 0;

    was_last_try_addition = false;
}

/*
 * Update the PID error variables given cross track error.
 */
void PID::UpdateError(double cte) {
    //if (!TotalError() < 0.0001) { // less than threshold
        // See what happened due to prev update
        //if(pow(prev_update_cte, 2) < pow(cte, 2)) { // what we did in last try didn't improve things, so revert that change
        if(abs(prev_update_cte) < abs(cte)) {
            if (was_last_try_addition) {
                // revert addition for last_component_index
                this->P[last_component_index] -= this->dP[last_component_index];

                // try subtraction for last_component_index
                this->P[last_component_index] -= this->dP[last_component_index];
                was_last_try_addition = false;
            } else {
                // revert subtraction for last_component_index
                this->P[last_component_index] += this->dP[last_component_index];

                // penalize error var for last_component_index since neither addition nor subtraction worked
                this->dP[last_component_index] *= PENALTY_FACTOR[last_component_index];

                // advance last_component_index
                last_component_index = fmod(last_component_index + 1, 3);

                // try addition for the next component
                this->P[last_component_index] += this->dP[last_component_index];
                was_last_try_addition = true;
            }
        } else { // reward, move to next component and try addition
            // reward error var for last_component_index
            this->dP[last_component_index] *= REWARD_FACTOR[last_component_index];

            // advance last_component_index
            last_component_index = fmod(last_component_index + 1, 3);

            // try addition for the next component
            this->P[last_component_index] += this->dP[last_component_index];
            was_last_try_addition = true;
        }

        // print P and dP along with normalizing
        cout << " ---- P: ";
        for (int i = 0; i < P.size(); i++)
        {
            switch (i) {
                case 0: // P
                    if (this->P[i] > MAX_P_VAL) { this->P[i] = MAX_P_VAL; }
                    if (this->P[i] < -MAX_P_VAL) { this->P[i] = -MAX_P_VAL; }
                    break;
                case 1: // D
                    if (this->P[i] > MAX_D_VAL) { this->P[i] = MAX_D_VAL; }
                    if (this->P[i] < 0) { this->P[i] = 0; }
                    break;
                case 2: // I
                    if (this->P[i] > MAX_I_VAL) { this->P[i] = MAX_I_VAL; }
                    if (this->P[i] < 0) { this->P[i] = 0; }
                    break;
            }

            cout << std::fixed << std::setprecision(2) << this->P[i] << " ";
        }
        cout << " ---- dP: ";
        for (int i = 0; i < this->dP.size(); i++)
        {
            switch (i) {
                case 0: // P
                    if (this->dP[i] > MAX_P_VAL) { this->dP[i] = MAX_P_VAL; }
                    if (this->dP[i] < -MAX_P_VAL) { this->dP[i] = -MAX_P_VAL; }
                    break;
                case 1: // D
                    if (this->dP[i] > MAX_D_VAL) { this->dP[i] = MAX_D_VAL; }
                    if (this->dP[i] < 0) { this->dP[i] = 0; }
                    break;
                case 2: // I
                    if (this->dP[i] > MAX_I_VAL) { this->dP[i] = MAX_I_VAL; }
                    if (this->dP[i] < 0) { this->dP[i] = 0; }
                    break;
            }
            cout << std::fixed << std::setprecision(2) << this->dP[i] << " ";
        }
        //cout << std::endl << std::endl;

        prev_update_cte = cte;
    //}
}

/*
 * Calculate the total PID error.
 */
double PID::TotalError() {
    return this->dP[0] + this->dP[1] + this->dP[2];
}

double PID::Run(double cte) {
//    // normalize cte to [-1, 1]
//    double range = 12 - (-12);
//    cte = (cte - (-12)) / range;
//    double range2 = 1 - (-1);
//    cte = (cte * range2) + (-1);
    if(cte > MAX_CTE) {cte = MAX_CTE;}
    if(cte < -MAX_CTE) {cte = -MAX_CTE;}

    double diff_cte = cte - prev_cte;
    prev_cte = cte;
    int_cte = int_cte * 0.9 + cte;
    if(int_cte > MAX_CTE) {int_cte = MAX_CTE;}
    if(int_cte < -MAX_CTE) {int_cte = -MAX_CTE;}
    double steering_angle = -(this->P[0] * cte) - (this->P[1] * diff_cte) - (this->P[2] * int_cte);

    UpdateError(cte);
    //std::cout << "CTE: " << cte << endl;
    std::cout << " --- Suggested steering: " << std::fixed << std::setprecision(2) << cte << " " << -(this->P[0] * cte) << " " << - (this->P[1] * diff_cte) << " " << - (this->P[2] * int_cte) << " " << steering_angle << endl;
//    if(temp_steering_val > 0.3) {temp_steering_val = 0.3;}
//    if(temp_steering_val < -0.3) {temp_steering_val = -0.3;}
//    while(temp_steering_val > 1.0) {temp_steering_val = temp_steering_val/10;}
//    while(temp_steering_val < -1.0) {temp_steering_val = temp_steering_val/10;}

//    if (NUM_CTES_ELAPSED >= NUM_CTES_BEFORE_UPDATE - 1 ) {
//        NUM_CTES_ELAPSED = 0;
//
//        UpdateError(cte);
//        std::cout << "CTE: " << cte << endl;
//        std::cout << "DIFF CTE: " << diff_cte << endl;
//
//
//        cumulative_steering_val += temp_steering_val;
//        prev_steering_val = cumulative_steering_val / NUM_CTES_BEFORE_UPDATE; // average
//        cumulative_steering_val = 0.0;
//
//
//        //prev_steering_val = temp_steering_val;
//        std::cout << "Suggested steering: " << prev_steering_val;
//        return prev_steering_val;
//    } else {
//        NUM_CTES_ELAPSED++;
//
//        cumulative_steering_val += temp_steering_val;
//
//        return prev_steering_val;
//    }

    if (PAST_STEERING_ANGLES.size() > NUM_FRAMES_FOR_AVERAGE) {
        PAST_STEERING_ANGLES.pop_back();
    }

    PAST_STEERING_ANGLES.push_front(steering_angle);
    return std::accumulate(PAST_STEERING_ANGLES.begin(), PAST_STEERING_ANGLES.end(), 0.0) / PAST_STEERING_ANGLES.size();
}

