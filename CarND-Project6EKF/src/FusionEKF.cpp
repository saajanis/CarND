#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  /**
    TODO:
      * Finish initializing the FusionEKF.
      * Set the process and measurement noises
    */
  // initialize state and measurement transition
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
	    0, 1, 0, 1,
	    0, 0, 1, 0,
	    0, 0, 0, 1;
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
			  0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);

  // initializing measurement covariance matrices - laser and radar
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
          0, 0.0225;
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
          0, 0.0009, 0,
          0, 0, 0.09;

  //state covariance matrix P
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
        0, 0, 1000, 0,
	    0, 0, 0, 1000;

  //process covariance matrix Q
  Q_ = MatrixXd(4, 4);

  //set the acceleration noise components
  double noise_ax = 9;
  double noise_ay = 9;
  Qv_ = MatrixXd(2, 2);
  Qv_ << noise_ax, 0.,
		  0.,       noise_ay;



}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF first measurement: " << endl;



    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      VectorXd xInitial = VectorXd(4);
      xInitial = tools.PolarToCartesian(xInitial);
      ekf_.Init(xInitial, F_, P_, Q_, H_laser_, Hj_, R_laser_, R_radar_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      VectorXd xInitial = VectorXd(4);
      xInitial << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

      ekf_.Init(xInitial, F_, P_, Q_, H_laser_, Hj_, R_laser_, R_radar_);
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

    if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        return;
    }
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  //1. Modify the F matrix so that the time is integrated
  ekf_.F_ << 1, 0, dt, 0,
  	    0, 1, 0,      dt,
  	    0, 0, 1,      0,
  	    0, 0, 0,      1;

  //2. Set the process covariance matrix Q
  float dt2Div2 = (dt * dt) / 2.;
  MatrixXd G = MatrixXd(4, 2);
  G <<   dt2Div2, 0.,
    	 0.,      dt2Div2,
  		 dt,      0.,
  	     0.,      dt;

  ekf_.Q_ = G * Qv_ * G.transpose();

  if(dt > 0.001) {
    ekf_.Predict();
  }


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	 ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
