#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Tools tools = Tools();

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &F_in, MatrixXd &P_in, MatrixXd &Q_in,
                        MatrixXd &H_laser_in, MatrixXd &H_j_in, MatrixXd &R_laser_in, MatrixXd &R_radar_in) {
  x_ = x_in;//
  F_ = F_in;//
  P_ = P_in;//
  Q_ = Q_in;//
  H_laser_ = H_laser_in;//
  H_j_ = H_j_in;
  R_laser_ = R_laser_in;
  R_radar_ = R_radar_in;

}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	VectorXd z_pred = H_laser_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_laser_.transpose();
	MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
	//VectorXd z_pred = H_ * x_;
	//VectorXd cartesianX_ = tools.PolarToCartesian(z);
	H_j_ = tools.CalculateJacobian(x_);
	VectorXd polarX_ = tools.CartesianToPolar(x_); //h(x)
	VectorXd y = z - polarX_;

    // Normalize
    bool phi_in_range = false;
    while (phi_in_range == false) {
        if(y(1) > 3.14159) {
            y(1) = y(1) - 6.2831;
        } else if (y(1) < -3.14159) {
            y(1) = y(1) + 6.2831;
        } else {
            phi_in_range = true;
        }
    }

	MatrixXd Hjt = H_j_.transpose();
	MatrixXd S = H_j_ * P_ * Hjt + R_radar_;
	MatrixXd Si = S.inverse();
	MatrixXd PHjt = P_ * Hjt;
	MatrixXd K = PHjt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_j_) * P_;
}
