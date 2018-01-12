#ifndef PID_H
#define PID_H

#include <array>

class PID {
public:
  /*
  * Errors
  */
  std::array<double, 3> dP;

  /*
  * Coefficients
  */
  std::array<double, 3> P;

  double maxVal;

  double int_cte;
  double prev_cte;
  double prev_update_cte;
  double prev_steering_val;
  double cumulative_steering_val;
  double err;

  int last_component_index;
  bool was_last_try_addition;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Kd, double Ki, double maxVal);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();


  double Run(double cte);
};

#endif /* PID_H */
