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
FusionEKF::FusionEKF()
{
	is_initialized_ = false;

	//create a 4D state vector, we don't know yet the values of the x state
	ekf_.x_ = VectorXd(4);

	// create matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);

	// init measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
				0, 0.0225;

	// init measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
				0, 0.0009, 0,
				0, 0, 0.09;

	// init measurement matrix - laser
	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

// normalize angle to range [-pi, pi]
inline float normalize_range(float phi)
{
	while (phi > M_PI)
		phi -= 2*M_PI;
	while (phi < -M_PI)
		phi += 2*M_PI;
	return phi;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
   /*****************************************************************************
   *  Initialization
   ****************************************************************************/
	if (!is_initialized_)
	{
	  /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
      */
		// first measurement
		ekf_.x_ = VectorXd(4); 		// state vector

		ekf_.P_ = MatrixXd(4, 4);	// state covariance matrix
		ekf_.P_ << 	1, 0, 0, 0,
			  		0, 1, 0, 0,
			  		0, 0, 1000, 0,
			  		0, 0, 0, 1000;

		ekf_.F_ = MatrixXd(4, 4);	// state transition matrix
		ekf_.F_ << 	1, 0, 1, 0,		// initialize constant terms
			  		0, 1, 0, 1,
			  		0, 0, 1, 0,
			  		0, 0, 0, 1;

		ekf_.Q_ = MatrixXd(4, 4);	// process covariance matrix

		previous_timestamp_ = measurement_pack.timestamp_;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			/**
     		 Convert radar from polar to cartesian coordinates and initialize state.
      		*/
			float rho = measurement_pack.raw_measurements_[0];
			float phi = measurement_pack.raw_measurements_[1];
			float rho_dot = measurement_pack.raw_measurements_[2];

			// normalize phi to range [-pi, pi]
			phi = normalize_range(phi);

			float px = rho * cos(phi);
			float py = rho * sin(phi);

			// approximate (vx, vy) assuming phi_dot = 0
			float vx = rho_dot * cos(phi);
			float vy = rho_dot * sin(phi);

			ekf_.x_ << px, py, vx, vy;
		}
		else {
			//set the state with the initial location and zero velocity
			assert(measurement_pack.sensor_type_ == MeasurementPackage::LASER);
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
   *  Prediction
   ****************************************************************************/
	/**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

	//compute the time elapsed between the current and previous measurements
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	const float noise_ax = 9.0;
	const float noise_ay = 9.0;

	float dt2 = dt*dt;
	float dt3by2 = 0.5*(dt2*dt);
	float dt4by4 = 0.25*(dt2*dt2);
	ekf_.Q_ << 	dt4by4*noise_ax, 0, dt3by2*noise_ax, 0,
				0, dt4by4*noise_ay, 0, dt3by2*noise_ay,
				dt3by2*noise_ax, 0, dt2*noise_ax, 0,
				0, dt3by2*noise_ay, 0, dt2*noise_ay;

	ekf_.Predict();

	/*****************************************************************************
   *  Update
   ****************************************************************************/

	/**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     */

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		ekf_.R_ = R_radar_;
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	}
	else {
		// Laser updates
		ekf_.R_ = R_laser_;
		ekf_.H_ = H_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}
}
