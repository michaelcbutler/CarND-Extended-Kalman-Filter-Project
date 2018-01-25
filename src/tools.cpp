#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
							  const vector<VectorXd> &ground_truth)
{
	VectorXd rmse(4);
	rmse << 1000, 1000, 1000, 1000; // init to "bad" value

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() == 0)
		cout << "CalculateRMSE(): empty estimations vector" << endl;
	else if (estimations.size() != ground_truth.size())
		cout << "CalculateRMSE(): inconsistent vector size" << endl;
	else
	{

		// accumulate squared residuals
		rmse << 0, 0, 0, 0;
		for (int i = 0; i < estimations.size(); ++i) {
			VectorXd diff = estimations[i] - ground_truth[i];
			VectorXd diff_sq = diff.array() * diff.array();
			rmse += diff_sq;
		}

		// calculate the mean
		rmse *= 1.0 / estimations.size();

		// calculate the square root
		rmse = rmse.array().sqrt();
	}

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{

	const float TOO_SMALL = 0.0001; // TODO: maybe better choice here

	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	float rho_mag_sq = px*px + py*py;
	if (rho_mag_sq < TOO_SMALL) {
		cout << "CalculateJacobian(): ill-conditioned matrix" << endl;
		return Hj; // TODO: maybe better choice here
	}

	//compute the Jacobian matrix
	float rho_mag = std::sqrt(rho_mag_sq);
	float rho_mag_3by2 = rho_mag_sq * rho_mag / 2;
	Hj << 	px/rho_mag, py/rho_mag, 0.0, 0.0,
			-py/rho_mag_sq, px/rho_mag_sq, 0.0, 0.0,
			py*(vx*py - vy*px)/rho_mag_3by2, px*(vy*px - vx*py)/rho_mag_3by2, px/rho_mag, py/rho_mag;

	return Hj;
}
