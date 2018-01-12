 /*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
  *     Modified: Saajan Shridhar
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <random> // Need this for sampling from distributions
#include <sstream>
#include <string>
#include <iterator>
#include "Eigen/Dense"

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


    // TODO: Reconsider
    num_particles = 200;
    particles = std::vector<Particle>();
    weights = std::vector<double>();

    // This creates a normal (Gaussian) distribution for x, y and theta.
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);


    for (int i = 0; i < num_particles; i++) {
        double sample_x, sample_y, sample_theta;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);
        while (sample_theta> M_PI) sample_theta-=2.*M_PI;
        while (sample_theta<-M_PI) sample_theta+=2.*M_PI;

        Particle current_particle =
                {i, sample_x, sample_y, sample_theta, /* weight */ 1.0,
                 /* associations */ vector<int>(), /* sense_x */ vector<double>(),
                 /* sense_y */ vector<double>()};
        particles.push_back(current_particle);

        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // This creates a normal (Gaussian) distribution for x, y and theta.
    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double yaw = particles[i].theta;

        double px_p, py_p, yaw_p;

        // Predict
        // avoid division by zero
        if (fabs(yaw_rate) > 0.001) {
            px_p = p_x + velocity/yaw_rate * ( sin(yaw + yaw_rate*delta_t) - sin(yaw));
            py_p = p_y + velocity/yaw_rate * ( cos(yaw) - cos(yaw+yaw_rate*delta_t) );
        }
        else {
            px_p = p_x + velocity*delta_t*cos(yaw);
            py_p = p_y + velocity*delta_t*sin(yaw);
        }
        yaw_p = yaw + yaw_rate*delta_t;

        // Add noise
        px_p = px_p + dist_x(gen);
        py_p = py_p + dist_y(gen);
        yaw_p = yaw_p + dist_theta(gen);

        particles[i].x = px_p;
        particles[i].y = py_p;
        particles[i].theta = yaw_p;
    }
}

 vector<LandmarkObs> ParticleFilter::dataAssociation(
         std::vector<LandmarkObs> transformed_measurements, const std::vector<LandmarkObs>& true_landmarks) {
     // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
     //   observed measurement to this particular landmark.
     // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
     //   implement this method and use it as a helper during the updateWeights phase.

     // TODO: Reconsider: Why are they calling them 'predicted'? - (I'm using the ground truth)

     // Landmarks closest to observations in order of observations
     vector<LandmarkObs> associated_true_landmarks; associated_true_landmarks.clear();
     associated_true_landmarks = vector<LandmarkObs>();
     // go through observations and find the closest true landmark for each
     for (int i = 0; i < transformed_measurements.size(); i++) {
         int closest_landmark_index = -1;
         double closest_distance = 100000000.0;
         for (int j = 0; j < true_landmarks.size(); j++) {
             double current_dist = dist(transformed_measurements[i].x, transformed_measurements[i].y,
                                        true_landmarks[j].x, true_landmarks[j].y);

             if (current_dist < closest_distance) {
                 closest_landmark_index = j;
                 closest_distance = current_dist;
             }
         }

         associated_true_landmarks.push_back(true_landmarks[closest_landmark_index]);
     }

     return associated_true_landmarks;
 }

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	//   NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


    // Go over each particle and find it's weight
    for (int i = 0; i < num_particles; i++) {
        // Extract variables for this particle
        double x_part= particles[i].x;
        double y_part= particles[i].y;
        double theta_part= particles[i].theta;

        // TODO: Reconsider: This can be moved to outside the loop
        // Convert map_landmarks to LandmarkObs
        // Also prune the ones not in range
        vector<LandmarkObs> true_landmarks; true_landmarks.clear();
        true_landmarks = vector<LandmarkObs>();
        for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
            // Skip if this landmark is not in the range of the sensor
            double current_dist = dist(x_part, y_part,
                                       map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);
            if (current_dist > sensor_range) {
                continue;
            }

            LandmarkObs true_landmark;
            true_landmark.id = map_landmarks.landmark_list[i].id_i;
            true_landmark.x = map_landmarks.landmark_list[i].x_f;
            true_landmark.y = map_landmarks.landmark_list[i].y_f;

            true_landmarks.push_back(true_landmark);
        }

        // transform observations (sensor measurements) from car's coordinates to map's coordinates wrt to the particle
        vector<LandmarkObs> transformed_measurements; transformed_measurements.clear();
        transformed_measurements = vector<LandmarkObs>();
        for (int i = 0; i < observations.size(); i++) {
            // Use stripped down version of homogeneous transformation to transform
            LandmarkObs transformed_measurement = LandmarkObs();

            transformed_measurement.id = observations[i].id;
            transformed_measurement.x =
                    x_part + (cos(theta_part) * observations[i].x) - (sin(theta_part) * observations[i].y);
            transformed_measurement.y =
                    y_part + (sin(theta_part) * observations[i].x) + (cos(theta_part) * observations[i].y);

            transformed_measurements.push_back(transformed_measurement);
        }

        // list of landmarks closest to each of our sensor measurements (in order)
        vector<LandmarkObs> associated_true_landmarks = dataAssociation(transformed_measurements, true_landmarks);

        // Set associations for debugging
        vector<int> associations; associations.clear();
        associations = vector<int>();
        vector<double> sense_x; sense_x.clear();
        sense_x = vector<double>();
        vector<double> sense_y; sense_y.clear();
        sense_y = vector<double>();
        for (int x = 0; x < associated_true_landmarks.size(); x++) {
            associations.push_back(associated_true_landmarks[x].id);
            sense_x.push_back(transformed_measurements[x].x);
            sense_y.push_back(transformed_measurements[x].y);
        }
        particles[i].associations = associations;
        particles[i].sense_x = sense_x;
        particles[i].sense_y = sense_y;

        // calculate likelihood of this set of observations given this particle
        double final_weight = 1.0;
        for (int i = 0; i < transformed_measurements.size(); i++) {
            // TODO: Reconsider not reusing landmarks once associated? -- no need, they had duplicates in the quiz (L2, L2)
            LandmarkObs transformed_measurement = transformed_measurements[i];
            // Check that there is an associated landmark for this measurement
            if (i > associated_true_landmarks.size() - 1){
                continue;
            }
            LandmarkObs associated_true_landmark = associated_true_landmarks[i];



            // TODO: Reconsider See if these calculations are correct -- they do give reasonable values
            // Get probability for this measurement
            double sig_x= std_landmark[0];
            double sig_y= std_landmark[1];
            double x_obs= transformed_measurement.x;
            double y_obs= transformed_measurement.y;
            double mu_x= associated_true_landmark.x;
            double mu_y= associated_true_landmark.y;

            // calculate normalization term
            Eigen::VectorXd x = Eigen::VectorXd(2);
            x << x_obs, y_obs;
            Eigen::VectorXd mu = Eigen::VectorXd(2);
            mu << mu_x, mu_y;
            Eigen::MatrixXd cov = Eigen::MatrixXd(2,2);
            cov << pow(sig_x,2), 0.0,
                    0.0, pow(sig_y,2);

            double weight = getMultivariateProbability(x, mu, cov);

            final_weight *= weight;
        }

        particles[i].weight = final_weight;
        weights[i] = final_weight;
    }

//    // normalize weights
//    double sum_weights = 0.0;
//    for (int i = 0; i < weights.size(); i++) {
//        sum_weights += weights[i];
//    }
//    for (int i = 0; i < weights.size(); i++) {
//        weights[i] /= sum_weights;
//        particles[i].weight /= sum_weights;
//    }
}

void ParticleFilter::resample() {
    // Set of new particles
    std::vector<Particle> resampled_particles; resampled_particles.clear();
    resampled_particles = vector<Particle>();

    // TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    for (int i = 0; i < particles.size(); i++) {
        int sampled_index = d(gen);
        resampled_particles.push_back(particles[sampled_index]);
    }

    particles.clear();
    particles = resampled_particles;

    // normalize weights again (using particle.weight)
    double sum_weights = 0.0;
    for (int i = 0; i < particles.size(); i++) {
        sum_weights += particles[i].weight;
    }
    for (int i = 0; i < particles.size(); i++) {
        weights[i] /= sum_weights;
        particles[i].weight /= sum_weights;
    }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


 double ParticleFilter::getMultivariateProbability(const Eigen::VectorXd &x, const Eigen::VectorXd &meanVec, const Eigen::MatrixXd &covMat)
 {
     // Got it from: https://stackoverflow.com/questions/41538095/evaluate-multivariate-normal-gaussian-density-in-c
     // avoid magic numbers in your code. Compilers will be able to compute this at compile time:
     const double logSqrt2Pi = 0.5*std::log(2*M_PI);
     typedef Eigen::LLT<Eigen::MatrixXd> Chol;
     Chol chol(covMat);
     // Handle non positive definite covariance somehow:
     if(chol.info()!=Eigen::Success) throw "decomposition failed!";
     const Chol::Traits::MatrixL& L = chol.matrixL();
     double quadform = (L.solve(x - meanVec)).squaredNorm();
     return std::exp(-x.rows()*logSqrt2Pi - 0.5*quadform) / L.determinant();
 }
