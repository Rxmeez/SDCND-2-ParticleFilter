/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang,
 *							Rameez Khan
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Set a number of particles to be initialized
    num_particles = 1;

    // Resize particle and weights vector to be the same as num_particles
    particles.resize(num_particles);  // particles
    //weights.resize(num_particles);  // weights

    // Gaussian distribution for x, y, and theta (std[x, y, theta])
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Random generator
    default_random_engine gen;

    // Initializes num_particles randomly from the Gaussian distribution
    for (int i=0; i < num_particles; i++) {

      particles[i].id = i;
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
      particles[i].weight = 1.0;

      weights.push_back(particles[i].weight);
    }

    // Initialized, so show as true
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


    // xf = x0 + velocity/yaw_rate[sin(yaw_rate(0) + yaw_rate(dt) - sin(yaw_rate(0)))]
    // yf = y0 + velocity/yaw_rate[cos(yaw_rate(0)) - cos(yaw_rate(0) + yaw_rate(dt))]
    // yaw_rate(f) = yaw_rate(0) + yaw_rate(dt)

    // Random generator
    default_random_engine gen;

    // Gaussian distribution for x, y, and theta (std_pos[x, y, theta])
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // Predict each particle with velocity and yaw_rate at delta_t
    for (int i=0; i < num_particles; i++) {

        // yaw_rate != 0, so it doesnt create error for division by zero
        //cout << abs(yaw_rate) << endl;
        if (abs(yaw_rate) != 0) {

            // Measurements for each particle
            particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t) - sin(particles[i].theta)));
            particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta - cos(particles[i].theta + (yaw_rate * delta_t))));
            particles[i].theta += (yaw_rate * delta_t);

        } else {

            // Where yaw_rate == 0;
            particles[i].x += (velocity * delta_t * cos(particles[i].theta));
            particles[i].y += (velocity * delta_t * sin(particles[i].theta));
            particles[i].theta = particles[i].theta;

        }

        // Add random gaussian noise to the predictions
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i=0; i < observations.size(); i++){

    int current_id = -1;
    double smallest_error = INFINITY;  // Large Number so future errors will be smaller

    for (int j=0; j < predicted.size(); j++){
      // difference between predicted and each observations
      double dx = predicted[j].x - observations[i].x;
      double dy = predicted[j].y - observations[i].y;
      double error = dx*dx + dy*dy;

      if (error < smallest_error){
        // nearby id
        current_id = j;
        smallest_error = error;
      }
    }
    //cout << current_id << endl;
    observations[i].id = current_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


  // P(x,y) = 1/(2*(pi)*std[x]*std[y]) e^ -((x-ux)^2/(2*std[x]^2) + (y-uy)^2/(2*std[y]^2))
  // where ux and uy are coordinates of the nearest landmarks

  const double stdx = std_landmark[0];
  const double stdy = std_landmark[1];
  const double gauss_norm = 1.0 / (2.0 * M_PI * stdx *stdy);
  const double n_x = 1.0 / (2.0 * stdx * stdx);
  const double n_y = 1.0 / (2.0 * stdy * stdy);


  // Transform each observations to map coordinates for each particle
  for (int i=0; i < num_particles; i++) {

    vector<LandmarkObs> pred_landmarks;
    vector<LandmarkObs> transformed_obs;

    for (int j=0; j < observations.size(); j++) {

    const double trans_x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
    const double trans_y = particles[i].y + observations[j].y * cos(particles[i].theta) + observations[j].x * sin(particles[i].theta);

    cout << "ID: " << observations[j].id << endl;
    cout << "x: " << trans_x << endl;
    cout << "y: " << trans_y << endl;
    LandmarkObs observation = {observations[j].id, trans_x, trans_y};
    transformed_obs.push_back(observation);
    }

    // Map landmarks within the sensor range
    for (int j=0; j < map_landmarks.landmark_list.size(); j++) {

      int map_id = map_landmarks.landmark_list[j].id_i;
      double map_x = map_landmarks.landmark_list[j].x_f;
      double map_y = map_landmarks.landmark_list[j].y_f;

      double dx = map_x - particles[i].x;
      double dy = map_y - particles[i].y;
      double error = sqrt((dx*dx) + (dy*dy));


      if (error < sensor_range) {
        LandmarkObs prediction = {map_id, map_x, map_y};
        pred_landmarks.push_back(prediction);
      }
    }

    // Landmarks near to landmark observations
    //cout << pred_landmarks.size() << " " << transformed_obs.size() << endl;
    dataAssociation(pred_landmarks, transformed_obs);

    // Comparison between observation vehicle and particles to update particle weight
    double w = 1.0;

    for (int j=0; j < transformed_obs.size(); j++) {

      int obs_id = transformed_obs[j].id;
      cout << "Obs_id: " << obs_id << endl;
      cout << "Pred landmark: " << pred_landmarks[obs_id].x << endl;

      double ux = 0; //pred_landmarks[obs_id].x;  // ux
      double uy = 0; //pred_landmarks[obs_id].y;  // uy

      double diff_x = transformed_obs[j].x - ux;
      double diff_y = transformed_obs[j].y - uy;

      double exponent = (diff_x * diff_x * n_x) + (diff_y * diff_y * n_y);
      double r = gauss_norm * exp(-exponent);
      w *= r;
    }
    particles[i].weight = w;
    weights[i] = w;
  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // New particles vector
    vector<Particle> new_particles;
    new_particles.resize(num_particles);

    // Discrete Distribution to return particles by weight
    default_random_engine gen;

    for (int i=0; i < num_particles; i++){

        discrete_distribution<int> index(weights.begin(), weights.end());
        new_particles[i] = particles[index(gen)];

    }

    // Replace old particles with the resampled particles
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
