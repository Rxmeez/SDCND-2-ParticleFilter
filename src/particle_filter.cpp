/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Rameez Khan
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

const int NUM_PARTICLES = 100;  // number of particles
static default_random_engine gen;  // Random generator

void ParticleFilter::init(double x, double y, double theta, double std[]) {
   // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
   //   x, y, theta and their uncertainties from GPS) and all weights to 1.
   // Add random Gaussian noise to each particle.
   // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (is_initialized) {
        return;
    }

    // Set a number of particles to be initialized
    num_particles = NUM_PARTICLES;

    // Resize weight vector to be size of NUM_PARTICLES
    weights = vector<double>(num_particles);

    // Gaussian distribution for x, y, and theta (std[x, y, theta])
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Initializes num_particles randomly from the Gaussian distribution
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y= dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0f;

        particles.push_back(p);
        weights.push_back(1.0f);
    }
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

    default_random_engine gen;
    for (int i = 0; i < num_particles; i++) {

        double prediction_x;
        double prediction_y;
        double prediction_theta;

        // Predict each particle with velocity and yaw_rate at delta_t
        if (fabs(yaw_rate) < 1e-10) {
            // check if yaw_rate == 0, where the car is moving straight
            prediction_x = (velocity * delta_t * cos(particles[i].theta)) + particles[i].x;
            prediction_y = (velocity * delta_t * sin(particles[i].theta)) + particles[i].y;
            prediction_theta = particles[i].theta;
        }
        else {
            // yaw_rate != 0, where the car is turning
            prediction_x = ((velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta))) + particles[i].x;
            prediction_y = ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t))) + particles[i].y;
            prediction_theta = (yaw_rate * delta_t) + particles[i].theta;
        }

        normal_distribution<double> dist_x(prediction_x, std_pos[0]);
        normal_distribution<double> dist_y(prediction_y, std_pos[1]);
        normal_distribution<double> dist_theta(prediction_theta, std_pos[2]);

        // Add noise
        particles[i].x =  dist_x(gen);
        particles[i].y =  dist_y(gen);
        particles[i].theta =  dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i=0; i < observations.size(); i++) {
        int map_id = -1;
        double max_distance = numeric_limits<double>::max();  // Large Number

        for (int j=0; j < predicted.size(); j++) {
            double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

            if (distance < max_distance) {
                map_id = j; // store index of landmark instead of id field)
                max_distance = distance;
            }
        }
        observations[i].id = map_id;
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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


    double std_xx = std_landmark[0] * std_landmark[0];
    double std_yy = std_landmark[1] * std_landmark[1];
    double std_xy = std_landmark[0] * std_landmark[1];

    for (int i = 0; i < num_particles; i++) {

        // Transform observations to map's coordinate
        auto p = particles[i];
        vector<LandmarkObs> obs_transformed; // transformed observation vector
        LandmarkObs obs;

        for (int j = 0; j < observations.size(); j++) {
            obs = observations[j];

            LandmarkObs t_obs;  //  single transformed observation
            t_obs.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            t_obs.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
            t_obs.id = obs.id;

            obs_transformed.push_back(t_obs);
        }

        // weights are 1.0
        p.weight = 1.0f;
        weights[i] = 1.0f;

        // Map landmarks within the sensor range
        vector<LandmarkObs> landmarks_within_range;
        for (int m = 0;  m < map_landmarks.landmark_list.size(); m++) {

            auto lm = map_landmarks.landmark_list[m];

            LandmarkObs lm_prediction;
            lm_prediction.x = lm.x_f;
            lm_prediction.y = lm.y_f;
            lm_prediction.id = lm.id_i;

            double distance = dist(lm_prediction.x, lm_prediction.y, p.x, p.y);

            // check if in sensor range
            if (distance <= sensor_range) {
                landmarks_within_range.push_back(lm_prediction);
            }
        }

        // Landmarks near to landmark observations
        dataAssociation(landmarks_within_range, obs_transformed);

        // Compute Multivariate Gaussian probability
        double w = 1;

        for (int t = 0; t < obs_transformed.size(); t++) {
            int oid = obs_transformed[t].id;
            double ox = obs_transformed[t].x;
            double oy = obs_transformed[t].y;

            double predicted_x = landmarks_within_range[oid].x;
            double predicted_y = landmarks_within_range[oid].y;

            double x_diff_square = pow((ox - predicted_x), 2);
            double y_diff_square = pow((oy - predicted_y), 2);

            double gauss_norm = 1 / (2 * M_PI * std_xy);
            double exponent = x_diff_square / (2 * std_xx) + y_diff_square / (2 * std_yy);
            double prob = gauss_norm * exp(-exponent);

            w *= prob;
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

    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    int index = uniintdist(gen);
    double max_weight = *max_element(weights.begin(), weights.end());

    // uniform random distribution
    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;
    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    //   associations: The landmark id that goes along with each listed association
    //   sense_x: the associations x mapping already converted to world coordinates
    //   sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;

}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
