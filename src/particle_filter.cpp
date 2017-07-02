/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang templates

 *  Created on: Jul 01, 2017
 *      Author: Philippe Weingertner implementation
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

default_random_engine gen;
normal_distribution<double> distn(0, 1); // normal distribution


double normpdfxy(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {
  double cx = (x - mu_x) / std_x;
  double cy = (y - mu_y) / std_y;
  double alpha = 1 / (2 * M_PI * std_x * std_y);

  return alpha * exp(-0.5 * (cx*cx + cy*cy));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  cout << "Init: x, y, theta = " << x << ", " << y << ", " << theta << endl;

  num_particles = 100;
  Particle particle;

  for (int i = 0; i < num_particles; i++) {
    particle.id = i + 1;
    particle.x = x + std[0] * distn(gen);
    particle.y = y + std[1] * distn(gen);
    particle.theta = theta + std[2] * distn(gen);
    particle.weight = 1.0;
		cout << "Particle: " << particle.id << " " << particle.x << " " << particle.y << " " << particle.theta << endl;
    weights.push_back(1.0);
    particles.push_back(particle);
  }

  cout << "Number of Particles = " << particles.size() << endl;
  cout << "Number of Weights = " << weights.size() << endl;

  is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

#ifdef DEBUG
  cout << "Prediction: dt, velocity, yaw_rate = " << dt << ", " << velocity << ", " << yaw_rate << endl;
#endif

  double x_new, y_new, theta_new;

  for (int i = 0; i < particles.size(); i++) {
    double theta = particles[i].theta;

    if (fabs(yaw_rate) < 0.001) {
      // CV model
      x_new = particles[i].x + velocity * cos(theta) * dt;
      y_new = particles[i].y + velocity * sin(theta) * dt;
      theta_new = particles[i].theta;
    }
    else {
      double alpha = velocity / yaw_rate;

      // CTRV model
      theta_new = theta + yaw_rate * dt;
      x_new = particles[i].x + alpha * (sin(theta_new) - sin(theta));
      y_new = particles[i].y + alpha * (cos(theta) - cos(theta_new));
    }

    // take sensor noise into account
    particles[i].x = x_new + std_pos[0] * distn(gen);
    particles[i].y = y_new + std_pos[1] * distn(gen);
    particles[i].theta = theta_new + std_pos[2] * distn(gen);

#ifdef DEBUG
		cout << "Prediction Particle: " << particles[i].id << " " << particles[i].x \
         << " " << particles[i].y << " " << particles[i].theta << endl;
#endif
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // observations: actual measurements vector gathered from the LIDAR
  // predicted: prediction between 1 particular particule and all of the map landmarks within sensor range
  //            prediction in the sense it is the subset of all landmarks that are within the particule sensor range

  // Nearest Neighbor data association: associate each observation to a Landmark id
  // O(mn) complexity

  double min_dist;
  double new_dist;

  for (int i = 0; i < observations.size(); i++) {
    // already converted from local to map coordinate system
    double x_obs = observations[i].x;
    double y_obs = observations[i].y;

    observations[i].id = -1; // Id of matching landmark in the map
    min_dist = 1e10;

    for (int j = 0; j < predicted.size(); j++) {
      // in map coordinate system
      new_dist = dist(x_obs, y_obs, predicted[j].x, predicted[j].y);
      if (new_dist < min_dist) {
        min_dist = new_dist;

        //observations[i].id = predicted[j].id; // Id of matching landmark in the map
        // FASTER: for direct access
        observations[i].id = j; // Index of matching landmark in the predicted_map
      }
    }
#ifdef DEBUG
    cout << "Association OBS " << i+1 << " with LM " << observations[i].id+1 << endl;
#endif
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

  double weight_sum = 0.0;

  for (int i = 0; i < particles.size(); i++) {

    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta_part = particles[i].theta;

    //---------------------------------------------------------------------------------------
    // 1) predict measurements to all the map landmarks within sensor range for each particle
    //---------------------------------------------------------------------------------------
    std::vector<LandmarkObs> predicted_map;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double x_lmark = map_landmarks.landmark_list[j].x_f;
      double y_lmark = map_landmarks.landmark_list[j].y_f;
      if (dist(x_part, y_part, x_lmark, y_lmark) <= sensor_range) {
        LandmarkObs lmark_obs;

        lmark_obs.id = map_landmarks.landmark_list[j].id_i;
        lmark_obs.x = x_lmark;
        lmark_obs.y = y_lmark;

        predicted_map.push_back(lmark_obs);
      } // end if
    } // end for lmarks in sensor_range

    //------------------------------------------------------------------------------------
    // 2) Convert all observations from local coord system to global map coord system
    //------------------------------------------------------------------------------------
		std::vector<LandmarkObs> observations_map;

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs observation;
      double xobs_loc = observations[j].x;
      double yobs_loc = observations[j].y;

#ifdef DEBUG
      cout << "xobs(" << j+1 << ") loc: x, y = " << xobs_loc << ", " << yobs_loc << endl;
#endif

      // observations in global map coord system
      observation.x = cos(theta_part) * xobs_loc - sin(theta_part) * yobs_loc + x_part;
      observation.y = sin(theta_part) * xobs_loc + cos(theta_part) * yobs_loc + y_part;
      observation.id = observations[j].id;
      observations_map.push_back(observation);

#ifdef DEBUG
      cout << "xobs(" << j+1 << ") map: x, y = " << observation.x << ", " << observation.y << endl;
#endif
    }

    //--------------------------------------------------------------------------------------
    // 3) dataAssociation
    //--------------------------------------------------------------------------------------
    dataAssociation(predicted_map, observations_map);

    //---------------------------------------------------------------------------------------
    // 4) Compute weight for this particule 
    //---------------------------------------------------------------------------------------
    particles[i].weight = 1.0;

    for (int j = 0; j < observations_map.size(); j++) {
      double x = observations_map[j].x;
      double y = observations_map[j].y;

      double mu_x = predicted_map[ observations_map[j].id ].x;
      double mu_y = predicted_map[ observations_map[j].id ].y;

      double wobs = normpdfxy(x, y, mu_x, mu_y, std_landmark[0], std_landmark[1]);
#ifdef DEBUG
      cout << "wobs: " << wobs << endl;
#endif
      particles[i].weight *= wobs;
    }
    weights[i] = particles[i].weight;
#ifdef DEBUG
    cout << "Weight" << i+1 << " = " << weights[i] << endl;
#endif
    weight_sum += particles[i].weight; // for later on weight normalization

    observations_map.clear(); // clear obs for that particle in global map coord system
    predicted_map.clear();
  } // end for particles

  //---------------------------------------------------------------------------------------
  // 5) Normalize weights
  //---------------------------------------------------------------------------------------

  // useless when using discrete_distribution when resampling
  // handled by discrete_distribution automatically

  //for (int i = 0; i < particles.size(); i++) {
  //  particles[i].weight /= weight_sum;
  //  weights[i] = particles[i].weight;
  //}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles;

	//uniform_real_distribution<double> dist_u(0.0, 1.0);

  //int index = int(dist_u(gen) * (double)num_particles);
  //double beta = 0.0;
  //double mw = *max_element(weights.begin(), weights.end());

  //for (int i = 0; i < num_particles; i++) {
  //  beta += dist_u(gen) * 2.0 * mw;
  //  while (beta > weights[index]) {
  //    beta -= weights[index];
  //    index = (index + 1) % num_particles;
  //  }
  //  new_particles.push_back(particles[index]);
  //}

  discrete_distribution<> distrib(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[ distrib(gen) ]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
