#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include "classifier.h"
#include <algorithm>
#include <random>

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

std::map<string, vector<double>> sMeanAndStdDev;
std::map<string, vector<double>> dMeanAndStdDev;
std::map<string, vector<double>> sDotMeanAndStdDev;
std::map<string, vector<double>> dDotMeanAndStdDev;

std::map <string, double> label_probabilities;

vector<double> GNB::getMeanAndStdDev(vector<double> sample) {
    vector<double> result;

    double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
    double mean = sum / sample.size();

    std::vector<double> diff(sample.size());
    std::transform(sample.begin(), sample.end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / sample.size());

    result.push_back(mean);
    result.push_back(stdev);

    return result;
}

double GNB::getGaussianProb(double x, vector<double> meanAndStdDev) {
    double result = (1.0/sqrt(2.0*M_PI*pow(meanAndStdDev[1], 2.0))) * exp(-(pow((x-meanAndStdDev[0]), 2.0) / (2.0*pow(meanAndStdDev[1], 2.0) )));

    //cout<<x<<' '<<meanAndStdDev[0]<<' '<<meanAndStdDev[1]<<' '<<result<<'\n';

    return result;
}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                  [3.5, 0.1, 5.9, -0.02],
                  [8.0, -0.3, 3.0, 2.2],
                  ...
              ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    std::map<string, vector<double>> s_per_label;
    std::map<string, vector<double>> d_per_label;
    std::map<string, vector<double>> s_dot_per_label;
    std::map<string, vector<double>> d_dot_per_label;
    std::map<string, int> label_counter;
    for(size_t i = 0; i < (data.size()); i++){
        if (s_per_label.find(labels[i]) == s_per_label.end()) {
            s_per_label[labels[i]] = vector<double>();
        } else {
            s_per_label[labels[i]].push_back(data[i][0]);
        }

        if (d_per_label.find(labels[i]) == d_per_label.end()) {
            d_per_label[labels[i]] = vector<double>();
        } else {
            d_per_label[labels[i]].push_back(data[i][1]);
        }

        if (s_dot_per_label.find(labels[i]) == s_dot_per_label.end()) {
            s_dot_per_label[labels[i]] = vector<double>();
        } else {
            s_dot_per_label[labels[i]].push_back(data[i][2]);
        }

        if (d_dot_per_label.find(labels[i]) == d_dot_per_label.end()) {
            d_dot_per_label[labels[i]] = vector<double>();
        } else {
            d_dot_per_label[labels[i]].push_back(data[i][3]);
        }

        if (label_counter.find(labels[i]) == label_counter.end()) {
            label_counter[labels[i]] = 0;
        } else {
            label_counter[labels[i]]++;
        }
    }

    for(size_t i = 0; i < (possible_labels.size()); i++) {
        sMeanAndStdDev[possible_labels[i]] = getMeanAndStdDev(s_per_label[possible_labels[i]]);
        dMeanAndStdDev[possible_labels[i]] = getMeanAndStdDev(d_per_label[possible_labels[i]]);
        sDotMeanAndStdDev[possible_labels[i]] = getMeanAndStdDev(s_dot_per_label[possible_labels[i]]);
        dDotMeanAndStdDev[possible_labels[i]] = getMeanAndStdDev(d_dot_per_label[possible_labels[i]]);
    }


    for(size_t i = 0; i < (possible_labels.size()); i++){
        const long sum = std::accumulate(label_counter.begin(), label_counter.end(), 0.0,
                                         [](const size_t result, const std::pair<std::string,size_t>& p) {
                                             return result + p.second; });
        label_probabilities[possible_labels[i]] = (double)label_counter[possible_labels[i]] / (double)sum;
    }


    // for (const auto& i: sMeanAndStdDev)
    //     std::cout << i.first << ' ' << i.second[0] << ' ' << i.second[1] << '\n';
    // for (const auto& i: dMeanAndStdDev)
    //     std::cout << i.first << ' ' << i.second[0] << ' ' << i.second[1] << '\n';
    // for (const auto& i: sDotMeanAndStdDev)
    //     std::cout << i.first << ' ' << i.second[0] << ' ' << i.second[1] << '\n';
    // for (const auto& i: dDotMeanAndStdDev)
    //     std::cout << i.first << ' ' << i.second[0] << ' ' << i.second[1] << '\n';


    // for (const auto& i: label_probabilities)
    //     std::cout << i.first << ' ' << i.second << '\n';
}


string GNB::predict(vector<double> sample)
{
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */

    std::map<string, double> label_probs;
    for(size_t i = 0; i < (possible_labels.size()); i++){
        label_probs[possible_labels[i]] = \
                getGaussianProb(sample[0], sMeanAndStdDev[possible_labels[i]]) *\
                getGaussianProb(sample[1], dMeanAndStdDev[possible_labels[i]]) *\
                getGaussianProb(sample[2], sDotMeanAndStdDev[possible_labels[i]]) *\
                getGaussianProb(sample[3], dDotMeanAndStdDev[possible_labels[i]]) *\
                label_probabilities[possible_labels[i]];
    }

    // for (const auto& i: label_probs)
    //     std::cout << i.first << ' ' << i.second << '\n';

    auto x = std::max_element(label_probs.begin(), label_probs.end(),
                              [](const pair<string, double>& p1, const pair<string, double>& p2) {
                                  return p1.second < p2.second; });

    //cout<< x->first + '\n';
    return x->first;
}