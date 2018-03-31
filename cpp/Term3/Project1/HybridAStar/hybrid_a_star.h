#ifndef HYBRIDASTAR_HYBRID_A_STAR_H
#define HYBRIDASTAR_HYBRID_A_STAR_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class HAS {
public:

    int NUM_THETA_CELLS = 90;
    double SPEED = 1.45;
    double LENGTH = 0.5;

    struct maze_s {

        double g;	// iteration
        int f;
        double x;
        double y;
        double theta;
    };

    struct maze_path {

        vector< vector< vector<int> > > closed;
        vector< vector< vector<maze_s> > > came_from;
        maze_s final;

    };


    /**
      * Constructor
      */
    HAS();

    /**
     * Destructor
     */
    virtual ~HAS();


    int theta_to_stack_number(double theta);

    int idx(double float_num);

    double euclidean(vector<double> state, vector<double> goal);

    vector<maze_s> expand(maze_s state, vector<double> goal);

    maze_path search(vector< vector<int> > grid, vector<double> start, vector<int> goal);

    vector<maze_s> reconstruct_path(vector< vector< vector<maze_s> > > came_from, vector<double> start, HAS::maze_s final);


};

#endif
