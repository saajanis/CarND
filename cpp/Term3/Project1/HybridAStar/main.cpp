#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "hybrid_breadth_first.h"
#include "hybrid_a_star.h"

using namespace std;


int X = 1;
int _ = 0;

double SPEED = 1.45;
double LENGTH = 0.5;

vector< vector<int> > MAZE = {
        {_,X,X,_,_,_,_,_,_,_,X,X,_,_,_,_,},
        {_,X,X,_,_,_,_,_,_,X,X,_,_,_,_,_,},
        {_,X,X,_,_,_,_,_,X,X,_,_,_,_,_,_,},
        {_,X,X,_,_,_,_,X,X,_,_,_,X,X,X,_,},
        {_,X,X,_,_,_,X,X,_,_,_,X,X,X,_,_,},
        {_,X,X,_,_,X,X,_,_,_,X,X,X,_,_,_,},
        {_,X,X,_,X,X,_,_,_,X,X,X,_,_,_,_,},
        {_,X,X,X,X,_,_,_,X,X,X,_,_,_,_,_,},
        {_,X,X,X,_,_,_,X,X,X,_,_,_,_,_,_,},
        {_,X,X,_,_,_,X,X,X,_,_,X,X,X,X,X,},
        {_,X,_,_,_,X,X,X,_,_,X,X,X,X,X,X,},
        {_,_,_,_,X,X,X,_,_,X,X,X,X,X,X,X,},
        {_,_,_,X,X,X,_,_,X,X,X,X,X,X,X,X,},
        {_,_,X,X,X,_,_,X,X,X,X,X,X,X,X,X,},
        {_,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,},
        {X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,},
};


vector< vector<int> > GRID = MAZE;

vector<double> START = {0.0,0.0,0.0};
vector<int> GOAL = {(int)GRID.size()-1, (int)GRID[0].size()-1};

int main() {

    cout << "Finding path through grid:" << endl;

    // TODO:: Create an Empty Maze and try testing the number of expansions with it

    for(int i = 0; i < GRID.size(); i++)
    {
        cout << GRID[i][0];
        for(int j = 1; j < GRID[0].size(); j++)
        {
            cout << "," << GRID[i][j];
        }
        cout << endl;
    }

    HAS has = HAS();

    HAS::maze_path get_path = has.search(GRID,START,GOAL);

    vector<HAS::maze_s> show_path = has.reconstruct_path(get_path.came_from, START, get_path.final);

    cout << "show path from start to finish" << endl;
    for(int i = show_path.size()-1; i >= 0; i--)
    {

        HAS::maze_s step = show_path[i];
        cout << "##### step " << step.g << " #####" << endl;
        cout << "x " << step.x << endl;
        cout << "y " << step.y << endl;
        cout << "theta " << step.theta << endl;

    }

    return 0;
}