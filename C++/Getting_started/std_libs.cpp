#include <iostream>
#include <cmath>

using namespace std;

int main() {
    /* This is a function to calculate the area of a circle
     * ...
     */
    double radius;
    cin >> radius;
    double area = M_PI * pow(radius, 2);
    cout << area << endl;
    return 0;
}