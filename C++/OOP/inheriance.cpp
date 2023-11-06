#include <iostream>

using std::string;

class Car
{
public:
    string model;
    int top_speed;
    int price;

    Car() : model("default"), top_speed(0), price(0)
    {
        std::cout << "Car created" << std::endl;
    }

    // You might want to create a constructor that initializes all values.
    Car(string m, int t, int p) : model(m), top_speed(t), price(p)
    {
        std::cout << "Car created with parameters" << std::endl;
    }
};

class ElectricCar : public Car
{
public:
    int range;
    int battery_capacity;

    ElectricCar() : Car(), range(0), battery_capacity(0)
    { // Call the default constructor of Car
        std::cout << "Electric Car created" << std::endl;
    }

    // If you want to directly set values for an ElectricCar
    ElectricCar(string m, int t, int p, int r, int b) : Car(m, t, p), range(r), battery_capacity(b)
    {
        std::cout << "Electric Car created with parameters" << std::endl;
    }
};

int main()
{
    Car car1("Tesla", 200, 100000);
    ElectricCar car2("Tesla Model X", 250, 120000, 300, 100);

    return 0;
}
