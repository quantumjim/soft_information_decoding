#include <iostream>

using std::string;


class Employee { // Blueprint for an Employee
public:  // Default access modifier is private:
    string Name;
    string Company;
    int Age;

    void IntroduceYourself() {
        std::cout << "Name - " << Name << std::endl;
        std::cout << "Company - " << Company << std::endl;
        std::cout << "Age - " << Age << std::endl; 
    }
};


class Employeev2 { // Constructor
public:  
    string Name;
    string Company;
    int Age;

    void IntroduceYourself() {
        std::cout << "Name - " << Name << std::endl;
        std::cout << "Company - " << Company << std::endl;
        std::cout << "Age - " << Age << std::endl; 
    }
    
    Employeev2(string name, string company, int age) { // Constructor (1. no void or anything, 2. same name as class, 3. needs to be public)
        Name = name;
        Company = company;
        Age = age;
    }
};

class Employeev3 { // Encapsulation to make private
private: 
    string Name;
    string Company;
    int Age;

public:
    void setName(string name) {
        Name = name;
    }
    string getName() {
        return Name;
    }
    void setCompany(string company) {
        Company = company;
    }
    string getCompany() {
        return Company;
    } 
    void setAge(int age) {
        if (age >= 18) {
            Age = age;
        }
        else {
            std::cout << "Sorry you are too young to work here" << std::endl;
        }
    }
    int getAge() {
        return Age;
    }
    void IntroduceYourself() {
        std::cout << "Name - " << Name << std::endl;
        std::cout << "Company - " << Company << std::endl;
        std::cout << "Age - " << Age << std::endl; 
    }
        
    Employeev3(string name, string company, int age) { // Constructor (1. no void or anything, 2. same name as class, 3. needs to be public)
        Name = name;
        Company = company;
        Age = age;
    }
};


int main() {
    Employeev3 employee3 = Employeev3("Maurice", "IBM", 23); 
    employee3.setAge(15);
} 






int old() {
    Employee employee1; // Create a variable of type Employee like int number;
    employee1.Name = "Saldina";
    employee1.Company = "YT-CodeBeauty";
    employee1.Age = 25;
    employee1.IntroduceYourself();

    Employee employee2;
    employee2.Name = "John";
    employee2.Company = "Amazon";
    employee2.Age = 35;
    employee2.IntroduceYourself();

    Employeev2 employee3 = Employeev2("Maurice", "IBM", 23); // Constructor
    employee3.IntroduceYourself();

    return 0;
}
