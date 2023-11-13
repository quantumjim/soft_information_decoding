#include <stdio.h>

// Function declaration (also known as a prototype)
double constant();
double add(int a, int b);

int main() {
    int file_size = 100;
    int counter = 1;
    int tmp = 0;
    tmp = counter; 
    counter = file_size;
    file_size = tmp;
    constant();
    printf("File size is %d\n", file_size);
    // add up the counter and the filesize
    int sum = add(counter, file_size);
    sum = add(sum, constant());
    printf("Sum is %d\n", sum);
    return 0;
}

double constant() {
    const double pi = 3.14;
    printf("Pi is %f\n", pi);
    return pi;

}

double add(int a, int b) {
    return a + b;
}
