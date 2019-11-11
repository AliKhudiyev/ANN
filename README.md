# Artificial Neural Network

This library is to create an artificial neural network in C++ easily by structuring it. There are various functionalities in order to play with the general structure of the neural network as well as working with the datasets.

This is an example to show how it works with the iris flower dataset.

_Note: There are lots of functions that do not work properly if the structure of neural network gets complicated. So, for now this is just for simple stuff. I'm working on it currently to make it better._

```
#include <iostream>

#include "ann.hpp"

using namespace std;

int main(){

    DataSet iris("IRIS.csv");
    iris.shuffle();
    vector<DataSet> dataSets=iris.split({70, 30});

    NNet nnet(4,3);
    nnet.set_labels(InputLayer, {"SLength", "SWidth", "PLength", "PWidth"});
    nnet.set_labels(OutputLayer, {"Setosa", "Versicolor", "Virginica"});
    nnet.set_dataSet(dataSets[0]);

    nnet.initialize(-1, 1);
    nnet.train(250);
    cout<<"Accuracy: "<<nnet.accuracy(dataSets[1])<<'\n';

    return 0;
}
```

To run the program on terminal: `g++ --std=c++17 main.cpp matrix.cpp dataset.cpp ann.cpp layer.cpp perceptron.cpp -o main && ./main`