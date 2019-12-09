#include <iostream>

#include "ann.hpp"

using namespace std;

int main(){

    DataSet iris;
    iris.load("/home/ali/Desktop/Projects/ANN/examples/heart_disease/heart_disease.csv", 0, NAUTOREAD);
    iris.normalize();
    iris.shuffle();
    // iris.print(1);

    vector<DataSet> dataSets=iris.split({70,30});
    cout<<"Train data shape: "<<dataSets[0].shape()<<'\n';
    cout<<"Test data shape: "<<dataSets[1].shape()<<'\n';

    NNet net({13, 5, 2});
    net.set_labels(InputLayer, 
        {
            "age", "sex", "chest pain type", "resting blood pressure",
            "cholesterol", "fasting blood sugar", "rest ecg",
            "max heart rate", "exercise induced angina", "st depression", 
            "st slope", "# major vessels", "thalassemia"
        });
    net.set_labels(OutputLayer, {"Positive", "Negative"});

    // net.print_structure();
    net.set_dataSet(dataSets[0]);
    net.initialize(-0.01, 0.01);

    uint n_epoch;
    cout<<"# epochs: ";
    cin>>n_epoch; // 4400
    net.train(n_epoch, 1);
    
    cout<<"Accuracy: "<<net.accuracy(dataSets[1])<<'\n';
    // cout<<"Do you want to save the result? [y/n] ";
    // char ans;
    // cin>>ans;
    // if(ans=='y' || ans=='Y'){
    //     string filename;
    //     cout<<"Enter a file name: ";
    //     cin>>filename;
    //     net.save(filename);
    // }

    // net.load("../examples/heart_disease/nweights83.csv");
    // cout<<"Accuracy [83%]: "<<net.accuracy(dataSets[1])<<'\n';

    return 0;
}