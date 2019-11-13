#include <iostream>

#include "ann.hpp"

using namespace std;

int main(){
    
    // cout<<"Opening file: "<<"/home/ali/Desktop/AI/ANN/test/heart_kaggle.csv"<<'\n';

    DataSet iris;
    iris.load("/home/ali/Desktop/AI/ANN/test/heart_kaggle.csv", 0, NAUTOREAD);
    iris.shuffle();
    // iris.print(1);

    vector<DataSet> dataSets=iris.split({90,10});
    cout<<"Train data shape: "<<dataSets[0].shape()<<'\n';
    cout<<"Test data shape: "<<dataSets[1].shape()<<'\n';

    NNet net(13, 2);
    net.set_labels(InputLayer, 
        {
            "age", "sex", "chest pain type", "resting blood pressure",
            "cholesterol", "fasting blood sugar", "rest ecg",
            "max heart rate", "exercise induced angina", "st depression", 
            "st slope", "# major vessels", "thalassemia"
        });
    net.set_labels(OutputLayer, {"Positive", "Negative"});

    // net.set_dataSet(dataSets[0]);
    // net.initialize(-0.01, 0.01);

    // uint n_epoch;
    // cout<<"# epochs: ";
    // cin>>n_epoch; // 4400
    // net.train(n_epoch);
    
    // cout<<"Accuracy: "<<net.accuracy(dataSets[1])<<'\n';
    // cout<<"Do you want to save the result? [y/n] ";
    // char ans;
    // cin>>ans;
    // if(ans=='y' || ans=='Y'){
    //     string filename;
    //     cout<<"Enter a file name: ";
    //     cin>>filename;
    //     net.save(filename);
    // }

    net.load("../examples/heart_disease/weights90.csv");
    cout<<"Accuracy [90%]: "<<net.accuracy(dataSets[1])<<'\n';

    return 0;
}