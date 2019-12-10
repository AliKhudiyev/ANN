#include <iostream>

#include "ann.hpp"
#include "argparser.hpp"

using namespace std;

int main(int argc, char* argv[]){

    DataSet heart_disease;
    heart_disease.load("/home/ali/Desktop/Projects/ANN/examples/heart_disease/heart_disease.csv", 0, NAUTOREAD);
    heart_disease.normalize();
    heart_disease.shuffle();
    // heart_disease.print(1);

    vector<DataSet> dataSets=heart_disease.split({60,40});
    cout<<"Train data shape: "<<dataSets[0].shape()<<'\n';
    cout<<"Test data shape: "<<dataSets[1].shape()<<'\n';

    NNet nnet({13, 5, 2});
    nnet.set_labels(InputLayer, 
        {
            "age", "sex", "chest pain type", "resting blood pressure",
            "cholesterol", "fasting blood sugar", "rest ecg",
            "max heart rate", "exercise induced angina", "st depression", 
            "st slope", "# major vessels", "thalassemia"
        });
    nnet.set_labels(OutputLayer, {"Positive", "Negative"});

    // nnet.set_dataSet(dataSets[0]);
    // nnet.initialize(-0.01, 0.01);

    // uint n_epoch;
    // cout<<"# epochs: ";
    // cin>>n_epoch;
    // nnet.train(n_epoch, 1);
    
    // cout<<"Accuracy: "<<nnet.accuracy(dataSets[1])<<'\n';
    // cout<<"Do you want to save the result? [y/n] ";
    // char ans;
    // cin>>ans;
    // if(ans=='y' || ans=='Y'){
    //     string filename;
    //     cout<<"Enter a file name: ";
    //     cin>>filename;
    //     nnet.save(filename);
    // }

    nnet.load("../examples/heart_disease/60l3weights83.csv");
    cout<<"Accuracy [83%]: "<<nnet.accuracy(dataSets[1])<<'\n';

    // Argument arg("sm:", argc, argv);
    // cout<<"[m] value: "<<arg.get_option_value('m')<<endl;

    return 0;
}