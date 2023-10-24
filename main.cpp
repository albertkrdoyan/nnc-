#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>

using namespace std;

enum ActivationFunction { Linear, ReLU, Sigmoid };
string GetActivationFunctionName(ActivationFunction a) {
    switch (a) {
    case ActivationFunction::Linear:
        return "Linear";
        break;
    case ActivationFunction::ReLU:
        return "ReLU";
        break;
    case ActivationFunction::Sigmoid:
        return "Sigmoid";
        break;
    default:
        return "";
    }
}

template <class T>
class NeuralLines1D {
private:
    T* layer;
    int len;
public:
    void Create(int length);
    T* GetLayerByRef();
    void SetLayerByRef(T* source);
    void SetLayerByValue(T* source);
    void GetLayerByValue(T* target);
    void Print();
    void Activation(ActivationFunction a);
};

template <class T>
class WeightsFF {
private:
    int width, height;
    T** weights;
public:
    void Create(int h, int w);
    void Print();
    void NeuralMultiplication(T* layer1, T* layer2);
};

class NeuralNetwork {
private:
    int neural_len;
    WeightsFF<float>* weis;
    NeuralLines1D<float>* neurons;
    ActivationFunction activ;
public:
    void Create(int* neurons, int length);
    void Forward(float* input_neuron_layer);
    void PrintNeuralLayers();
    void PrintWeights();
    void SetActivationFunction(ActivationFunction activation);
};

void NeuralNetwork::Create(int* neuron_array, int length) {
    this->neural_len = length;
    this->weis = new WeightsFF<float>[this->neural_len - 1];
    this->neurons = new NeuralLines1D<float>[this->neural_len];

    srand((unsigned)time(NULL));

    for (int i = 0; i < this->neural_len - 1; ++i) {
        weis[i].Create(neuron_array[i], neuron_array[i + 1]);
        neurons[i].Create(neuron_array[i]);
    }
    neurons[this->neural_len - 1].Create(neuron_array[this->neural_len - 1]);
}

void NeuralNetwork::Forward(float* input_neuron_layer) {
    this->neurons[0].SetLayerByValue(input_neuron_layer);
    for (int i = 0; i < this->neural_len - 1; ++i) {
        this->weis[i].NeuralMultiplication(this->neurons[i].GetLayerByRef(), this->neurons[i + 1].GetLayerByRef());
        if (i != this->neural_len - 2)
            this->neurons[i + 1].Activation(this->activ);
    }
}

void NeuralNetwork::PrintNeuralLayers() {
    for (int i = 0; i < this->neural_len; ++i) {
        cout << "Neural Layer [" << i + 1 << "]:\n";
        this->neurons[i].Print();
    }
}

void NeuralNetwork::PrintWeights() {
    for (int i = 0; i < this->neural_len - 1; ++i) {
        cout << "Weight [" << i + 1 << "]:\n";
        this->weis[i].Print();
    }
}

void NeuralNetwork::SetActivationFunction(ActivationFunction activation) {
    this->activ = activation;
}

template <class T>
void NeuralLines1D<T>::Create(int length) {
    this->len = length;
    this->layer = new T[length];
    for (int i = 0; i < length; ++i)
        this->layer[i] = 0;
}

template <class T>
T* NeuralLines1D<T>::GetLayerByRef() {
    return this->layer;
}

template <class T>
void NeuralLines1D<T>::SetLayerByRef(T* source) {
    this->layer = source;
}

template<class T>
void NeuralLines1D<T>::SetLayerByValue(T* source)
{
    for (int i = 0; i < this->len; ++i)
        this->layer[i] = source[i];
}

template<class T>
void NeuralLines1D<T>::GetLayerByValue(T* target)
{
    for (int i = 0; i < this->len; ++i)
        target[i] = this->layer[i];
}

template<class T>
void NeuralLines1D<T>::Print()
{
    for (int i = 0; i < this->len; ++i)
        cout << this->layer[i] << ", ";
    cout << endl << endl;
}

template<class T>
void NeuralLines1D<T>::Activation(ActivationFunction a) {
    switch (a) {
    case ActivationFunction::ReLU:
        for (int i = 0; i < this->len; ++i) {
            if (this->layer[i] < 0)
                this->layer[i] = 0;
        }
        break;
    case ActivationFunction::Sigmoid:
        for (int i = 0; i < this->len; ++i)
            this->layer[i] = 1 / (1 + exp(-this->layer[i]));
        break;
    }
}

template <class T>
void WeightsFF<T>::Create(int input, int output) {
    this->width = input + 1;
    this->height = output;

    weights = new T * [height];
    for (int i = 0; i < height; ++i) {
        weights[i] = new T[width];
        for (int j = 0; j < width; ++j)
            weights[i][j] = (T)rand() / RAND_MAX;
    }
}

template <class T>
void WeightsFF<T>::Print() {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j)
            cout << weights[i][j] << ' ';
        cout << endl;
    }
    cout << endl;
}

template <class T>
void WeightsFF<T>::NeuralMultiplication(T* layer1, T* layer2) {
    for (int i = 0; i < this->height; ++i) {
        layer2[i] = 0;
        for (int j = 0; j < this->width - 1; ++j) {
            layer2[i] += weights[i][j] * layer1[j];
        }
        layer2[i] += weights[i][this->width - 1];
    }
}

int main()
{
    cout << "TEST\n";
    NeuralNetwork nn;
    int neuralLayer[] = { 2, 5, 3, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ActivationFunction::Sigmoid);

    float inputs[] = { 1.5f, 0.9f };
    nn.Forward(inputs);

    nn.PrintNeuralLayers();
    nn.PrintWeights();

    system("pause");
    return 0;
}