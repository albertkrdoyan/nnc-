#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <limits>

using namespace std;

enum ActivationFunction { Linear, ReLU, Sigmoid, SoftMaX };
string GetActivationFunctionName(ActivationFunction a) {
    switch (a) {
    case Linear:
        return "Linear";
    case ReLU:
        return "ReLU";
    case Sigmoid:
        return "Sigmoid";
    case SoftMaX:
        return "SoftMax";
    default:
        return "";
    }
}

enum LossFunction { CrossEntropy, SquareError };
string GetLossFunctionName(LossFunction lf) {
    switch (lf) {
    case CrossEntropy:
        return "Cross Entropy";
    case SquareError:
        return "Square Error";
    default:
        return "";
    }
}

template <class T>
void SoftMax(T* arr, size_t len) {
    size_t i;
    T m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < len; ++i) {
        if (m < arr[i]) {
            m = arr[i];
        }
    }

    sum = 0.0f;
    for (i = 0; i < len; ++i) {
        sum += exp(arr[i] - m);
    }

    constant = m + log(sum);
    for (i = 0; i < len; ++i) {
        arr[i] = exp(arr[i] - constant);
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
    size_t GetLength() {
        return (size_t)this->len;
    }
    T GetError(LossFunction lf, T* org_output) {
        float err = 0;
        if (lf == SquareError) {
            for (int i = 0; i < this->len; ++i)
                err += powf(org_output[i] - this->layer[i], 2);
            err /= this->len;
        }
        else if (lf == CrossEntropy) {
            for (int i = 0; i < this->len; ++i)
                err += org_output[i] * log(this->layer[i]);
            err = 0 - err;
        }
        return err;
    }
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
    ActivationFunction activ1, activ2;
    LossFunction lf;
public:
    void Create(int* neurons, int length);
    void Forward(float* input_neuron_layer);
    void PrintNeuralLayers();
    void PrintWeights();
    void SetActivationFunction(ActivationFunction activation1, ActivationFunction activation2);
    void SetLossFunction(LossFunction lf);
    float GetLoss(float* org_output) {
        return this->neurons[neural_len - 1].GetError(lf, org_output);
    }
    void BackPropagation(float* org_output);
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
    int i;
    for (i = 0; i < this->neural_len - 2; ++i) {
        this->weis[i].NeuralMultiplication(this->neurons[i].GetLayerByRef(), this->neurons[i + 1].GetLayerByRef());
        this->neurons[i + 1].Activation(this->activ1);
    }
    this->weis[i].NeuralMultiplication(this->neurons[i].GetLayerByRef(), this->neurons[i + 1].GetLayerByRef());
    ++i;
    if (this->activ2 == SoftMaX)
        SoftMax<float>(this->neurons[i].GetLayerByRef(), (size_t)this->neurons[i].GetLength());
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

void NeuralNetwork::SetActivationFunction(ActivationFunction activation1, ActivationFunction activation2) {
    this->activ1 = activation1;
    this->activ2 = activation2;
}

void NeuralNetwork::SetLossFunction(LossFunction lf) {
    this->lf = lf;
}

void NeuralNetwork::BackPropagation(float* org_output)
{
    size_t lli = this->neural_len - 1; // last layer index
    size_t llnc = this->neurons[lli].GetLength(); // last later neurons count
    size_t i;

    float* lln = this->neurons[lli].GetLayerByRef();
    float* llncopy = new float[llnc];
    this->neurons[lli].GetLayerByValue(llncopy);

    if (lf == CrossEntropy) {
        if (activ2 == SoftMaX) {
            for (i = 0; i < llnc; ++i)
                lln[i] = lln[i] - org_output[i];
        }
        else {
            for (i = 0; i < llnc; ++i)
                llncopy[i] = -(org_output[i] / lln[i]);
        }
    }
    else if (lf == SquareError) {
        for (i = 0; i < llnc; ++i)
            llncopy[i] = 2 * (lln[i] - org_output[i]);
    }

    cout << "\n\n";
    for (i = 0; i < llnc; ++i)
        cout << llncopy[i] << ' ';
    cout << "\n\n";

    if (activ2 == Sigmoid) {
        for (i = 0; i < llnc; ++i)
            lln[i] = llncopy[i] * lln[i] * (1 - lln[i]);
    }
    else if (activ2 == ReLU) {
        for (i = 0; i < llnc; ++i) {
            if (lln[i] != 0)
                lln[i] = llncopy[i];
        }
    }

    // lln = dL/dz[last layer]
    
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
    case ActivationFunction::Linear:
        break;
    case SoftMaX:
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
    NeuralNetwork nn;
    int neuralLayer[] = { 2, 3, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ReLU, SoftMaX);
    nn.SetLossFunction(CrossEntropy);

    float inputs[] = { 1.5f, 0.9f };
    float output[] = { 1, 0 };
    nn.Forward(inputs);
    cout << "Loss: " << nn.GetLoss(output) << endl;
    nn.PrintNeuralLayers();
    nn.BackPropagation(output);
    nn.PrintNeuralLayers();

//    nn.PrintNeuralLayers();
//    nn.PrintWeights();

//    system("pause");
    return 0;
}
