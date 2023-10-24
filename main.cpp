#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace std;

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
    int len = 3;
    WeightsFF<float>* weis = new WeightsFF<float>[len];
    NeuralLines1D<float>* neurons = new NeuralLines1D<float>[len + 1];

    float inputs[] = { 1.5f, 0.9f };

    srand((unsigned)time(NULL));
    weis[0].Create(2, 3);
    weis[1].Create(3, 5);
    weis[2].Create(5, 2);
    weis[0].Print();
    weis[1].Print();
    weis[2].Print();

    neurons[0].Create(2);
    neurons[1].Create(3);
    neurons[2].Create(5);
    neurons[3].Create(2);
    neurons[0].Print();
    neurons[1].Print();
    neurons[2].Print();
    neurons[3].Print();

    neurons[0].SetLayerByValue(inputs);
    for(int i = 0; i < len; ++i){        
        weis[i].NeuralMultiplication(neurons[i].GetLayerByRef(), neurons[i + 1].GetLayerByRef());
    }

    neurons[0].Print();
    neurons[1].Print();
    neurons[2].Print();
    neurons[3].Print();

    system("pause");

    return 0;
}
