#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <fstream>

using namespace std;

template <class T>
void plot(T* arr, size_t len) {
    ofstream wr;
    wr.open("plot.txt");

    for (size_t i = 0; i < len; ++i)
        wr << arr[i] << '\n';

    wr.close();

    system("plot.py");
    system("pause");
}

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
string GetLossFunctionName(LossFunction lf){
    switch (lf) {
    case CrossEntropy:
        return "Cross Entropy";
    case SquareError:
        return "Square Error";
    default:
        return "";
    }
}

enum Optimizer { GradientDescent, ADAM };
string GetOptimizerName(LossFunction opt) {
    switch (opt) {
    case GradientDescent:
        return "Gradient Descent";
    case ADAM:
        return "ADAM";
    default:
        return "";
    }
}

template <class T>
void SoftMax(T* arr, size_t len) {
    // softmax function by ChatGPT
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
    T GetValue(size_t index) { return layer[index]; }
};

template <class T>
class WeightsFF {
private:
    int width, height;
    T** weights;
    T** gradient;
public:
    void Create(int h, int w);
    void Print();
    void NeuralMultiplication(T* layer1, T* layer2);
    void PrintGradients();
    void ResetGradients();
    T** GetWeightsByRef();
    T** GetGradientsByRef();
};

class NeuralNetwork {
private:
    int neural_len;
    WeightsFF<float>* weis;
    NeuralLines1D<float>* neurons;
    ActivationFunction activ1, activ2;
    LossFunction lf;
    Optimizer opt;
public:
    void Create(int* neurons, int length);
    void Forward(float* input_neuron_layer);
    void PrintNeuralLayers();
    void PrintWeights();
    void SetActivationFunction(ActivationFunction activation1, ActivationFunction activation2);
    void SetLossFunction(LossFunction lf);
    void SetOptimizer(Optimizer opt) {
        this->opt = opt;
    }
    float GetLoss(float* org_output) {
        return this->neurons[neural_len - 1].GetError(lf, org_output);
    }
    void BackPropagation(float* org_output);
    void PrintGradients() {
        for (size_t i = 0; i < (size_t)neural_len - 1; ++i)
        {
            cout << "Gradient[" << i << "]:\n";
            weis[i].PrintGradients();
        }
    }
    void Train(float** inputs, float** outputs, int io_len, int levels, float speed, int batches);
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
    size_t lli = (size_t) this->neural_len - 1; // last layer index
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


    if (activ2 == Sigmoid) {
        for (i = 0; i < llnc; ++i)
            lln[i] = llncopy[i] * lln[i] * (1 - lln[i]);
    }
    else if (activ2 == ReLU) {
        for (i = 0; i < llnc; ++i) {
            if (lln[i] > 0)
                lln[i] = llncopy[i];
            else
                lln[i] = 0;
        }
    }
    else  if (activ2 == Linear) {
        for (i = 0; i < llnc; ++i) {
            lln[i] = llncopy[i];
        }
    }

    for (lli; lli > 0; --lli) {
        --lli;

        size_t lenofwei = neurons[lli].GetLength();
        rsize_t lenofhei = neurons[lli + 1].GetLength();
        float** gradCopy = weis[lli].GetGradientsByRef();
        float* llnccc = this->neurons[lli + 1].GetLayerByRef();
        //bias

        for (i = 0; i < llnc; ++i)
            gradCopy[i][lenofwei] += llnccc[i];

        //weis
        for (i = 0; i < llnc; ++i) {
            for (size_t j = 0; j < lenofwei; ++j)
                gradCopy[i][j] += llnccc[i] * neurons[lli].GetValue(j);
        }

        //neurons 
        if (lli == 0)
            break;
        float** weiCopy = weis[lli].GetWeightsByRef();
        float* currNeuronCopy = neurons[lli].GetLayerByRef();
        float* deriv = new float[lenofwei];

        for (rsize_t j = 0; j < lenofwei; ++j) {
            deriv[j] = 0;

            for (i = 0; i < lenofhei; ++i) {
                deriv[j] += llnccc[i] * weiCopy[i][j];
            }

            if (activ1 == Sigmoid) {
                currNeuronCopy[j] = deriv[j] * currNeuronCopy[j] * (1 - currNeuronCopy[j]);
            }
            else if (activ1 == ReLU) {
                if (currNeuronCopy[j] > 0)
                    currNeuronCopy[j] = deriv[j];
                else
                    currNeuronCopy[j] = 0;
            }
            else if (activ1 == Linear)
                currNeuronCopy[j] = deriv[j];
        }

        llnc = lenofwei;
        ++lli;
    }
}

template<class T>
void randomSwapAllElements(T **arr1, T **arr2, int size) {
    // Initialize the random number generator with the current time by ChatGPT
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = size - 1; i > 0; i--) {
        // Generate a random index between 0 and i (inclusive)
        int j = rand() % (i + 1);
        swap(arr1[i], arr1[j]);
        swap(arr2[i], arr2[j]);
    }
}

void NeuralNetwork::Train(float** inputs, float** outputs, int io_len, int levels, float speed, int batches)
{
    int batch_count = (int)(io_len / batches) + ((io_len % batches == 0) ? 0 : 1);
    float *loss = new float[batch_count * levels];

    for (size_t lvl = 0, limit = 0, current_batch_index = 0; lvl < (size_t)levels; ++lvl) {
        for (size_t batch_counter = 0; batch_counter < (size_t)batch_count; ++batch_counter) {
            limit = ((batch_counter + 1) * batches < io_len) ? (batch_counter + 1) * batches : io_len;
            current_batch_index = batch_count * lvl + batch_counter;
            loss[current_batch_index] = 0;

            for (size_t i = batch_counter * batches; i < limit; ++i) {
                Forward(inputs[i]);
                loss[current_batch_index] += GetLoss(outputs[i]);
                BackPropagation(outputs[i]);


                // gradients changing weigths part
            }
        }

        if (lvl != (size_t)(levels - 1))
            randomSwapAllElements<float>(inputs, outputs, io_len);    
    }

    // print loss function graphics
    plot<float>(loss, batch_count * levels);
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
    cout << "[ ";
    for (size_t j = 0; j < (size_t) len; ++j) {
        cout << setprecision(10) << this->layer[j];
        if (j != (size_t) len - 1)
            cout << ", ";
    }
    cout << " ]\n\n";
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
    gradient = new T * [height];
    for (int i = 0; i < height; ++i) {
        weights[i] = new T[width];
        gradient[i] = new T[width];
        for (int j = 0; j < width; ++j) {
            weights[i][j] = (T)rand() / RAND_MAX;
            gradient[i][j] = 0;
        }
    }
}

template <class T>
void WeightsFF<T>::Print() {
    cout << '[';
    for (int i = 0; i < height; ++i) {
        if (i != 0)
            cout << ' ';
        cout << "[ ";
        for (int j = 0; j < width; ++j) {
            cout << setprecision(10) << weights[i][j];
            if (j != width - 1)
                cout << ", ";
        }
        cout << " ]";
        if (i != height - 1)
            cout << endl;
    }
    cout << ']';
    cout << "\n\n";
}

template <class T>
void WeightsFF<T>::PrintGradients() {
    cout << '[';
    for (int i = 0; i < height; ++i) {
        if (i != 0)
            cout << ' ';
        cout << "[ ";
        for (int j = 0; j < width; ++j) {
            cout << setprecision(10) << gradient[i][j];
            if (j != width - 1)
                cout << ", ";
        }
        cout << " ]";
        if (i != height - 1)
            cout << endl;
    }
    cout << ']';
    cout << "\n\n";
}

template <class T>
void WeightsFF<T>::ResetGradients() {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j)
            weights[i][j] = 0;
    }
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

template <class T>
T** WeightsFF<T>::GetWeightsByRef() {
    return this->weights;
}

template <class T>
T** WeightsFF<T>::GetGradientsByRef() {
    return this->gradient;
}

int main()
{
    NeuralNetwork nn;
    int neuralLayer[] = { 2, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(Sigmoid, SoftMaX);
    nn.SetLossFunction(CrossEntropy);

    size_t len = 1000;
    float** inputs = new float* [len];
    float** outputs = new float* [len];
    for (size_t i = 0; i < len; ++i) {
        inputs[i] = new float[2] {(float)(1 + (float)(i / 100)), 1.0f};
        outputs[i] = new float[2] {log(float(i + 1)), 1.0f};
    }

    nn.Train(inputs, outputs, (int)len, 5, 1, 32);


    /*
    for (size_t i = 0; i < len; ++i) {
        cout << i <<"\t: LOG(" << inputs[i][0] << ") = " << outputs[i][0] << ".\n";
    }
    */

    system("pause");
    return 0;
}
