#include <cstdlib>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <windows.h>

#include "addit.cpp"

template <class T>
void plot(T* arr, size_t len) {
    ofstream wr;
    wr.open("plot.txt");

    for (size_t i = 0; i < len; ++i)
        wr << arr[i] << '\n';

    wr.close();

    system("plot.py");
    //system("pause");
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
        T err = 0;
        if (lf == SquareError) {
            for (int i = 0; i < this->len; ++i)
                err += pow(org_output[i] - this->layer[i], 2);
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
    int GetHeight() { return height; }
    int GetWidth() { return width; }
    T** GetWeightsByRef();
    T** GetGradientsByRef();
    void Save() {
        ofstream wr;
        wr.open("weis.dat", std::ios_base::app);
        for (int i = 0; i < this->height; ++i) {
            for (int j = 0; j < this->width; ++j) {
                wr << std::setprecision(10) << weights[i][j];
                if (j != this->width - 1)
                    wr << ' ';
            }
            wr << '\n';
        }
        wr << '\n';
        wr.close();
    }

    void Printt() {
        cout << height << ' ' << width << '\n';
    }
};

class NeuralNetwork {
private:
    int neural_len;
    WeightsFF<double>* weis;
    NeuralLines1D<double>* neurons;
    ActivationFunction activ1, activ2;
    LossFunction lf;
    Optimizer opt;
public:
    void Create(int* neurons, int length);
    void Forward(double* input_neuron_layer);
    void PrintNeuralLayers();
    void PrintWeights();
    void SetActivationFunction(ActivationFunction activation1, ActivationFunction activation2);
    void SetLossFunction(LossFunction lf);
    void SetOptimizer(Optimizer opt) {
        this->opt = opt;
    }
    double GetLoss(double* org_output) {
        return this->neurons[neural_len - 1].GetError(lf, org_output);
    }
    void BackPropagation(double* org_output);
    void PrintGradients() {
        for (size_t i = 0; i < (size_t)neural_len - 1; ++i)
        {
            cout << "Gradient[" << i << "]:\n";
            weis[i].PrintGradients();
        }
    }
    void Train(double** inputs, double** outputs, unsigned int io_len, unsigned int levels, double speed, unsigned int batches);
    void ChangeWeights(double speed, double batch);
    void GetResultAsArray(double* dat, bool print) { neurons[neural_len - 1].GetLayerByValue(dat);  if(print) neurons[neural_len - 1].Print();}
    void SaveWeights() {
        for (size_t i = 0; i < neural_len - 1; ++i) {
            weis[i].Save();
        }
    }
    void Printt() {
        weis[0].Printt();
        weis[1].Printt();
    }

    void load() {
        ifstream reader;
        reader.open("weis.dat");
        string ww;

        for (size_t i = 0; i < neural_len - 1; ++i) {
            double** w = weis[i].GetWeightsByRef();
            for (size_t j = 0; j < weis[i].GetHeight(); ++j) {
                for (size_t k = 0; k < weis[i].GetWidth(); ++k) {
                    reader >> ww;
                    w[j][k] = stod(ww);
                }
            }
        }
    }
};

void NeuralNetwork::Create(int* neuron_array, int length) {
    this->neural_len = length;
    this->weis = new WeightsFF<double>[this->neural_len - 1];
    this->neurons = new NeuralLines1D<double>[this->neural_len];

    srand((unsigned)time(NULL));

    for (int i = 0; i < this->neural_len - 1; ++i) {
        weis[i].Create(neuron_array[i], neuron_array[i + 1]);
        neurons[i].Create(neuron_array[i]);
    }
    neurons[this->neural_len - 1].Create(neuron_array[this->neural_len - 1]);
}

void NeuralNetwork::Forward(double* input_neuron_layer) {
    this->neurons[0].SetLayerByValue(input_neuron_layer);
    int i;
    for (i = 0; i < this->neural_len - 2; ++i) {
        this->weis[i].NeuralMultiplication(this->neurons[i].GetLayerByRef(), this->neurons[i + 1].GetLayerByRef());
        this->neurons[i + 1].Activation(this->activ1);
    }
    this->weis[i].NeuralMultiplication(this->neurons[i].GetLayerByRef(), this->neurons[i + 1].GetLayerByRef());
    ++i;
    if (this->activ2 == SoftMaX)
        SoftMax<double>(this->neurons[i].GetLayerByRef(), (size_t)this->neurons[i].GetLength());
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

void NeuralNetwork::BackPropagation(double* org_output)
{
    size_t lli = (size_t)this->neural_len - 1; // last layer index
    size_t llnc = this->neurons[lli].GetLength(); // last later neurons count
    size_t i;

    double* lln = this->neurons[lli].GetLayerByRef();
    double* llncopy = new double[llnc];
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

    for (; lli > 0; --lli) {
        --lli;

        size_t lenofwei = neurons[lli].GetLength();
        size_t lenofhei = neurons[lli + 1].GetLength();
        double** gradCopy = weis[lli].GetGradientsByRef();
        double* llnccc = this->neurons[lli + 1].GetLayerByRef();
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
        double** weiCopy = weis[lli].GetWeightsByRef();
        double* currNeuronCopy = neurons[lli].GetLayerByRef();
        double* deriv = new double[lenofwei];

        for (size_t j = 0; j < lenofwei; ++j) {
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
void randomSwapAllElements(T** arr1, T** arr2, int size) {
    // Initialize the random number generator with the current time by ChatGPT
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = size - 1; i > 0; i--) {
        // Generate a random index between 0 and i (inclusive)
        int j = rand() % (i + 1);
        swap(arr1[i], arr1[j]);
        swap(arr2[i], arr2[j]);
    }
}

void NeuralNetwork::Train(double** inputs, double** outputs, unsigned int io_len, unsigned int levels, double speed, unsigned int batches)
{
    unsigned int batch_count = (int)(io_len / batches) + ((io_len % batches == 0) ? 0 : 1);
    double* loss = new double[batch_count * levels];

    for (size_t lvl = 0, limit = 0, current_batch_index = 0; lvl < (size_t)levels; ++lvl) {
        randomSwapAllElements<double>(inputs, outputs, io_len);

        clock_t begin_time2 = clock();
        for (size_t batch_counter = 0; batch_counter < (size_t)batch_count; ++batch_counter) {
            limit = ((batch_counter + 1) * batches < io_len) ? (batch_counter + 1) * batches : io_len;
            current_batch_index = batch_count * lvl + batch_counter;
            loss[current_batch_index] = 0;

            for (size_t i = batch_counter * batches; i < limit; ++i) {
                Forward(inputs[i]);
                loss[current_batch_index] += GetLoss(outputs[i]);
                BackPropagation(outputs[i]);
            }
            ChangeWeights(speed, (double)batches);
        }

        cout << "LVL: " << lvl << "/" << levels << " : ";
        cout << float(clock() - begin_time2) / CLOCKS_PER_SEC << '\n';
    }
    /* Beep(523, 800);
    Beep(523, 800);
    Beep(523, 800);
    Beep(523, 800);*/
    plot<double>(loss, batch_count * levels);
}

void NeuralNetwork::ChangeWeights(double speed, double batch)
{
    if (opt == GradientDescent) {
        for (size_t g_i = 0; g_i < (size_t)(neural_len - 1); ++g_i) {
            double** weiCopi = weis[g_i].GetWeightsByRef();
            double** gradCopi = weis[g_i].GetGradientsByRef();
            unsigned int hei = weis[g_i].GetHeight(), wid = weis[g_i].GetWidth();
            for (size_t h = 0; h < hei; ++h) {
                for (size_t w = 0; w < wid; ++w) {
                    weiCopi[h][w] -= gradCopi[h][w] * speed / batch;
                    gradCopi[h][w] = 0;
                }
            }
        }
    }
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
    for (size_t j = 0; j < (size_t)len; ++j) {
        cout << setprecision(10) << this->layer[j];
        if (j != (size_t)len - 1)
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
            gradient[i][j] = 0;
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

int main11() {
    NeuralNetwork nn{};
    int neuralLayer[] = { 8, 2, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ReLU, SoftMaX);
    nn.SetLossFunction(CrossEntropy);
    nn.SetOptimizer(GradientDescent);

    int len = 256;
    double** data = new double* [len];
    double** outp = new double* [len];

    data[0] = new double[8] {};
    outp[0] = new double[2] { 1, 0 };
    for (int i = 1; i < len; ++i) {
        data[i] = new double[8] {};
        for (int j = 7; j > -1; --j)
            data[i][j] = data[i - 1][j];

        for (int j = 7; j > -1; --j) {
            if (data[i][j] == 0) {
                data[i][j] = 1;
                break;
            }
            else
                data[i][j] = 0;
        }
        if (i % 2 == 0)
            outp[i] = new double[2] { 1, 0 };
        else
            outp[i] = new double[2] { 0, 1 };
    }

    //nn.Train(data, outp, 3*len/4, 50, 0.03, 8);

//    nn.SaveWeights();
    nn.load();

    double count = 0;
    double* d = new double[2];
    for (int i = len * 3 / 4; i < len; ++i) {
        nn.Forward(data[i]);

        nn.GetResultAsArray(d, false);

        if (abs(outp[i][0] - d[0]) + abs(outp[i][1] - d[1]) < 0.499)
            count++;
    }

    cout << 100 * count / (len/4) << " % correct.\n";

    double* inp = new double[8] {1, 0, 0, 0, 0, 1, 1, 0};
    nn.Forward(inp);
    nn.GetResultAsArray(d, false);
    cout << d[0] << ' ' << d[1] << '\n';

    nn.PrintNeuralLayers();

    return 0;
}

int main()
{
    NeuralNetwork nn{};

    const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28;
    int neuralLayer[] = { 3, 5, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ReLU, SoftMaX);
    nn.SetLossFunction(CrossEntropy);
    nn.SetOptimizer(GradientDescent);

    cout << "DOWNLOADING..\n";
    const clock_t begin_time = clock();
    double** tr_img = new double* [tr_len] {};
    double** tr_img_info = new double* [tr_len] {};
    double** tst_img = new double* [tst_len] {};
    double** tst_img_info = new double* [tst_len] {};

    for (int i = 0; i < tr_len; ++i) {
        tr_img[i] = new double[img_len];
        tr_img_info[i] = new double[10] {0};
    }
    for (int i = 0; i < tst_len; ++i) {
        tst_img[i] = new double[img_len];
        tst_img_info[i] = new double[10] {0};
    }

    /*LoadX("C:\\Users\\alber\\Desktop\\GitHub\\NeuralNetworkCPP\\x64\\Debug\\Digits2\\trainX.txt", tr_len, img_len, tr_img);
    LoadY("C:\\Users\\alber\\Desktop\\GitHub\\NeuralNetworkCPP\\x64\\Debug\\Digits2\\trainY.txt", tr_len, 10, tr_img_info);
    LoadX("C:\\Users\\alber\\Desktop\\GitHub\\NeuralNetworkCPP\\x64\\Debug\\Digits2\\testX.txt", tst_len, img_len, tst_img);
    LoadY("C:\\Users\\alber\\Desktop\\GitHub\\NeuralNetworkCPP\\x64\\Debug\\Digits2\\testY.txt", tst_len, 10, tst_img_info);*/

    cout << "End of DOWNLOADING.. Time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << '\n';

    cout << "Start training...\n";
    //nn.Train(tr_img, tr_img_info, tr_len, 5, 0.03, 32); // (28*28)x128x10 - 20 / 0.03 / 32 - 95.04%
    cout << "End of training... Time: \n";

    /*
    double* d = new double[10];
    double count = 0, summ = 0;
    for (int i = 0; i < tst_len; ++i) {
        nn.Forward(tst_img[i]);

        nn.GetResultAsArray(d, false);

        int ind = -1, match = -1;
        double max = -1;

        for (int j = 0; j < 10; ++j) {
            if (d[j] > max) {
                max = d[j];
                ind = j;
            }
            if (tst_img_info[i][j] == 1)
                match = j;
        }

        if (match == ind)
            count++;
    }

    cout << '\n' << "%% " << 100 * count / tst_len << '\n';*/
    double* inp = new double[3] {.5, .8, .4};
    double* out = new double[2] {1, 0};

    nn.Forward(inp);
    delete[] inp, out;

    nn.PrintNeuralLayers();

    nn.BackPropagation(out);
    nn.SaveWeights();
    nn.PrintWeights();
    nn.PrintNeuralLayers();
    nn.PrintGradients();

    /*int num;
    cin >> num;

    cout << "Number is : -- ";
    for (int i = 0; i < 10; ++i) cout << tr_img_info[num][i] << ' ';
    cout << "--.\n";

    for (int i = 1; i < img_len + 1; ++i) {
        if (tr_img[num][i - 1] == 0)
            cout << ' ';
        else
            cout << 'X';
        if (i % 28 == 0)
            cout << '\n';
    }

    cin >> num;

    cout << "Number is : -- ";
    for (int i = 0; i < 10; ++i) cout << tst_img_info[num][i] << ' ';
    cout << "--.\n";

    for (int i = 1; i < img_len + 1; ++i) {
        if (tst_img[num][i - 1] == 0)
            cout << ' ';
        else
            cout << 'X';
        if (i % 28 == 0)
            cout << '\n';
    }*/

    for (int i = 0; i < tr_len; ++i)
        delete[] tr_img[i];
    delete[] tr_img;
    delete[] tr_img_info;

    for (int i = 0; i < tst_len; ++i)
        delete[] tst_img[i];
    delete[] tst_img;
    delete[] tst_img_info;

    return 0;
 /*   double count = 0;
    double* d = new double[2];
    for (int i = len * 3 / 4; i < len; ++i) {
        nn.Forward(inputs[i]);

        nn.GetResultAsArray(d, false);

        if (abs(outputs[i][0] - d[0]) + abs(outputs[i][1] - d[1]) < 0.999)
            count++;
    }

   // cout << 100 * count / (len2) << " % correct.\n";

    

  //  nn.PrintWeights();

    system("pause");

    return 0;*/
}

/*
=======
    NeuralNetwork nn1;
    int neuralLayers[] = { 3,5,3 };
    nn1.Create(neuralLayers, sizeof(neuralLayers) / sizeof(neuralLayers[0]));
    nn1.SetActivationFunction(ReLU, SoftMaX);
    nn1.SetLossFunction(CrossEntropy);
    nn1.SetOptimizer(GradientDescent);

    

    return 0;
>>>>>>> fa56b90fe818cac32a19da8c64a148496441f485
    NeuralNetwork nn;
    int neuralLayer[] = { 2, 10, 10, 3 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ReLU, SoftMaX);
    nn.SetLossFunction(CrossEntropy);
    nn.SetOptimizer(GradientDescent);

    size_t len = 1000;
    double** inputs = new double* [len];
    double** outputs = new double* [len];

    for (int i = 0; i < len; ++i) {
        double vval = (double)i / len;
        inputs[i] = new double[2] { vval, 1 };


        if(vval < 0.3)
            outputs[i] = new double[3] { 1, 0, 0 };
        else if(vval < 0.61)
            outputs[i] = new double[3] { 0, 1, 0 };
        else
            outputs[i] = new double[3] { 0, 0, 1 };
    }

    nn.Train(inputs, outputs, (int)len, 50, 0.03, 8);

    //nn.PrintWeights();

    len = 100;
    double* errf1 = new double[len];
    double* errf2 = new double[len];
    double* errf3 = new double[len];
    for (int i = 0; i < len; ++i) {
        double val = (double)i / len;
        double inp[] = { val, 1 };
        nn.Forward(inp);
        //cout << "Org: " << ((val < 0.3) ? "[1, 0]" : "[0, 1]") << " NN: ";
        double *d = new double[3];
        nn.GetResultAsArray(d, false);
        errf1[i] = d[0];
        errf2[i] = d[1];
        errf3[i] = d[2];
    }

    plot<double>(errf1, len);
    plot<double>(errf2, len);
    plot<double>(errf3, len);

    return 0;
*/