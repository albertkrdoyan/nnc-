#pragma once

/*  ### EXPERIMENT NO1:
    NeuralNetwork nn;
    int neuralLayer[] = { 1, 3, 3, 2 };

    nn.Create(neuralLayer, sizeof(neuralLayer) / sizeof(neuralLayer[0]));
    nn.SetActivationFunction(ReLU, SoftMaX);
    nn.SetLossFunction(CrossEntropy);
    nn.SetOptimizer(GradientDescent);

    size_t len = 1000;
    double** inputs = new double* [len];
    double** outputs = new double* [len];

    for (int i = 0; i < len; ++i) {
        double vval = (double)i / len;
        inputs[i] = new double[1] { vval };


        if(vval < 0.31)
            outputs[i] = new double[2] { 1, 0 };
        else
            outputs[i] = new double[2] { 0, 1 };
    }

    nn.Train(inputs, outputs, (int)len, 50, 0.03, 8);

    //nn.PrintWeights();

    len = 100;
    double *errf = new double[len];
    for (int i = 0; i < len; ++i) {
        double val = (double)i / len;
        double inp[] = { val };
        nn.Forward(inp);
        //cout << "Org: " << ((val < 0.3) ? "[1, 0]" : "[0, 1]") << " NN: ";
        double *d = new double[2];
        nn.PrintResult(d, false);
        errf[i] = d[0] * 1;
    }

    plot<double>(errf, len);

    return 0;

*/