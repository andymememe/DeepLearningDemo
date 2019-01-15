/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.andymememe.deeplearningdemo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Andy Chen
 */
public class Main {
    
    private static final Logger LOG = LoggerFactory.getLogger(Main.class);

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        LOG.info("Get Data");
        INDArray x = Nd4j.create(new double[][] {
            {0, 0, 0}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 1, 0}, {1, 1, 1}
        });
        
        INDArray y = Nd4j.create(new double[][] {
            {0}, {0}, {1},
            {0}, {1}, {1}
        });
        
        INDArray test = Nd4j.create(new double[][] {
            {1, 0, 1}, {0, 0, 1}
        });
        
        LOG.info("Get Model");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed((long) (Math.random() * 1024 * 1024))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.ADAM)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(3)
                        .nOut(256)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(256)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1),
                           new PerformanceListener(1));
        
        LOG.info("Train Model");
        for(int i = 0; i < 100; i++) {
            model.fit(x, y);
        }
        
        INDArray result = model.output(test);
        LOG.info("Answer 1: {}, get {}", 1, Math.round(result.getDouble(0, 0)));
        LOG.info("Answer 2: {}, get {}", 1, Math.round(result.getDouble(1, 0)));
    }
    
}
