package es.upm.etsisi.knodis.lv;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.*;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.cf4j.util.optimization.RandomSearchCV;

public class RandomSearch {

    private static final String DATASET = "anime";

    private static final long RANDOM_SEED = 42;

    private static final int CV = 4;

    private static final double COVERAGE = 0.5;

    public static void main(String[] args) throws Exception {

        DataModel datamodel = null;

        double[] ratings = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
            ratings = new double[]{1, 2, 3, 4, 5};
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            ratings = new double[]{1, 2, 3, 4, 5};
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
            ratings = new double[]{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        } else if (DATASET.equals("anime")) {
            datamodel = BenchmarkDataModels.MyAnimeList();
            ratings = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        }


        // PMF

        ParamsGrid paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("lambda", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("gamma", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", RANDOM_SEED);

        RandomSearchCV randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, PMF.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/pmf.csv");


        // BiasedMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("lambda", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("gamma", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", RANDOM_SEED);

        randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, BiasedMF.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/biasedmf.csv");


        // NMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", RANDOM_SEED);

        randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, NMF.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/nmf.csv");


        // BeMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("learningRate", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("regularization", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("ratings", ratings);
        paramsGrid.addFixedParam("seed", RANDOM_SEED);

        randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, BeMF.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/bemf.csv");


        // BNMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("alpha", new double[]{0.2, 0.4, 0.6, 0.8});
        paramsGrid.addParam("beta", new double[]{5, 15, 25});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", RANDOM_SEED);

        randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, BNMF.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/bnmf.csv");


        // URP

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", RANDOM_SEED);
        paramsGrid.addFixedParam("ratings", ratings);

        randomSearchCV = new RandomSearchCV(datamodel, paramsGrid, URP.class, MAE.class, CV, COVERAGE, RANDOM_SEED);
        randomSearchCV.fit();
        randomSearchCV.exportResults("results/" + DATASET + "/urp.csv");
    }
}
