package es.upm.etsisi.knodis.lv;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.*;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.cf4j.util.optimization.RandomSearchCV;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.stream.IntStream;

public class RandomSearch {
    private static final Logger logger = Logger.getLogger(RandomSearchCV.class.getName());
    private static final String DATASET = "bgg";
    private static final long RANDOM_SEED = 42;
    private static final int CV = 4;
    private static final double COVERAGE = 0.5;
    private static final String EXPORT_PATH = "results/%s/%s.csv";
    private static final boolean OVERWRITE_EXPORT_FILE = false;
    private static final Class<? extends QualityMeasure> METRIC = MAE.class;
    public static void main(String[] args) throws Exception {
        final DataModel datamodel;
        final double[] ratings;

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
        } else if (DATASET.equals("bgg")) {
            datamodel = BenchmarkDataModels.BoardGameGeek();
            ratings = IntStream.range(10, 101).mapToDouble(i -> i / 10.0).toArray();
        } else {
            throw new Exception(DATASET + " not known");
        }

        // PMF
        ParamsGrid paramsGrid = new ParamsGrid();
        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("lambda", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("gamma", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        randomSearch(datamodel, paramsGrid, PMF.class);

        // BiasedMF
        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("lambda", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("gamma", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        randomSearch(datamodel, paramsGrid, BiasedMF.class);

        // NMF
        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        randomSearch(datamodel, paramsGrid, NMF.class);

        // BeMF
        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("learningRate", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("regularization", new double[]{0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("ratings", ratings);

        randomSearch(datamodel, paramsGrid, BeMF.class);

        // BNMF
        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("alpha", new double[]{0.2, 0.4, 0.6, 0.8});
        paramsGrid.addParam("beta", new double[]{5, 15, 25});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        randomSearch(datamodel, paramsGrid, BNMF.class);

        // URP
        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{4, 8, 12});
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("ratings", ratings);

        randomSearch(datamodel, paramsGrid, URP.class);
    }

    private static void randomSearch(
            DataModel datamodel,
            ParamsGrid params,
            Class<? extends Recommender> recommender
    ) {
        final String exportPath = String.format(EXPORT_PATH, DATASET, recommender.getSimpleName().toLowerCase());

        boolean pathExists = new File(exportPath).exists();
        if (OVERWRITE_EXPORT_FILE || !pathExists) {
            params.addFixedParam("seed", RANDOM_SEED);

            RandomSearchCV rs = new RandomSearchCV(datamodel, params, recommender, METRIC, CV, COVERAGE, RANDOM_SEED);
            rs.fit();
            try {
                rs.exportResults(exportPath);
            } catch (IOException e) {
                logger.severe("Could not export results to: " + exportPath);
            }
        } else {
            logger.info("Skipping " + recommender.getName());
        }
    }
}
