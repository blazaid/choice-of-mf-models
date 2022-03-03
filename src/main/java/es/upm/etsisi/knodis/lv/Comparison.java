package es.upm.etsisi.knodis.lv;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.*;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.*;
import es.upm.etsisi.cf4j.util.plot.ColumnPlot;
import es.upm.etsisi.cf4j.util.plot.LinePlot;
import es.upm.etsisi.cf4j.util.plot.PlotSettings;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Comparison {

    private static final String DATASET = "anime";

    private static final long RANDOM_SEED = 42;

    private static final int[] NUM_RECOMMENDATIONS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    public static void main(String[] args) throws Exception {

        DataModel datamodel = null;

        double[] ratings = null;
        double threshold = 0;

        Map<String, Object> pmfParams = null;
        Map<String, Object> biasedmfParams = null;
        Map<String, Object> bemfParamas = null;
        Map<String, Object> nmfParamas = null;
        Map<String, Object> bnmfParams = null;
        Map<String, Object> urpParams = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();

            ratings = new double[]{1, 2, 3, 4, 5};
            threshold = 4;

            pmfParams = Map.of("numIters", 75, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 4);;
            biasedmfParams = Map.of("numIters", 75, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 4);
            bemfParamas = Map.of("numIters", 50, "seed", RANDOM_SEED, "ratings", ratings, "learningRate", 0.01, "regularization", 1.0, "numFactors", 4);
            nmfParamas = Map.of("seed", RANDOM_SEED, "numIters", 25, "numFactors", 4);
            bnmfParams = Map.of("numIters", 75, "seed", RANDOM_SEED, "beta", 5.0, "alpha", 0.8, "numFactors", 4);
            urpParams = Map.of("numIters", 75, "seed", RANDOM_SEED, "ratings", ratings, "numFactors", 8);

        } else if(DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();

            ratings = new double[]{1, 2, 3, 4, 5};
            threshold = 4;

            pmfParams = Map.of("numIters", 100, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 8);;
            biasedmfParams = Map.of("numIters", 100, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 8);
            bemfParamas = Map.of("numIters", 75, "seed", RANDOM_SEED, "ratings", ratings, "learningRate", 0.01, "regularization", 1.0, "numFactors", 4);
            nmfParamas = Map.of("seed", RANDOM_SEED, "numIters", 100, "numFactors", 4);
            bnmfParams = Map.of("numIters", 75, "seed", RANDOM_SEED, "beta", 5.0, "alpha", 0.6, "numFactors", 8);
            urpParams = Map.of("numIters", 50, "seed", RANDOM_SEED, "ratings", ratings, "numFactors", 8);

        } else if(DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();

            ratings = new double[]{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
            threshold = 3;

            pmfParams = Map.of("numIters", 75, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 4);;
            biasedmfParams = Map.of("numIters", 75, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 4);
            bemfParamas = Map.of("numIters", 75, "seed", RANDOM_SEED, "ratings", ratings, "learningRate", 0.01, "regularization", 1.0, "numFactors", 4);
            nmfParamas = Map.of("seed", RANDOM_SEED, "numIters", 25, "numFactors", 4);
            bnmfParams = Map.of("numIters", 50, "seed", RANDOM_SEED, "beta", 25.0, "alpha", 0.6, "numFactors", 12);
            urpParams = Map.of("numIters", 50, "seed", RANDOM_SEED, "ratings", ratings, "numFactors", 8);

        } else if(DATASET.equals("anime")) {
            datamodel = BenchmarkDataModels.MyAnimeList();

            ratings = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            threshold = 8;

            pmfParams = Map.of("numIters", 100, "lambda", 0.001, "seed", RANDOM_SEED, "gamma", 0.001, "numFactors", 4);;
            biasedmfParams = Map.of("numIters", 75, "lambda", 0.1, "seed", RANDOM_SEED, "gamma", 0.01, "numFactors", 4);
            bemfParamas = Map.of("numIters", 100, "seed", RANDOM_SEED, "ratings", ratings, "learningRate", 0.001, "regularization", 1.0, "numFactors", 4);
            nmfParamas = Map.of("seed", RANDOM_SEED, "numIters", 100, "numFactors", 4);
            bnmfParams = Map.of("numIters", 100, "seed", RANDOM_SEED, "beta", 5.0, "alpha", 0.4, "numFactors", 8);
            urpParams = Map.of("numIters", 25, "seed", RANDOM_SEED, "ratings", ratings, "numFactors", 4);
        }

        List<Recommender> recommenders = new ArrayList<>();

        recommenders.add(new PMF(datamodel, pmfParams));
        recommenders.add(new BiasedMF(datamodel, biasedmfParams));
        recommenders.add(new BeMF(datamodel, bemfParamas));
        recommenders.add(new NMF(datamodel, nmfParamas));
        recommenders.add(new BNMF(datamodel, bnmfParams));
        recommenders.add(new URP(datamodel, urpParams));

        for (Recommender recommender : recommenders) {
            recommender.fit();
        }

        // mae

        ColumnPlot maePlot = new ColumnPlot("Recommender", "MAE");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            double mae = new MAE(recommender).getScore();
            maePlot.addColumn(name, mae);
        }

        maePlot.draw();
        maePlot.exportData("results/" + DATASET + "/mae.csv");

        // discovery

        LinePlot discoveryPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "Discovery");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            discoveryPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double discovery = new Discovery(recommender, N).getScore();
                discoveryPlot.setValue(name, N, discovery);
            }
        }

        discoveryPlot.draw();
        discoveryPlot.exportData("results/" + DATASET + "/discovery.csv");

        // diversity

        LinePlot diversityPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "Diversity");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            diversityPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double diversity = new Diversity(recommender, N).getScore();
                diversityPlot.setValue(name, N, diversity);
            }
        }

        diversityPlot.draw();
        diversityPlot.exportData("results/" + DATASET + "/diversity.csv");

        // F1

        LinePlot f1Plot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "F1");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            f1Plot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double f1 = new F1(recommender, N, threshold).getScore();
                f1Plot.setValue(name, N, f1);
            }
        }

        f1Plot.draw();
        f1Plot.exportData("results/" + DATASET + "/f1.csv");

        // NDCG

        LinePlot ndcgPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "NDCG");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            ndcgPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double ndcg = new NDCG(recommender, N).getScore();
                ndcgPlot.setValue(name, N, ndcg);
            }
        }

        ndcgPlot.draw();
        ndcgPlot.exportData("results/" + DATASET + "/ndcg.csv");

        // Novelty

        LinePlot noveltyPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "Novelty");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            noveltyPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double novelty = new Novelty(recommender, N).getScore();
                noveltyPlot.setValue(name, N, novelty);
            }
        }

        noveltyPlot.draw();
        noveltyPlot.exportData("results/" + DATASET + "/novelty.csv");

        // recall

        LinePlot precisionPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "Precision");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            precisionPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double precision = new Precision(recommender, N, threshold).getScore();
                precisionPlot.setValue(name, N, precision);
            }
        }

        precisionPlot.draw();
        precisionPlot.exportData("results/" + DATASET + "/precision.csv");

        // recall

        LinePlot recallPlot = new LinePlot(NUM_RECOMMENDATIONS, "Number of recommendations", "Recall");

        for (Recommender recommender : recommenders) {
            String name = recommender.getClass().getSimpleName();
            recallPlot.addSeries(name);

            for (int N : NUM_RECOMMENDATIONS) {
                double recall = new Recall(recommender, N, threshold).getScore();
                recallPlot.setValue(name, N, recall);
            }
        }

        recallPlot.draw();
        recallPlot.exportData("results/" + DATASET + "/recall.csv");
    }
}
