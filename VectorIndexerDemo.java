package com.topsec.ti.patronus;

import java.util.Map;

import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



/**
 * Created by hhy on 2017/09/07.
 */
public class VectorIndexerDemo {
    public static void main(String[] args){
        SparkSession spark= SparkSession.builder().master("local").appName("").getOrCreate();
        Dataset<Row> data = spark.read().format("libsvm").load("data/sample_libsvm_data.txt");

        VectorIndexer indexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexed")
                .setMaxCategories(10);
        VectorIndexerModel indexerModel = indexer.fit(data);
        data.show();

        Map<Integer, Map<Double, Integer>> categoryMaps = indexerModel.javaCategoryMaps();
        System.out.print("Chose " + categoryMaps.size() + " categorical features:");

        for (Integer feature : categoryMaps.keySet()) {
            System.out.print(" " + feature);
        }
        System.out.println();

// Create new column "indexed" with categorical values transformed to indices
        Dataset<Row> indexedData = indexerModel.transform(data);
        indexedData.show();
    }
}
