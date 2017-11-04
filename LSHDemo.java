package com.topsec.ti.patronus;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH;
import org.apache.spark.ml.feature.BucketedRandomProjectionLSHModel;
import org.apache.spark.ml.feature.MinHashLSH;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.glassfish.hk2.api.Self;
import org.apache.log4j.*;

/**
 * Created by hhy on 2017/09/07.
 */
public class LSHDemo {
    public static void main(String[] args){
        SparkSession spark = SparkSession.builder().master("local").appName("").getOrCreate();
        //spark.sparkContext().setLogLevel("WARN");
        Logger.getLogger("org").setLevel(Level.ERROR);
        List<Row> dataA = Arrays.asList(
                RowFactory.create(0, Vectors.dense(1.0, 1.0)),
                RowFactory.create(1, Vectors.dense(1.0, -1.0)),
                RowFactory.create(2, Vectors.dense(-1.0, -1.0)),
                RowFactory.create(3, Vectors.dense(-1.0, 1.0))
        );

        List<Row> dataB = Arrays.asList(
                RowFactory.create(4, Vectors.dense(1.0, 0.0)),
                RowFactory.create(5, Vectors.dense(-1.0, 0.0)),
                RowFactory.create(6, Vectors.dense(0.0, 1.0)),
                RowFactory.create(7, Vectors.dense(0.0, -1.0))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("keys", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> dfA = spark.createDataFrame(dataA, schema);
        Dataset<Row> dfB = spark.createDataFrame(dataB, schema);
        dfA.show();
        dfB.show();

        Vector key = Vectors.dense(1.0, 0.0);

        BucketedRandomProjectionLSH mh = new BucketedRandomProjectionLSH()
                .setBucketLength(2.0)
                .setNumHashTables(3)
                .setInputCol("keys")
                .setOutputCol("values");

        BucketedRandomProjectionLSHModel model = mh.fit(dfA);
        // Feature Transformation
        model.transform(dfA).show();
// Cache the transformed columns
        Dataset<Row> transformedA = model.transform(dfA).cache();
        Dataset<Row> transformedB = model.transform(dfB).cache();

// Approximate similarity join
        System.out.println("Approximate similarity join  开始");
        model.approxSimilarityJoin(dfA, dfB, 5).show();
        model.approxSimilarityJoin(dfA, dfB, 1).show();
        model.approxSimilarityJoin(transformedA, transformedB, 1.2).show();
        System.out.println("Approximate similarity join  结束");

// Self Join
        System.out.println("Self Join");
        model.approxSimilarityJoin(dfA, dfA, 2.5).filter("datasetA.id < datasetB.id").show();

// Approximate nearest neighbor search
        System.out.println("Approximate nearest neighbor search");
        model.approxNearestNeighbors(dfA, key, 3).show();
        model.approxNearestNeighbors(transformedA, key, 2).show();

    }

}
