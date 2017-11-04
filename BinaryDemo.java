package com.topsec.ti.patronus;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.Binarizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Created by hhy on 2017/09/06.
 */
public class BinaryDemo {
    public static void main(String[] args){
        SparkSession spark= SparkSession.builder().master("local").appName("").getOrCreate();
        List<Row> data = Arrays.asList(
                RowFactory.create(0, 0.1),
                RowFactory.create(1, 0.8),
                RowFactory.create(2, 0.2)
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
        });
        Dataset<Row> continuousDataFrame = spark.createDataFrame(data, schema);

        Binarizer binarizer = new Binarizer()
                .setInputCol("feature")
                .setOutputCol("binarized_feature")
                .setThreshold(0.5);

        Dataset<Row> binarizedDataFrame = binarizer.transform(continuousDataFrame);

        System.out.println("Binarizer output with Threshold = " + binarizer.getThreshold());
        binarizedDataFrame.show();
    }
}
