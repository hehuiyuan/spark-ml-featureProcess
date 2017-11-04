package com.topsec.ti.patronus;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
/**
 * Created by hhy on 2017/09/12.
 */
public class CountVectorizerDemo {
    public static void main(String[] args){
        SparkSession spark=SparkSession.builder().master("local").appName(" ").getOrCreate();
        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList("a", "b", "c","d","d")),
                RowFactory.create(Arrays.asList("a", "b", "b", "c", "a"))
        );
        StructType schema = new StructType(new StructField [] {
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> df = spark.createDataFrame(data, schema);

// fit a CountVectorizerModel from the corpus
        CountVectorizerModel cvModel = new CountVectorizer()
                .setInputCol("text")
                .setOutputCol("feature")
                .setVocabSize(6)
                .setMinDF(1)
                .fit(df);

// alternatively, define CountVectorizerModel with a-priori vocabulary
        CountVectorizerModel cvm = new CountVectorizerModel(new String[]{"a", "b", "c"})
                .setInputCol("text")
                .setOutputCol("feature");

        cvModel.transform(df).show(false);

    }
}
