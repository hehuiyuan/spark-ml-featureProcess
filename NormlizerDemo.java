package com.topsec.ti.patronus;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.*;
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

/**
 * Created by hhy on 2017/09/07.
 */
public class NormlizerDemo {
    public static void main(String[] args){
        SparkSession spark= SparkSession.builder().master("local").appName("").getOrCreate();

        List<Row> data = Arrays.asList(
                RowFactory.create(0, Vectors.dense(1.0, 0.1, -8.0)),
                RowFactory.create(1, Vectors.dense(2.0, 1.0, -4.0)),
                RowFactory.create(2, Vectors.dense(4.0, 10.0, 8.0))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
        dataFrame.show();

// Normalize each Vector using $L^1$ norm.
        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("normFeatures")
                .setP(1.0);

        Dataset<Row> l1NormData = normalizer.transform(dataFrame);
        l1NormData.show();

// Normalize each Vector using $L^\infty$ norm.
        Dataset<Row> lInfNormData =
                normalizer.transform(dataFrame, normalizer.p().w(Double.POSITIVE_INFINITY));
        lInfNormData.show();


       //标准化数据
        StandardScaler standardScaler=new StandardScaler().setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(false)
                .setWithMean(true);
        StandardScalerModel model = standardScaler.fit(dataFrame);
        model.transform(dataFrame).show();



        //min-max标准化
        System.out.println("min-max标准化");
        MinMaxScaler minmax=new MinMaxScaler().setInputCol("features")
                .setOutputCol("scaledFeatures");
        MinMaxScalerModel scalerModel = minmax.fit(dataFrame);
        scalerModel.transform(dataFrame).show();


        //MaxAbsScaler标准化数据
        MaxAbsScaler scaler = new MaxAbsScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        MaxAbsScalerModel sModel = scaler.fit(dataFrame);
        Dataset<Row> scaledData = sModel.transform(dataFrame);
        scaledData.select("features", "scaledFeatures").show();


        //连续特征的划分处理
        double[] splits = {Double.NEGATIVE_INFINITY, -0.5, 0.0, 0.5, Double.POSITIVE_INFINITY};

        List<Row> data1 = Arrays.asList(
                RowFactory.create(-999.9),
                RowFactory.create(-0.5),
                RowFactory.create(-0.3),
                RowFactory.create(0.0),
                RowFactory.create(0.2),
                RowFactory.create(999.9)
        );
        StructType schema1 = new StructType(new StructField[]{
                new StructField("features", DataTypes.DoubleType, false, Metadata.empty())
        });
        Dataset<Row> dataFrame1 = spark.createDataFrame(data1, schema1);

        Bucketizer bucketizer = new Bucketizer()
                .setInputCol("features")
                .setOutputCol("bucketedFeatures")
                .setSplits(splits);

        Dataset<Row> bucketedData = bucketizer.transform(dataFrame1);

        System.out.println("Bucketizer output with " + (bucketizer.getSplits().length-1) + " buckets");
        bucketedData.show();



        //给向量乘上权重向量
        List<Row> data2 = Arrays.asList(
                RowFactory.create("a", Vectors.dense(1.0, 2.0, 3.0)),
                RowFactory.create("b", Vectors.dense(4.0, 5.0, 6.0))
        );

        List<StructField> fields = new ArrayList<>(2);
        fields.add(DataTypes.createStructField("id", DataTypes.StringType, false));
        fields.add(DataTypes.createStructField("vector", new VectorUDT(), false));

        StructType schema2 = DataTypes.createStructType(fields);

        Dataset<Row> dataFrame2 = spark.createDataFrame(data2, schema2);


        Vector transformingVector = Vectors.dense(0.0, 1.0, 2.0);

        ElementwiseProduct transformer = new ElementwiseProduct()
                .setScalingVec(transformingVector)
                .setInputCol("vector")
                .setOutputCol("transformedVector");

// Batch transform the vectors to create new column:
        transformer.transform(dataFrame2).show();


    }
}
