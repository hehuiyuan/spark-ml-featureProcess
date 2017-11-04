package com.topsec.ti.patronus;

import org.apache.spark.ml.feature.Interaction;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static javafx.beans.binding.Bindings.select;

/**
 * Created by hhy on 2017/09/07.
 */
public class InteractionDemo {
    public static void main(String[] args){
        SparkSession spark= SparkSession.builder().master("local").appName("").getOrCreate();
        List<Row> data = Arrays.asList(
                RowFactory.create(1, 1, 2, 3, 8, 4, 5),
                RowFactory.create(2, 4, 3, 8, 7, 9, 8),
                RowFactory.create(3, 6, 1, 9, 2, 3, 6),
                RowFactory.create(4, 10, 8, 6, 9, 4, 5),
                RowFactory.create(5, 9, 2, 7, 10, 7, 3),
                RowFactory.create(6, 1, 1, 4, 2, 8, 4)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id1", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id2", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id3", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id4", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id5", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id6", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("id7", DataTypes.IntegerType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);

        VectorAssembler assembler1 = new VectorAssembler()
                .setInputCols(new String[]{"id2", "id3"})
                .setOutputCol("vec1");

        Dataset<Row> assembled1 = assembler1.transform(df);

        VectorAssembler assembler2 = new VectorAssembler()
                .setInputCols(new String[]{"id4","id5"})
                .setOutputCol("vec2");
        Dataset<Row> assembled2 = assembler2.transform(assembled1).select("id1", "vec1", "vec2");

        VectorAssembler assembler3 = new VectorAssembler()
                .setInputCols(new String[]{"id6", "id7"})
                .setOutputCol("vec3");
        Dataset<Row> assembled3 = assembler3.transform(assembled2).select("id1", "vec1", "vec2","vec3");

        Interaction interaction = new Interaction()
                .setInputCols(new String[]{"id1","vec1","vec2","vec3"})
                .setOutputCol("interactedCol");

        Dataset<Row> interacted = interaction.transform(assembled2);

        interacted.show(false);

    }
}
