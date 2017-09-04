# spark机器学习库评估指标总结
## 回归评估指标
<table>
    <tr>
        <td></td>
        <td>RMSE（均方根误差）</td>
        <td>MSE（均方误差）</td>
        <td>R2（拟合优度检验）</td>
        <td>MAE(平均绝对误差)</td>
    </tr>
    <tr>
        <td>MLLIB库</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
    <tr>
        <td>ML库</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
</table>

### 1.1	MLLIB库
#### 1.1.1	RegressionMetrics介绍
这个类位于org.apache.spark.mllib.evaluation包下

        class RegressionMetrics @Since("1.2.0") (
    predictionAndObservations: RDD[(Double, Double)]) 

参数说明：
参数是一个RDD类型的参数，第一列为预测列 第二列为原始标签列

#### 1.1.2  评估指标获取方法
*	def  meanAbsoluteError: Double     获取MAE指标 
*	def meanSquaredError: Double       获取MSE
*	ddef rootMeanSquaredError: Double  获取RMSE
*	def r2: Double                    获取R2
### 1.2	 ML库
#### 1.2.1	RegressionEvaluator介绍
这个类位于org.apache.spark.ml.evaluation包下面，通过设置下面的metricName包含以上四种指标的评估：

<pre><code>
     val metricName: Param[String] = {
       val allowedParams = ParamValidators.inArray(Array("mse", "rmse", "r2", "mae"))
       new Param(this, "metricName", "metric name in evaluation (mse|rmse|r2|mae)", allowedParams)
   }

</code></pre>

#### 1.2.2	评估指标获取方法
<pre><code>
def evaluate(dataset: Dataset[_]): Double 
</pre></code>
通过这个方法可以获取评估结果，其中参数是包含两列：一列是预测列，一列是原始标签列。<br>
选择哪个标准可以通过def getMetricName: String函数设置参数的值。

### 1.3	 评估效果图
如下图所示<br>
![](img/lr.png)
#### 2	分类评估指标
### 2.1	 二元分类
 
<table>
    <tr>
        <td></td>
        <td>AreaUnderROC</td>
        <td>ROC</td>
        <td>(precision, recall) curve</td>
        <td>areaUnderPR</td>
        <td>(threshold, recall) curve</td>
        <td>(threshold, F-Measure) curve</td>
    </tr>
    <tr>
        <td>MLLIB库</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
    <tr>
        <td>ML库</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td>
    </tr>
</table>

#### 2.1.1	MLLIB库
##### 2.1.1.1	  BinaryClassificationMetrics介绍
这个类在org.apache.spark.mllib.evaluation包下面：
<pre><code>
class BinaryClassificationMetrics @Since("1.3.0") (
    @Since("1.3.0") val scoreAndLabels: RDD[(Double, Double)],
    @Since("1.3.0") val numBins: Int)
</code></pre>
参数介绍： 
第一个参数类型是RDD第一列为预测列 第二列为原始标签列；

                  第二个参数是大于0，那么在求roc curve  和pr curve时使用降采样方法到这么多个numBins中去。
##### 2.1.1.2	  评估指标获取方法
*	def roc(): RDD[(Double, Double)]   获取ROC曲线的横纵坐标
*	def areaUnderROC(): Double        获取roc曲线面积
*	def pr(): RDD[(Double, Double)]     获取precision-recall 曲线的横纵坐标
*	def areaUnderPR(): Double          获取pr曲线面积
*	def fMeasureByThreshold(beta: Double): RDD[(Double, Double)]   获取(threshold, F-Measure)组成曲线的横纵坐标
*	def precisionByThreshold(): RDD[(Double, Double)]   获取(threshold, precision) curve




#### 2.1.2	 ML库
##### 2.1.2.1  BinaryClassificationEvaluator类介绍
org.apache.spark.ml.evaluation. BinaryClassificationEvaluator类：
<pre><code>
val metricName: Param[String] = {
  val allowedParams = ParamValidators.inArray(Array("areaUnderROC", "areaUnderPR"))
  new Param(
    this, "metricName", "metric name in evaluation (areaUnderROC|areaUnderPR)", allowedParams)
}
</code></pre>
通过下面的方法可以设置评估指标的类型:
<pre><code>
def setMetricName(value: String): this.type 
</code></pre>
##### 2.1.2.2  评估指标获取方法
通过这个方法override def evaluate(dataset: Dataset[_]): Double
参数为两列：  第一列是预测列 第二列是原始标签列










### 2.2	多元分类
<table>
    <tr>
        <td></td>
        <td>weightedFMeasure加权后的F值</td>
        <td>	weightedPrecision加权后的准确率，对于每一个标签的准确率加权</td>
        <td>weightedRecall加权后的召回</td>
        <td>Accuracy正确分类的样本/总样本</td>
    </tr>
    <tr>
        <td>MLLIB库</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
    <tr>
        <td>ML库</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
</table>


下面的几个评估指标主要是针对每一个标签来说的，MLLIB库中含有的：

<table>
    <tr>
        <td></td>
        <td>confusionMatrix返回一个混淆矩阵</td>
        <td>PositiveRate(label: Double)</td>
        <td>precision(label: Double)</td>
        <td>Recall(label: Double)</td>
        <td>fMeasure(label: Double, beta: Double)</td>
        <td>falsePositiveRate(label: Double)</td>
    </tr>
    <tr>
        <td>MLLIB库</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td>
    </tr>
</table>


#### 2.2.1  MLLIB库
##### 2.2.1.1  MulticlassMetrics介绍
这个类位于org.apache.spark.mllib.evaluation包下面
<pre><code>
@Since("1.1.0")
class MulticlassMetrics @Since("1.1.0") (predictionAndLabels: RDD[(Double, Double)]) 
</code></pre>
参数是一个RDD类型，分为两列 第一列是预测列   第二列是原始标签列

##### 2.2.1.2  评估指标获取方法
*	def confusionMatrix: Matrix  返回一个混淆矩阵
*	def truePositiveRate(label: Double): Double  返回所给标签的TPR指标值
*	def falsePositiveRate(label: Double): Double   返回所给标签的FPR指标
*	def precision(label: Double): Double  返回所给标签的准确率
*	def recall(label: Double): Double   返回所给标签的召回率
*	def fMeasure(label: Double, beta: Double): Double  返回所给标签的F-measure
*	lazy val precision: Double   加权后的准确率
*	lazy val weightedTruePositiveRate  加权后的TPR
*	lazy val weightedFalsePositiveRate: Double  加权后的FPR
*	lazy val weightedRecall: Double  加权后的召回率
*	def weightedFMeasure(beta: Double): Double  加权后的F-measure

#### 2.2.2  ML库
##### 2.2.2.1  MulticlassClassificationEvaluator介绍
这个类位于org.apache.spark.ml.evaluation包下面，包含以下四种评估指标:
*	f1
*	weightedPrecision
*	weightedRecall
*	accuracy
<pre><code>
val metricName: Param[String] = {
  val allowedParams = ParamValidators.inArray(Array("f1", "weightedPrecision",
    "weightedRecall", "accuracy"))
  new Param(this, "metricName", "metric name in evaluation " +
    "(f1|weightedPrecision|weightedRecall|accuracy)", allowedParams)
}
</code></pre>
可以通过下面的函数去设置评估方法：
<pre><code>
def setMetricName(value: String): this.type \
</code></pre>


##### 2.2.2.2  评估指标获取方法
<pre><code>
override def evaluate(dataset: Dataset[_]): Double
</code></pre>
通过调用evaluate方法传入dataset 包含两列一列是预测列 一列是原始标签列
函数内会根据metricName去匹配调用相应的方法得到结果
### 2.3  评估效果图
  ![](img/classification.png)
## 3	聚类评估指标
<table>
    <tr>
        <td></td><td>	the sum of squared distances</td>
    </tr>
    <tr>
        <td>MLLIB库</td><td>√</td>
    </tr>
    <tr>
        <td>ML库</td><td>√</td>
    </tr>
</table>


### 3.1  MLIB库
org.apache.spark.ml.clustering包下面：<br>

每个聚类方法的评估指标在该类对应的model里面，例如 Kmeans方法的评估指标是通过其model类KmeansModel调用def computeCost(data: RDD[Vector]): Double这个函数得到。

### 3.2  ML库
针对聚类ML库同MLLIB库，还没有一个聚类的评估类，只能在自己的model类了吗调用相关方法计算。
### 3.3  评估效果图
根据每个点到其最近聚类中心的距离可以作图如下：
横坐标是点到最近聚类中心距离的范围，纵坐标是某个范围内对应的样本个数<br>

 ![](img/clustering.png) 
