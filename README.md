# spark-ml-featureProcess
Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@hehuiyuan 
2
1 1 hehuiyuan/spark-ml-featureProcess
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Security  Insights  Settings
spark-ml-featureProcess
/
spark机器学习库评估指标总结.md
 

1
# spark机器学习库评估指标总结
2
## 1 回归评估指标
3
<table>
4
    <tr>
5
        <td></td>
6
        <td>RMSE（均方根误差）</td>
7
        <td>MSE（均方误差）</td>
8
        <td>R2（拟合优度检验）</td>
9
        <td>MAE(平均绝对误差)</td>
10
    </tr>
11
    <tr>
12
        <td>MLLIB库</td><td>√</td><td>√</td><td>√</td><td>√</td>
13
    </tr>
14
    <tr>
15
        <td>ML库</td><td>√</td><td>√</td><td>√</td><td>√</td>
16
    </tr>
17
</table>
18
​
19
### 1.1 MLLIB库
20
#### 1.1.1  RegressionMetrics介绍
21
这个类位于org.apache.spark.mllib.evaluation包下
22
​
23
        class RegressionMetrics @Since("1.2.0") (
24
    predictionAndObservations: RDD[(Double, Double)]) 
25
​
26
参数说明：
27
参数是一个RDD类型的参数，第一列为预测列 第二列为原始标签列
28
​
29
#### 1.1.2  评估指标获取方法
30
*   def  meanAbsoluteError: Double     获取MAE指标 
31
*   def meanSquaredError: Double       获取MSE
32
*   ddef rootMeanSquaredError: Double  获取RMSE
33
*   def r2: Double                    获取R2
34
### 1.2  ML库
35
#### 1.2.1  RegressionEvaluator介绍
36
这个类位于org.apache.spark.ml.evaluation包下面，通过设置下面的metricName包含以上四种指标的评估：
37
​
38
<pre><code>
39
     val metricName: Param[String] = {
40
       val allowedParams = ParamValidators.inArray(Array("mse", "rmse", "r2", "mae"))
41
       new Param(this, "metricName", "metric name in evaluation (mse|rmse|r2|mae)", allowedParams)
42
   }
43
​
44
</code></pre>
45
​
46
#### 1.2.2  评估指标获取方法
47
<pre><code>
48
def evaluate(dataset: Dataset[_]): Double 
49
</pre></code>
@hehuiyuan
Commit changes
Commit summary 
Update spark机器学习库评估指标总结.md
Optional extended description
Add an optional extended description…
  Commit directly to the master branch.
  Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
