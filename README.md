# Text Classification and Sentiment Analysis

## Project Description

This project analyses a dataset of hotel reviews and build a machine learning model to categorise them as positive or negative.

## Experiment Design

### 1. Data cleaning

- Remove non-English reviews.
- Expand contractions (we'll --> we will).
- Remove digits and words containing digits.

### 2. Preprocessing

- Each review receives a rating score between 1 and 5. We group 1 and 2 as negative, and 4 and 5 as positive. Ratings of 3 are left out because they contain mixed reviews which may introduce noise to the data.
- Use Word2Vec to convert raw text to embeddings.
- Lower-casing, top words removal, and lemmatisation.

### 3. Data splitting

- First 10000 samples: train set. Next 10000 samples: development set. The remaining (roughly 5000 samples): test set.
- Perform 10-fold cross validation on train + development sets.
- Build the final model on train + development sets and predict on test set.

### 4. Model choices

- Multinomial Naive Bayes.
- Random forest.
- Logistic regression with l1 penalty.
- Logistic regression with l2 penalty.
- Support vector machine.
- K-nearest neighbours.
- Multilayer perceptron.


### 5. Extra improvement

We combine the best model found with [VADER](https://github.com/cjhutto/vaderSentiment) using a stacking method to improve its performance.

## Results

The best models are summarised in the table below.

<table>
  <tr>
    <th></th>
    <th colspan="4">Validation Set</th>
    <th colspan="4">Test Set</th>
  </tr>
  <tr>
    <th>Models</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>Logistic regression</td>
    <td>0.9519</td>
    <td>0.9519</td>
    <td>0.9519</td>
    <td>0.9510</td>
    <td>0.9537</td>
    <td>0.9526</td>
    <td>0.9537</td>
    <td>0.9528</td>
  </tr>
  <tr>
    <td>Stacking Model</td>
    <td>0.9525</td>
    <td>0.9516</td>
    <td>0.9525</td>
    <td>0.9518</td>
    <td>0.9552</td>
    <td>0.9541</td>
    <td>0.9552</td>
    <td>0.9542</td>
  </tr>
</table>

The best single model is logistic regression using L2 normalisation. The stacking model uses 2 base models and 1 meta model as follows:
- Base model 1: Logistic regression with L2 normalisation.
- Base model 2: VADER.
- Meta model: Random forest.

Even though the stacking method does not produce any noticeable performance, it might give better results by adding more base models to vary our predictions.

## Install dependencies

```
cd anlp-assignment1

conda create --name new_environment_name --file dependencies.txt

conda activate new_environment_name
```