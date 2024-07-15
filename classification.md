```{code-cell} ipython3
:tags: [remove-cell]
from chapter_preamble import *
from IPython.display import HTML
from IPython.display import Image
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```

(classification1)=
# Classification I: training & predicting

## Overview
In previous chapters, we focused solely on descriptive and exploratory
data analysis questions.
This chapter and the next together serve as our first
foray into answering *predictive* questions about data. In particular, we will
focus on *classification*, i.e., using one or more
variables to predict the value of a categorical variable of interest. This chapter
will cover the basics of classification, how to preprocess data to make it
suitable for use in a classifier, and how to use our observed data to make
predictions. The next chapter will focus on how to evaluate how accurate the
predictions from our classifier are, as well as how to improve our classifier
(where possible) to maximize its accuracy.

## Chapter learning objectives

By the end of the chapter, readers will be able to do the following:

- Recognize situations where a classifier would be appropriate for making predictions.
- Describe what a training data set is and how it is used in classification.
- Interpret the output of a classifier.
- Compute, by hand, the straight-line (Euclidean) distance between points on a graph when there are two predictor variables.
- Explain the K-nearest neighbors classification algorithm.
- Perform K-nearest neighbors classification in Python using `scikit-learn`.
- Use methods from `scikit-learn` to center, scale, balance, and impute data as a preprocessing step.
- Combine preprocessing and model training into a `Pipeline` using `make_pipeline`.

+++

## The classification problem

```{index} predictive question, classification, class, categorical variable
```

```{index} see: feature ; predictor
```

In many situations, we want to make predictions based on the current situation
as well as past experiences. For instance, a doctor may want to diagnose a
patient as either diseased or healthy based on their symptoms and the doctor's
past experience with patients; an email provider might want to tag a given
email as "spam" or "not spam" based on the email's text and past email text data;
or a credit card company may want to predict whether a purchase is fraudulent based
on the current purchase item, amount, and location as well as past purchases.
These tasks are all examples of **classification**, i.e., predicting a
categorical class (sometimes called a *label*) for an observation given its
other variables (sometimes called *features*).

```{index} training set
```

Generally, a classifier assigns an observation without a known class (e.g., a new patient)
to a class (e.g., diseased or healthy) on the basis of how similar it is to other observations
for which we do know the class (e.g., previous patients with known diseases and
symptoms). These observations with known classes that we use as a basis for
prediction are called a **training set**; this name comes from the fact that
we use these data to train, or teach, our classifier. Once taught, we can use
the classifier to make predictions on new data for which we do not know the class.

```{index} K-nearest neighbors, classification; binary
```

There are many possible methods that we could use to predict
a categorical class/label for an observation. In this book, we will
focus on the widely used **K-nearest neighbors** algorithm {cite:p}`knnfix,knncover`.
In your future studies, you might encounter decision trees, support vector machines (SVMs),
logistic regression, neural networks, and more; see the additional resources
section at the end of the next chapter for where to begin learning more about
these other methods. It is also worth mentioning that there are many
variations on the basic classification problem. For example,
we focus on the setting of **binary classification** where only two
classes are involved (e.g., a diagnosis of either healthy or diseased), but you may
also run into multiclass classification problems with more than two
categories (e.g., a diagnosis of healthy, bronchitis, pneumonia, or a common cold).

## Exploring a data set

```{index} breast cancer, question; classification
```

In this chapter and the next, we will study a data set of
[digitized breast cancer image features](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29),
created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian {cite:p}`streetbreastcancer`.
Each row in the data set represents an
image of a tumor sample, including the diagnosis (benign or malignant) and
several other measurements (nucleus texture, perimeter, area, and more).
Diagnosis for each image was conducted by physicians.

As with all data analyses, we first need to formulate a precise question that
we want to answer. Here, the question is *predictive*: can
we use the tumor
image measurements available to us to predict whether a future tumor image
(with unknown diagnosis) shows a benign or malignant tumor? Answering this
question is important because traditional, non-data-driven methods for tumor
diagnosis are quite subjective and dependent upon how skilled and experienced
the diagnosing physician is. Furthermore, benign tumors are not normally
dangerous; the cells stay in the same place, and the tumor stops growing before
it gets very large. By contrast, in malignant tumors, the cells invade the
surrounding tissue and spread into nearby organs, where they can cause serious
damage {cite:p}`stanfordhealthcare`.
Thus, it is important to quickly and accurately diagnose the tumor type to
guide patient treatment.

+++

### Loading the cancer data

Our first step is to load, wrangle, and explore the data using visualizations
in order to better understand the data we are working with. We start by
loading the `pandas` and `altair` packages needed for our analysis.

```{code-cell} ipython3
import pandas as pd
import altair as alt
```

In this case, the file containing the breast cancer data set is a `.csv`
file with headers. We'll use the `read_csv` function with no additional
arguments, and then inspect its contents:

```{index} read function; read_csv
```

```{code-cell} ipython3
:tags: ["output_scroll"]
cancer = pd.read_csv("data/wdbc.csv")
cancer
```

### Describing the variables in the cancer data set

Breast tumors can be diagnosed by performing a *biopsy*, a process where
tissue is removed from the body and examined for the presence of disease.
Traditionally these procedures were quite invasive; modern methods such as fine
needle aspiration, used to collect the present data set, extract only a small
amount of tissue and are less invasive. Based on a digital image of each breast
tissue sample collected for this data set, ten different variables were measured
for each cell nucleus in the image (items 3&ndash;12 of the list of variables below), and then the mean
 for each variable across the nuclei was recorded. As part of the
data preparation, these values have been *standardized (centered and scaled)*; we will discuss what this
means and why we do it later in this chapter. Each image additionally was given
a unique ID and a diagnosis by a physician.  Therefore, the
total set of variables per image in this data set is:

1. ID: identification number
2. Class: the diagnosis (M = malignant or B = benign)
3. Radius: the mean of distances from center to points on the perimeter
4. Texture: the standard deviation of gray-scale values
5. Perimeter: the length of the surrounding contour
6. Area: the area inside the contour
7. Smoothness: the local variation in radius lengths
8. Compactness: the ratio of squared perimeter and area
9. Concavity: severity of concave portions of the contour
10. Concave Points: the number of concave portions of the contour
11. Symmetry: how similar the nucleus is when mirrored
12. Fractal Dimension: a measurement of how "rough" the perimeter is

+++

```{index} DataFrame; info
```

Below we use the `info` method to preview the data frame. This method can
make it easier to inspect the data when we have a lot of columns:
it prints only the column names down the page (instead of across),
as well as their data types and the number of non-missing entries.

```{code-cell} ipython3
cancer.info()
```

```{index} Series; unique
```

From the summary of the data above, we can see that `Class` is of type `object`.
We can use the `unique` method on the `Class` column to see all unique values
present in that column. We see that there are two diagnoses:
benign, represented by `"B"`, and malignant, represented by `"M"`.

```{code-cell} ipython3
cancer["Class"].unique()
```

We will improve the readability of our analysis
by renaming `"M"` to `"Malignant"` and `"B"` to `"Benign"` using the `replace`
method. The `replace` method takes one argument: a dictionary that maps
previous values to desired new values.
We will verify the result using the `unique` method.

```{index} Series; replace
```

```{code-cell} ipython3
cancer["Class"] = cancer["Class"].replace({
    "M" : "Malignant",
    "B" : "Benign"
})

cancer["Class"].unique()
```

### Exploring the cancer data

```{index} DataFrame; groupby, Series;size
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("benign_count", "{:0.0f}".format(cancer["Class"].value_counts()["Benign"]))
glue("benign_pct", "{:0.0f}".format(100*cancer["Class"].value_counts(normalize=True)["Benign"]))
glue("malignant_count", "{:0.0f}".format(cancer["Class"].value_counts()["Malignant"]))
glue("malignant_pct", "{:0.0f}".format(100*cancer["Class"].value_counts(normalize=True)["Malignant"]))
```

Before we start doing any modeling, let's explore our data set. Below we use
the `groupby` and `size` methods to find the number and percentage
of benign and malignant tumor observations in our data set. When paired with
`groupby`, `size` counts the number of observations for each value of the `Class`
variable. Then we calculate the percentage in each group by dividing by the total
number of observations and multiplying by 100.
The total number of observations equals the number of rows in the data frame,
which we can access via the `shape` attribute of the data frame
(`shape[0]` is the number of rows and `shape[1]` is the number of columns).
We have
{glue:text}`benign_count` ({glue:text}`benign_pct`\%) benign and
{glue:text}`malignant_count` ({glue:text}`malignant_pct`\%) malignant
tumor observations.

```{code-cell} ipython3
100 * cancer.groupby("Class").size() / cancer.shape[0]
```

```{index} Series; value_counts
```

The `pandas` package also has a more convenient specialized `value_counts` method for
counting the number of occurrences of each value in a column. If we pass no arguments
to the method, it outputs a series containing the number of occurences
of each value. If we instead pass the argument `normalize=True`, it instead prints the fraction
of occurrences of each value.

```{code-cell} ipython3
cancer["Class"].value_counts()
```

```{code-cell} ipython3
cancer["Class"].value_counts(normalize=True)
```

```{index} visualization; scatter
```

Next, let's draw a colored scatter plot to visualize the relationship between the
perimeter and concavity variables. Recall that the default palette in `altair`
is colorblind-friendly, so we can stick with that here.

```{code-cell} ipython3
:tags: ["remove-output"]
perim_concav = alt.Chart(cancer).mark_circle().encode(
    x=alt.X("Perimeter").title("Perimeter (standardized)"),
    y=alt.Y("Concavity").title("Concavity (standardized)"),
    color=alt.Color("Class").title("Diagnosis")
)
perim_concav
```

```{code-cell} ipython3
:tags: ["remove-cell"]
glue("fig:05-scatter", perim_concav)
```

:::{glue:figure} fig:05-scatter
:name: fig:05-scatter

Scatter plot of concavity versus perimeter colored by diagnosis label.
:::

+++

In {numref}`fig:05-scatter`, we can see that malignant observations typically fall in
the upper right-hand corner of the plot area. By contrast, benign
observations typically fall in the lower left-hand corner of the plot. In other words,
benign observations tend to have lower concavity and perimeter values, and malignant
ones tend to have larger values. Suppose we
obtain a new observation not in the current data set that has all the variables
measured *except* the label (i.e., an image without the physician's diagnosis
for the tumor class). We could compute the standardized perimeter and concavity values,
resulting in values of, say, 1 and 1. Could we use this information to classify
that observation as benign or malignant? Based on the scatter plot, how might
you classify that new observation? If the standardized concavity and perimeter
values are 1 and 1 respectively, the point would lie in the middle of the
orange cloud of malignant points and thus we could probably classify it as
malignant. Based on our visualization, it seems like
it may be possible to make accurate predictions of the `Class` variable (i.e., a diagnosis) for
tumor images with unknown diagnoses.

+++

## Classification with K-nearest neighbors

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [2, 4]
glue("new_point_1_0", "{:.1f}".format(new_point[0]))
glue("new_point_1_1", "{:.1f}".format(new_point[1]))
attrs = ["Perimeter", "Concavity"]
points_df = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df = pd.concat((cancer, points_df), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal data set
my_distances = euclidean_distances(perim_concav_with_new_point_df[attrs])[
    len(cancer)
][:-1]
```

```{index} K-nearest neighbors; classification
```

In order to actually make predictions for new observations in practice, we
will need a classification algorithm.
In this book, we will use the K-nearest neighbors classification algorithm.
To predict the label of a new observation (here, classify it as either benign
or malignant), the K-nearest neighbors classifier generally finds the $K$
"nearest" or "most similar" observations in our training set, and then uses
their diagnoses to make a prediction for the new observation's diagnosis. $K$
is a number that we must choose in advance; for now, we will assume that someone has chosen
$K$ for us. We will cover how to choose $K$ ourselves in the next chapter.

To illustrate the concept of K-nearest neighbors classification, we
will walk through an example.  Suppose we have a
new observation, with standardized perimeter
of {glue:text}`new_point_1_0` and standardized concavity
of {glue:text}`new_point_1_1`, whose
diagnosis "Class" is unknown. This new observation is
depicted by the red, diamond point in {numref}`fig:05-knn-2`.

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point = (
    alt.Chart(perim_concav_with_new_point_df)
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter").title("Perimeter (standardized)"),
        y=alt.Y("Concavity").title("Concavity (standardized)"),
        color=alt.Color("Class").title("Diagnosis"),
        shape=alt.Shape("Class").scale(range=["circle", "circle", "diamond"]),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(100), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None)),
    )
)
glue('fig:05-knn-2', perim_concav_with_new_point, display=True)
```

:::{glue:figure} fig:05-knn-2
:name: fig:05-knn-2

Scatter plot of concavity versus perimeter with new observation represented as a red diamond.
:::

```{code-cell} ipython3
:tags: [remove-cell]

near_neighbor_df = pd.concat([
    cancer.loc[[np.argmin(my_distances)], attrs],
    perim_concav_with_new_point_df.loc[[cancer.shape[0]], attrs],
])
glue("1-neighbor_per", "{:.1f}".format(near_neighbor_df.iloc[0, :]["Perimeter"]))
glue("1-neighbor_con", "{:.1f}".format(near_neighbor_df.iloc[0, :]["Concavity"]))
```

{numref}`fig:05-knn-3` shows that the nearest point to this new observation is
**malignant** and located at the coordinates ({glue:text}`1-neighbor_per`,
{glue:text}`1-neighbor_con`). The idea here is that if a point is close to another
in the scatter plot, then the perimeter and concavity values are similar,
and so we may expect that they would have the same diagnosis.

```{code-cell} ipython3
:tags: [remove-cell]

line = (
    alt.Chart(near_neighbor_df)
    .mark_line()
    .encode(x="Perimeter", y="Concavity", color=alt.value("black"))
)

glue('fig:05-knn-3', (perim_concav_with_new_point + line), display=True)
```

:::{glue:figure} fig:05-knn-3
:name: fig:05-knn-3

Scatter plot of concavity versus perimeter. The new observation is represented
as a red diamond with a line to the one nearest neighbor, which has a malignant
label.
:::

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0.2, 3.3]
attrs = ["Perimeter", "Concavity"]
points_df2 = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df2 = pd.concat((cancer, points_df2), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal data set
my_distances2 = euclidean_distances(perim_concav_with_new_point_df2[attrs])[
    len(cancer)
][:-1]
glue("new_point_2_0", "{:.1f}".format(new_point[0]))
glue("new_point_2_1", "{:.1f}".format(new_point[1]))
```

```{code-cell} ipython3
:tags: [remove-cell]

perim_concav_with_new_point2 = (
    alt.Chart(
        perim_concav_with_new_point_df2,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None)),
    )
)

near_neighbor_df2 = pd.concat([
    cancer.loc[[np.argmin(my_distances2)], attrs],
    perim_concav_with_new_point_df2.loc[[cancer.shape[0]], attrs],
])
line2 = alt.Chart(near_neighbor_df2).mark_line().encode(
    x="Perimeter",
    y="Concavity",
    color=alt.value("black")
)

glue("2-neighbor_per", "{:.1f}".format(near_neighbor_df2.iloc[0, :]["Perimeter"]))
glue("2-neighbor_con", "{:.1f}".format(near_neighbor_df2.iloc[0, :]["Concavity"]))
glue('fig:05-knn-4', (perim_concav_with_new_point2 + line2), display=True)
```

Suppose we have another new observation with standardized perimeter
{glue:text}`new_point_2_0` and concavity of {glue:text}`new_point_2_1`. Looking at the
scatter plot in {numref}`fig:05-knn-4`, how would you classify this red,
diamond observation? The nearest neighbor to this new point is a
**benign** observation at ({glue:text}`2-neighbor_per`, {glue:text}`2-neighbor_con`).
Does this seem like the right prediction to make for this observation? Probably
not, if you consider the other nearby points.

+++

:::{glue:figure} fig:05-knn-4
:name: fig:05-knn-4

Scatter plot of concavity versus perimeter. The new observation is represented
as a red diamond with a line to the one nearest neighbor, which has a benign
label.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# The index of 3 rows that has smallest distance to the new point
min_3_idx = np.argpartition(my_distances2, 3)[:3]
near_neighbor_df3 = pd.concat([
    cancer.loc[[min_3_idx[1]], attrs],
    perim_concav_with_new_point_df2.loc[[cancer.shape[0]], attrs],
])
near_neighbor_df4 = pd.concat([
    cancer.loc[[min_3_idx[2]], attrs],
    perim_concav_with_new_point_df2.loc[[cancer.shape[0]], attrs],
])
```

```{code-cell} ipython3
:tags: [remove-cell]

line3 = alt.Chart(near_neighbor_df3).mark_line().encode(
    x="Perimeter",
    y="Concavity",
    color=alt.value("black")
)
line4 = alt.Chart(near_neighbor_df4).mark_line().encode(
    x="Perimeter",
    y="Concavity",
    color=alt.value("black")
)
glue("fig:05-knn-5", (perim_concav_with_new_point2 + line2 + line3 + line4), display=True)
```

To improve the prediction we can consider several
neighboring points, say $K = 3$, that are closest to the new observation
to predict its diagnosis class. Among those 3 closest points, we use the
*majority class* as our prediction for the new observation. As shown in {numref}`fig:05-knn-5`, we
see that the diagnoses of 2 of the 3 nearest neighbors to our new observation
are malignant. Therefore we take majority vote and classify our new red, diamond
observation as malignant.

+++

:::{glue:figure} fig:05-knn-5
:name: fig:05-knn-5

Scatter plot of concavity versus perimeter with three nearest neighbors.
:::

+++

Here we chose the $K=3$ nearest observations, but there is nothing special
about $K=3$. We could have used $K=4, 5$ or more (though we may want to choose
an odd number to avoid ties). We will discuss more about choosing $K$ in the
next chapter.

+++

### Distance between points

```{index} distance; K-nearest neighbors, straight line; distance
```

We decide which points are the $K$ "nearest" to our new observation using the
*straight-line distance* (we will often just refer to this as *distance*).
Suppose we have two observations $a$ and $b$, each having two predictor
variables, $x$ and $y$.  Denote $a_x$ and $a_y$ to be the values of variables
$x$ and $y$ for observation $a$; $b_x$ and $b_y$ have similar definitions for
observation $b$.  Then the straight-line distance between observation $a$ and
$b$ on the x-y plane can be computed using the following formula:

$$\mathrm{Distance} = \sqrt{(a_x -b_x)^2 + (a_y - b_y)^2}$$

+++

To find the $K$ nearest neighbors to our new observation, we compute the distance
from that new observation to each observation in our training data, and select the $K$ observations corresponding to the
$K$ *smallest* distance values. For example, suppose we want to use $K=5$ neighbors to classify a new
observation with perimeter {glue:text}`3-new_point_0` and
concavity {glue:text}`3-new_point_1`, shown as a red diamond in {numref}`fig:05-multiknn-1`. Let's calculate the distances
between our new point and each of the observations in the training set to find
the $K=5$ neighbors that are nearest to our new point.
You will see in the code below, we compute the straight-line
distance using the formula above: we square the differences between the two observations' perimeter
and concavity coordinates, add the squared differences, and then take the square root.
In order to find the $K=5$ nearest neighbors, we will use the `nsmallest` function from `pandas`.

```{index} nsmallest
```

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0, 3.5]
attrs = ["Perimeter", "Concavity"]
points_df3 = pd.DataFrame(
    {"Perimeter": new_point[0], "Concavity": new_point[1], "Class": ["Unknown"]}
)
perim_concav_with_new_point_df3 = pd.concat((cancer, points_df3), ignore_index=True)
perim_concav_with_new_point3 = (
    alt.Chart(
        perim_concav_with_new_point_df3,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None)),
    )
)

glue("3-new_point_0", "{:.1f}".format(new_point[0]))
glue("3-new_point_1", "{:.1f}".format(new_point[1]))
glue("fig:05-multiknn-1", perim_concav_with_new_point3)
```

:::{glue:figure} fig:05-multiknn-1
:name: fig:05-multiknn-1

Scatter plot of concavity versus perimeter with new observation represented as a red diamond.
:::


```{code-cell} ipython3
new_obs_Perimeter = 0
new_obs_Concavity = 3.5
cancer["dist_from_new"] = (
       (cancer["Perimeter"] - new_obs_Perimeter) ** 2
     + (cancer["Concavity"] - new_obs_Concavity) ** 2
)**(1/2)
cancer.nsmallest(5, "dist_from_new")[[
    "Perimeter",
    "Concavity",
    "Class",
    "dist_from_new"
]]
```

```{code-cell} ipython3
:tags: [remove-cell]
# code needed to render the latex table with distance calculations
from IPython.display import Latex
five_neighbors = (
    cancer
   [["Perimeter", "Concavity", "Class"]]
   .assign(dist_from_new = (
       (cancer["Perimeter"] - new_obs_Perimeter) ** 2
     + (cancer["Concavity"] - new_obs_Concavity) ** 2
   )**(1/2))
   .nsmallest(5, "dist_from_new")
).reset_index()

for i in range(5):
    glue(f"gn{i}_perim", "{:0.2f}".format(five_neighbors["Perimeter"][i]))
    glue(f"gn{i}_concav", "{:0.2f}".format(five_neighbors["Concavity"][i]))
    glue(f"gn{i}_class", five_neighbors["Class"][i])

    # typeset perimeter,concavity with parentheses if negative for latex
    nperim = f"{five_neighbors['Perimeter'][i]:.2f}" if five_neighbors['Perimeter'][i] > 0 else f"({five_neighbors['Perimeter'][i]:.2f})"
    nconcav = f"{five_neighbors['Concavity'][i]:.2f}" if five_neighbors['Concavity'][i] > 0 else f"({five_neighbors['Concavity'][i]:.2f})"

    glue(f"gdisteqn{i}", Latex(f"\sqrt{{(0-{nperim})^2+(3.5-{nconcav})^2}}={five_neighbors['dist_from_new'][i]:.2f}"))
```

In {numref}`tab:05-multiknn-mathtable` we show in mathematical detail how
we computed the `dist_from_new` variable (the
distance to the new observation) for each of the 5 nearest neighbors in the
training data.

```{table} Evaluating the distances from the new observation to each of its 5 nearest neighbors
:name: tab:05-multiknn-mathtable
| Perimeter | Concavity | Distance            | Class |
|-----------|-----------|----------------------------------------|-------|
| {glue:text}`gn0_perim`  | {glue:text}`gn0_concav`  | {glue:}`gdisteqn0` | {glue:text}`gn0_class`     |
| {glue:text}`gn1_perim`  | {glue:text}`gn1_concav`  | {glue:}`gdisteqn1` | {glue:text}`gn1_class`     |
| {glue:text}`gn2_perim`  | {glue:text}`gn2_concav`  | {glue:}`gdisteqn2` | {glue:text}`gn2_class`     |
| {glue:text}`gn3_perim`  | {glue:text}`gn3_concav`  | {glue:}`gdisteqn3` | {glue:text}`gn3_class`     |
| {glue:text}`gn4_perim`  | {glue:text}`gn4_concav`  | {glue:}`gdisteqn4` | {glue:text}`gn4_class`     |
```

+++

The result of this computation shows that 3 of the 5 nearest neighbors to our new observation are
malignant; since this is the majority, we classify our new observation as malignant.
These 5 neighbors are circled in {numref}`fig:05-multiknn-3`.

```{code-cell} ipython3
:tags: [remove-cell]

circle_path_df = pd.DataFrame(
    {
        "Perimeter": new_point[0] + 1.4 * np.cos(np.linspace(0, 2 * np.pi, 100)),
        "Concavity": new_point[1] + 1.4 * np.sin(np.linspace(0, 2 * np.pi, 100)),
    }
)
circle = alt.Chart(circle_path_df.reset_index()).mark_line(color="black").encode(
    x="Perimeter",
    y="Concavity",
    order="index"
)

glue("fig:05-multiknn-3", (perim_concav_with_new_point3 + circle))
```

:::{glue:figure} fig:05-multiknn-3
:name: fig:05-multiknn-3

Scatter plot of concavity versus perimeter with 5 nearest neighbors circled.
:::

+++

### More than two explanatory variables

Although the above description is directed toward two predictor variables,
exactly the same K-nearest neighbors algorithm applies when you
have a higher number of predictor variables.  Each predictor variable may give us new
information to help create our classifier.  The only difference is the formula
for the distance between points. Suppose we have $m$ predictor
variables for two observations $a$ and $b$, i.e.,
$a = (a_{1}, a_{2}, \dots, a_{m})$ and
$b = (b_{1}, b_{2}, \dots, b_{m})$.

```{index} distance; more than two variables
```

The distance formula becomes

$$\mathrm{Distance} = \sqrt{(a_{1} -b_{1})^2 + (a_{2} - b_{2})^2 + \dots + (a_{m} - b_{m})^2}.$$

This formula still corresponds to a straight-line distance, just in a space
with more dimensions. Suppose we want to calculate the distance between a new
observation with a perimeter of 0, concavity of 3.5, and symmetry of 1, and
another observation with a perimeter, concavity, and symmetry of 0.417, 2.31, and
0.837 respectively. We have two observations with three predictor variables:
perimeter, concavity, and symmetry. Previously, when we had two variables, we
added up the squared difference between each of our (two) variables, and then
took the square root. Now we will do the same, except for our three variables.
We calculate the distance as follows

$$\mathrm{Distance} =\sqrt{(0 - 0.417)^2 + (3.5 - 2.31)^2 + (1 - 0.837)^2} = 1.27.$$

Let's calculate the distances between our new observation and each of the
observations in the training set to find the $K=5$ neighbors when we have these
three predictors.

```{code-cell} ipython3
new_obs_Perimeter = 0
new_obs_Concavity = 3.5
new_obs_Symmetry = 1
cancer["dist_from_new"] = (
      (cancer["Perimeter"] - new_obs_Perimeter) ** 2
    + (cancer["Concavity"] - new_obs_Concavity) ** 2
    + (cancer["Symmetry"] - new_obs_Symmetry) ** 2
)**(1/2)
cancer.nsmallest(5, "dist_from_new")[[
    "Perimeter",
    "Concavity",
    "Symmetry",
    "Class",
    "dist_from_new"
]]
```

Based on $K=5$ nearest neighbors with these three predictors we would classify
the new observation as malignant since 4 out of 5 of the nearest neighbors are malignant class.
{numref}`fig:05-more` shows what the data look like when we visualize them
as a 3-dimensional scatter with lines from the new observation to its five nearest neighbors.

```{code-cell} ipython3
:tags: [remove-cell]

new_point = [0, 3.5, 1]
attrs = ["Perimeter", "Concavity", "Symmetry"]
points_df4 = pd.DataFrame(
    {
        "Perimeter": new_point[0],
        "Concavity": new_point[1],
        "Symmetry": new_point[2],
        "Class": ["Unknown"],
    }
)
perim_concav_with_new_point_df4 = pd.concat((cancer, points_df4), ignore_index=True)
# Find the euclidean distances from the new point to each of the points
# in the orginal data set
my_distances4 = euclidean_distances(perim_concav_with_new_point_df4[attrs])[
    len(cancer)
][:-1]
```

```{code-cell} ipython3
:tags: [remove-cell]

# The index of 5 rows that has smallest distance to the new point
min_5_idx = np.argpartition(my_distances4, 5)[:5]

neighbor_df_list = []
for idx in min_5_idx:
    neighbor_df = pd.concat(
        (
            cancer.loc[idx, attrs + ["Class"]],
            perim_concav_with_new_point_df4.loc[len(cancer), attrs + ["Class"]],
        ),
        axis=1,
    ).T
    neighbor_df_list.append(neighbor_df)
```

```{code-cell} ipython3
:tags: [remove-input]

fig = px.scatter_3d(
    perim_concav_with_new_point_df4,
    x="Perimeter",
    y="Concavity",
    z="Symmetry",
    color="Class",
    symbol="Class",
    opacity=0.5,
)
# specify trace names and symbols in a dict
symbols = {"Malignant": "circle", "Benign": "circle", "Unknown": "diamond"}

# set all symbols in fig
for i, d in enumerate(fig.data):
    fig.data[i].marker.symbol = symbols[fig.data[i].name]

# specify trace names and colors in a dict
colors = {"Malignant": "#ff7f0e", "Benign": "#1f77b4", "Unknown": "red"}

# set all colors in fig
for i, d in enumerate(fig.data):
    fig.data[i].marker.color = colors[fig.data[i].name]

# set a fixed custom marker size
fig.update_traces(marker={"size": 5})

# add lines
for neighbor_df in neighbor_df_list:
    fig.add_trace(
        go.Scatter3d(
            x=neighbor_df["Perimeter"],
            y=neighbor_df["Concavity"],
            z=neighbor_df["Symmetry"],
            line_color=colors[neighbor_df.iloc[0]["Class"]],
            name=neighbor_df.iloc[0]["Class"],
            mode="lines",
            line=dict(width=2),
            showlegend=False,
        )
    )


# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=1), template="plotly_white")

# if HTML, use the plotly 3d image; if PDF, use static image
if "BOOK_BUILD_TYPE" in os.environ and os.environ["BOOK_BUILD_TYPE"] == "PDF":
    glue("fig:05-more", Image("img/classification1/plot3d_knn_classification.png"))
else:
    glue("fig:05-more", fig)
```

```{figure} data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7
:name: fig:05-more
:figclass: caption-hack

3D scatter plot of the standardized symmetry, concavity, and perimeter
variables. Note that in general we recommend against using 3D visualizations;
here we show the data in 3D only to illustrate what higher dimensions and
nearest neighbors look like, for learning purposes.
```

+++

### Summary of K-nearest neighbors algorithm

In order to classify a new observation using a K-nearest neighbors classifier, we have to do the following:

1. Compute the distance between the new observation and each observation in the training set.
2. Find the $K$ rows corresponding to the $K$ smallest distances.
3. Classify the new observation based on a majority vote of the neighbor classes.

+++

## K-nearest neighbors with `scikit-learn`

```{index} scikit-learn
```

Coding the K-nearest neighbors algorithm in Python ourselves can get complicated,
especially if we want to handle multiple classes, more than two variables,
or predict the class for multiple new observations. Thankfully, in Python,
the K-nearest neighbors algorithm is
implemented in [the `scikit-learn` Python package](https://scikit-learn.org/stable/index.html) {cite:p}`sklearn_api` along with
many [other models](https://scikit-learn.org/stable/user_guide.html) that you will encounter in this and future chapters of the book. Using the functions
in the `scikit-learn` package (named `sklearn` in Python) will help keep our code simple, readable and accurate; the
less we have to code ourselves, the fewer mistakes we will likely make.
Before getting started with K-nearest neighbors, we need to tell the `sklearn` package
that we prefer using `pandas` data frames over regular arrays via the `set_config` function.
```{note}
You will notice a new way of importing functions in the code below: `from ... import ...`. This lets us
import *just* `set_config` from `sklearn`, and then call `set_config` without any package prefix.
We will import functions using `from` extensively throughout
this and subsequent chapters to avoid very long names from `scikit-learn`
that clutter the code
(like `sklearn.neighbors.KNeighborsClassifier`, which has 38 characters!).
```

```{code-cell} ipython3
from sklearn import set_config

# Output dataframes instead of arrays
set_config(transform_output="pandas")
```

We can now get started with K-nearest neighbors. The first step is to
 import the `KNeighborsClassifier` from the `sklearn.neighbors` module.

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier
```

Let's walk through how to use `KNeighborsClassifier` to perform K-nearest neighbors classification.
We will use the `cancer` data set from above, with
perimeter and concavity as predictors and $K = 5$ neighbors to build our classifier. Then
we will use the classifier to predict the diagnosis label for a new observation with
perimeter 0, concavity 3.5, and an unknown diagnosis label. Let's pick out our two desired
predictor variables and class label and store them with the name `cancer_train`:

```{code-cell} ipython3
cancer_train = cancer[["Class", "Perimeter", "Concavity"]]
cancer_train
```

```{index} scikit-learn; model object, scikit-learn; KNeighborsClassifier
```

Next, we create a *model object* for K-nearest neighbors classification
by creating a `KNeighborsClassifier` instance, specifying that we want to use $K = 5$ neighbors;
we will discuss how to choose $K$ in the next chapter.

```{note}
You can specify the `weights` argument in order to control
how neighbors vote when classifying a new observation. The default is `"uniform"`, where
each of the $K$ nearest neighbors gets exactly 1 vote as described above. Other choices,
which weigh each neighbor's vote differently, can be found on
[the `scikit-learn` website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier).
```

```{code-cell} ipython3
knn = KNeighborsClassifier(n_neighbors=5)
knn
```

```{index} scikit-learn; fit, scikit-learn; predictors, scikit-learn; response
```

In order to fit the model on the breast cancer data, we need to call `fit` on
the model object. The `X` argument is used to specify the data for the predictor
variables, while the `y` argument is used to specify the data for the response variable.
So below, we set `X=cancer_train[["Perimeter", "Concavity"]]` and
`y=cancer_train["Class"]` to specify that `Class` is the response
variable (the one we want to predict), and both `Perimeter` and `Concavity` are
to be used as the predictors. Note that the `fit` function might look like it does not
do much from the outside, but it is actually doing all the heavy lifting to train
the K-nearest neighbors model, and modifies the `knn` model object.

```{code-cell} ipython3
knn.fit(X=cancer_train[["Perimeter", "Concavity"]], y=cancer_train["Class"]);
```

```{index} scikit-learn; predict
```

After using the `fit` function, we can make a prediction on a new observation
by calling `predict` on the classifier object, passing the new observation
itself. As above, when we ran the K-nearest neighbors classification
algorithm manually, the `knn` model object classifies the new observation as
"Malignant". Note that the `predict` function outputs an `array` with the
model's prediction; you can actually make multiple predictions at the same
time using the `predict` function, which is why the output is stored as an `array`.

```{code-cell} ipython3
new_obs = pd.DataFrame({"Perimeter": [0], "Concavity": [3.5]})
knn.predict(new_obs)
```

Is this predicted malignant label the actual class for this observation?
Well, we don't know because we do not have this
observation's diagnosis&mdash; that is what we were trying to predict! The
classifier's prediction is not necessarily correct, but in the next chapter, we will
learn ways to quantify how accurate we think our predictions are.

+++

## Data preprocessing with `scikit-learn`

### Centering and scaling

```{index} scaling
```

When using K-nearest neighbors classification, the *scale* of each variable
(i.e., its size and range of values) matters. Since the classifier predicts
classes by identifying observations nearest to it, any variables with
a large scale will have a much larger effect than variables with a small
scale. But just because a variable has a large scale *doesn't mean* that it is
more important for making accurate predictions. For example, suppose you have a
data set with two features, salary (in dollars) and years of education, and
you want to predict the corresponding type of job. When we compute the
neighbor distances, a difference of \$1000 is huge compared to a difference of
10 years of education. But for our conceptual understanding and answering of
the problem, it's the opposite; 10 years of education is huge compared to a
difference of \$1000 in yearly salary!

+++

```{index} centering
```

In many other predictive models, the *center* of each variable (e.g., its mean)
matters as well. For example, if we had a data set with a temperature variable
measured in degrees Kelvin, and the same data set with temperature measured in
degrees Celsius, the two variables would differ by a constant shift of 273
(even though they contain exactly the same information). Likewise, in our
hypothetical job classification example, we would likely see that the center of
the salary variable is in the tens of thousands, while the center of the years
of education variable is in the single digits. Although this doesn't affect the
K-nearest neighbors classification algorithm, this large shift can change the
outcome of using many other predictive models.

```{index} standardization; K-nearest neighbors
```

To scale and center our data, we need to find
our variables' *mean* (the average, which quantifies the "central" value of a
set of numbers) and *standard deviation* (a number quantifying how spread out values are).
For each observed value of the variable, we subtract the mean (i.e., center the variable)
and divide by the standard deviation (i.e., scale the variable). When we do this, the data
is said to be *standardized*, and all variables in a data set will have a mean of 0
and a standard deviation of 1. To illustrate the effect that standardization can have on the K-nearest
neighbors algorithm, we will read in the original, unstandardized Wisconsin breast
cancer data set; we have been using a standardized version of the data set up
until now. We will apply the same initial wrangling steps as we did earlier,
and to keep things simple we will just use the `Area`, `Smoothness`, and `Class`
variables:

```{code-cell} ipython3
unscaled_cancer = pd.read_csv("data/wdbc_unscaled.csv")[["Class", "Area", "Smoothness"]]
unscaled_cancer["Class"] = unscaled_cancer["Class"].replace({
   "M" : "Malignant",
   "B" : "Benign"
})
unscaled_cancer
```

Looking at the unscaled and uncentered data above, you can see that the differences
between the values for area measurements are much larger than those for
smoothness. Will this affect predictions? In order to find out, we will create a scatter plot of these two
predictors (colored by diagnosis) for both the unstandardized data we just
loaded, and the standardized version of that same data. But first, we need to
standardize the `unscaled_cancer` data set with `scikit-learn`.

```{index} see: Pipeline; scikit-learn
```

```{index} see: make_column_transformer; scikit-learn
```

```{index} scikit-learn;Pipeline, scikit-learn; make_column_transformer
```

The `scikit-learn` framework provides a collection of *preprocessors* used to manipulate
data in the [`preprocessing` module](https://scikit-learn.org/stable/modules/preprocessing.html).
Here we will use the `StandardScaler` transformer to standardize the predictor variables in
the `unscaled_cancer` data. In order to tell the `StandardScaler` which variables to standardize,
we wrap it in a
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) object
using the [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer) function.
`ColumnTransformer` objects also enable the use of multiple preprocessors at
once, which is especially handy when you want to apply different preprocessing to each of the predictor variables.
The primary argument of the `make_column_transformer` function is a sequence of
pairs of (1) a preprocessor, and (2) the columns to which you want to apply that preprocessor.
In the present case, we just have the one `StandardScaler` preprocessor to apply to the `Area` and `Smoothness` columns.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
preprocessor
```

```{index} scikit-learn; make_column_transformer, scikit-learn; StandardScaler 
```

```{index} see: StandardScaler; scikit-learn
```

```{index} scikit-learn; fit, scikit-learn; make_column_selector, scikit-learn; StandardScaler
```

You can see that the preprocessor includes a single standardization step
that is applied to the `Area` and `Smoothness` columns.
Note that here we specified which columns to apply the preprocessing step to
by individual names; this approach can become quite difficult, e.g., when we have many
predictor variables. Rather than writing out the column names individually,
we can instead use the
[`make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector) function. For
example, if we wanted to standardize all *numerical* predictors,
we would use `make_column_selector` and specify the `dtype_include` argument to be `"number"`.
This creates a preprocessor equivalent to the one we created previously.

```{code-cell} ipython3
from sklearn.compose import make_column_selector

preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include="number")),
)
preprocessor
```

```{index} see: fit ; scikit-learn
```

```{index} scikit-learn; transform
```

We are now ready to standardize the numerical predictor columns in the `unscaled_cancer` data frame.
This happens in two steps. We first use the `fit` function to compute the values necessary to apply
the standardization (the mean and standard deviation of each variable), passing the `unscaled_cancer` data as an argument.
Then we use the `transform` function to actually apply the standardization.
It may seem a bit unnecessary to use two steps---`fit` *and* `transform`---to standardize the data.
However, we do this in two steps so that we can specify a different data set in the `transform` step if we want.
This enables us to compute the quantities needed to standardize using one data set, and then
apply that standardization to another data set.

```{code-cell} ipython3
preprocessor.fit(unscaled_cancer)
scaled_cancer = preprocessor.transform(unscaled_cancer)
scaled_cancer
```
```{code-cell} ipython3
:tags: [remove-cell]
glue("scaled-cancer-column-0", '"'+scaled_cancer.columns[0]+'"')
glue("scaled-cancer-column-1", '"'+scaled_cancer.columns[1]+'"')
```
It looks like our `Smoothness` and `Area` variables have been standardized. Woohoo!
But there are two important things to notice about the new `scaled_cancer` data frame. First, it only keeps
the columns from the input to `transform` (here, `unscaled_cancer`) that had a preprocessing step applied
to them. The default behavior of the `ColumnTransformer` that we build using `make_column_transformer`
is to *drop* the remaining columns. This default behavior works well with the rest of `sklearn` (as we will see below
in {numref}`08:puttingittogetherworkflow`), but for visualizing the result of preprocessing it can be useful to keep the other columns
in our original data frame, such as the `Class` variable here.
To keep other columns, we need to set the `remainder` argument to `"passthrough"` in the `make_column_transformer` function.
Furthermore, you can see that the new column names---{glue:text}`scaled-cancer-column-0`
and {glue:text}`scaled-cancer-column-1`---include the name
of the preprocessing step separated by underscores. This default behavior is useful in `sklearn` because we sometimes want to apply
multiple different preprocessing steps to the same columns; but again, for visualization it can be useful to preserve
the original column names. To keep original column names, we need to set the `verbose_feature_names_out` argument to `False`.

```{note}
Only specify the `remainder` and `verbose_feature_names_out` arguments when you want to examine the result
of your preprocessing step. In most cases, you should leave these arguments at their default values.
```

```{code-cell} ipython3
preprocessor_keep_all = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include="number")),
    remainder="passthrough",
    verbose_feature_names_out=False
)
preprocessor_keep_all.fit(unscaled_cancer)
scaled_cancer_all = preprocessor_keep_all.transform(unscaled_cancer)
scaled_cancer_all
```

You may wonder why we are doing so much work just to center and
scale our variables. Can't we just manually scale and center the `Area` and
`Smoothness` variables ourselves before building our K-nearest neighbors model? Well,
technically *yes*; but doing so is error-prone.  In particular, we might
accidentally forget to apply the same centering / scaling when making
predictions, or accidentally apply a *different* centering / scaling than what
we used while training. Proper use of a `ColumnTransformer` helps keep our code simple,
readable, and error-free. Furthermore, note that using `fit` and `transform` on
the preprocessor is required only when you want to inspect the result of the
preprocessing steps
yourself. You will see further on in
{numref}`08:puttingittogetherworkflow` that `scikit-learn` provides tools to
automatically streamline the preprocesser and the model so that you can call `fit`
and `transform` on the `Pipeline` as necessary without additional coding effort.

{numref}`fig:05-scaling-plt` shows the two scatter plots side-by-side&mdash;one for `unscaled_cancer` and one for
`scaled_cancer`. Each has the same new observation annotated with its $K=3$ nearest neighbors.
In the original unstandardized data plot, you can see some odd choices
for the three nearest neighbors. In particular, the "neighbors" are visually
well within the cloud of benign observations, and the neighbors are all nearly
vertically aligned with the new observation (which is why it looks like there
is only one black line on this plot). {numref}`fig:05-scaling-plt-zoomed`
shows a close-up of that region on the unstandardized plot. Here the computation of nearest
neighbors is dominated by the much larger-scale area variable. The plot for standardized data
on the right in {numref}`fig:05-scaling-plt` shows a much more intuitively reasonable
selection of nearest neighbors. Thus, standardizing the data can change things
in an important way when we are using predictive algorithms.
Standardizing your data should be a part of the preprocessing you do
before predictive modeling and you should always think carefully about your problem domain and
whether you need to standardize your data.

```{code-cell} ipython3
:tags: [remove-cell]

def class_dscp(x):
    if x == "M":
        return "Malignant"
    elif x == "B":
        return "Benign"
    else:
        return x


attrs = ["Area", "Smoothness"]
new_obs = pd.DataFrame({"Class": ["Unknown"], "Area": 400, "Smoothness": 0.135})
unscaled_cancer["Class"] = unscaled_cancer["Class"].apply(class_dscp)
area_smoothness_new_df = pd.concat((unscaled_cancer, new_obs), ignore_index=True)
my_distances = euclidean_distances(area_smoothness_new_df[attrs])[
    len(unscaled_cancer)
][:-1]
area_smoothness_new_point = (
    alt.Chart(
        area_smoothness_new_df,
        title=alt.TitleParams(text="Unstandardized data", anchor="start"),
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area"),
        y=alt.Y("Smoothness"),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None))
    )
)

# The index of 3 rows that has smallest distance to the new point
min_3_idx = np.argpartition(my_distances, 3)[:3]
neighbor1 = pd.concat([
    unscaled_cancer.loc[[min_3_idx[0]], attrs],
    new_obs[attrs],
])
neighbor2 = pd.concat([
    unscaled_cancer.loc[[min_3_idx[1]], attrs],
    new_obs[attrs],
])
neighbor3 = pd.concat([
    unscaled_cancer.loc[[min_3_idx[2]], attrs],
    new_obs[attrs],
])

line1 = (
    alt.Chart(neighbor1)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line2 = (
    alt.Chart(neighbor2)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line3 = (
    alt.Chart(neighbor3)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)

area_smoothness_new_point = area_smoothness_new_point + line1 + line2 + line3
```

```{code-cell} ipython3
:tags: [remove-cell]

attrs = ["Area", "Smoothness"]
new_obs_scaled = pd.DataFrame({"Class": ["Unknown"], "Area": -0.72, "Smoothness": 2.8})
scaled_cancer_all["Class"] = scaled_cancer_all["Class"].apply(class_dscp)
area_smoothness_new_df_scaled = pd.concat(
    (scaled_cancer_all, new_obs_scaled), ignore_index=True
)
my_distances_scaled = euclidean_distances(area_smoothness_new_df_scaled[attrs])[
    len(scaled_cancer_all)
][:-1]
area_smoothness_new_point_scaled = (
    alt.Chart(
        area_smoothness_new_df_scaled,
        title=alt.TitleParams(text="Standardized data", anchor="start"),
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area", title="Area (standardized)"),
        y=alt.Y("Smoothness", title="Smoothness (standardized)"),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None))
    )
)
min_3_idx_scaled = np.argpartition(my_distances_scaled, 3)[:3]
neighbor1_scaled = pd.concat([
    scaled_cancer_all.loc[[min_3_idx_scaled[0]], attrs],
    new_obs_scaled[attrs],
])
neighbor2_scaled = pd.concat([
    scaled_cancer_all.loc[[min_3_idx_scaled[1]], attrs],
    new_obs_scaled[attrs],
])
neighbor3_scaled = pd.concat([
    scaled_cancer_all.loc[[min_3_idx_scaled[2]], attrs],
    new_obs_scaled[attrs],
])

line1_scaled = (
    alt.Chart(neighbor1_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line2_scaled = (
    alt.Chart(neighbor2_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)
line3_scaled = (
    alt.Chart(neighbor3_scaled)
    .mark_line()
    .encode(x="Area", y="Smoothness", color=alt.value("black"))
)

area_smoothness_new_point_scaled = (
    area_smoothness_new_point_scaled + line1_scaled + line2_scaled + line3_scaled
)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue(
    "fig:05-scaling-plt",
    area_smoothness_new_point | area_smoothness_new_point_scaled
)
```

:::{glue:figure} fig:05-scaling-plt
:name: fig:05-scaling-plt

Comparison of K = 3 nearest neighbors with unstandardized and standardized data.
:::

```{code-cell} ipython3
:tags: [remove-cell]

zoom_area_smoothness_new_point = (
    alt.Chart(
        area_smoothness_new_df,
        title=alt.TitleParams(text="Unstandardized data", anchor="start"),
    )
    .mark_point(clip=True, opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Area", scale=alt.Scale(domain=(395, 405))),
        y=alt.Y("Smoothness", scale=alt.Scale(domain=(0.08, 0.14))),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None))
    )
)
zoom_area_smoothness_new_point + line1 + line2 + line3
glue("fig:05-scaling-plt-zoomed", (zoom_area_smoothness_new_point + line1 + line2 + line3))
```

:::{glue:figure} fig:05-scaling-plt-zoomed
:name: fig:05-scaling-plt-zoomed

Close-up of three nearest neighbors for unstandardized data.
:::

+++

### Balancing

```{index} balance, imbalance
```

Another potential issue in a data set for a classifier is *class imbalance*,
i.e., when one label is much more common than another. Since classifiers like
the K-nearest neighbors algorithm use the labels of nearby points to predict
the label of a new point, if there are many more data points with one label
overall, the algorithm is more likely to pick that label in general (even if
the "pattern" of data suggests otherwise). Class imbalance is actually quite a
common and important problem: from rare disease diagnosis to malicious email
detection, there are many cases in which the "important" class to identify
(presence of disease, malicious email) is much rarer than the "unimportant"
class (no disease, normal email).

```{index} concat
```

To better illustrate the problem, let's revisit the scaled breast cancer data,
`cancer`; except now we will remove many of the observations of malignant tumors, simulating
what the data would look like if the cancer was rare. We will do this by
picking only 3 observations from the malignant group, and keeping all
of the benign observations. We choose these 3 observations using the `.head()`
method, which takes the number of rows to select from the top.
We will then use the [`concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)
function from `pandas` to glue the two resulting filtered
data frames back together. The `concat` function *concatenates* data frames
along an axis. By default, it concatenates the data frames vertically along `axis=0` yielding a single
*taller* data frame, which is what we want to do here. If we instead wanted to concatenate horizontally
to produce a *wider* data frame, we would specify `axis=1`.
The new imbalanced data is shown in {numref}`fig:05-unbalanced`,
and we print the counts of the classes using the `value_counts` function.

```{code-cell} ipython3
:tags: ["remove-output"]
rare_cancer = pd.concat((
    cancer[cancer["Class"] == "Benign"],
    cancer[cancer["Class"] == "Malignant"].head(3)
))

rare_plot = alt.Chart(rare_cancer).mark_circle().encode(
    x=alt.X("Perimeter").title("Perimeter (standardized)"),
    y=alt.Y("Concavity").title("Concavity (standardized)"),
    color=alt.Color("Class").title("Diagnosis")
)
rare_plot
```

```{code-cell} ipython3
:tags: ["remove-cell"]
glue("fig:05-unbalanced", rare_plot)
```

:::{glue:figure} fig:05-unbalanced
:name: fig:05-unbalanced

Imbalanced data.
:::

```{code-cell} ipython3
rare_cancer["Class"].value_counts()
```

+++

Suppose we now decided to use $K = 7$ in K-nearest neighbors classification.
With only 3 observations of malignant tumors, the classifier
will *always predict that the tumor is benign, no matter what its concavity and perimeter
are!* This is because in a majority vote of 7 observations, at most 3 will be
malignant (we only have 3 total malignant observations), so at least 4 must be
benign, and the benign vote will always win. For example, {numref}`fig:05-upsample`
shows what happens for a new tumor observation that is quite close to three observations
in the training data that were tagged as malignant.

```{code-cell} ipython3
:tags: [remove-cell]

attrs = ["Perimeter", "Concavity"]
new_point = [2, 2]
new_point_df = pd.DataFrame(
    {"Class": ["Unknown"], "Perimeter": new_point[0], "Concavity": new_point[1]}
)
rare_cancer["Class"] = rare_cancer["Class"].apply(class_dscp)
rare_cancer_with_new_df = pd.concat((rare_cancer, new_point_df), ignore_index=True)
my_distances = euclidean_distances(rare_cancer_with_new_df[attrs])[
    len(rare_cancer)
][:-1]

# First layer: scatter plot, with unknwon point labeled as red "unknown" diamond
rare_plot = (
    alt.Chart(
        rare_cancer_with_new_df
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color(
            "Class",
            title="Diagnosis",
        ),
        shape=alt.Shape(
            "Class", scale=alt.Scale(range=["circle", "circle", "diamond"])
        ),
        size=alt.condition("datum.Class == 'Unknown'", alt.value(80), alt.value(30)),
        stroke=alt.condition("datum.Class == 'Unknown'", alt.value("black"), alt.value(None))
    )
)

# Find the 7 NNs
min_7_idx = np.argpartition(my_distances, 7)[:7]

# For loop: each iteration adds a line segment of corresponding color
for i in range(7):
    clr = "#1f77b4"
    if rare_cancer.iloc[min_7_idx[i], :]["Class"] == "Malignant":
        clr = "#ff7f0e"
    neighbor = pd.concat([
        rare_cancer.iloc[[min_7_idx[i]], :][attrs],
        new_point_df[attrs],
    ])
    rare_plot = rare_plot + (
        alt.Chart(neighbor)
        .mark_line(opacity=0.3)
        .encode(x="Perimeter", y="Concavity", color=alt.value(clr))
    )

glue("fig:05-upsample", rare_plot)
```

:::{glue:figure} fig:05-upsample
:name: fig:05-upsample

Imbalanced data with 7 nearest neighbors to a new observation highlighted.
:::

+++

{numref}`fig:05-upsample-2` shows what happens if we set the background color of
each area of the plot to the prediction the K-nearest neighbors
classifier would make for a new observation at that location. We can see that the decision is
always "benign," corresponding to the blue color.

```{code-cell} ipython3
:tags: [remove-cell]

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X=rare_cancer[["Perimeter", "Concavity"]], y=rare_cancer["Class"])

# create a prediction pt grid
per_grid = np.linspace(
    rare_cancer["Perimeter"].min() * 1.05, rare_cancer["Perimeter"].max() * 1.05, 50
)
con_grid = np.linspace(
    rare_cancer["Concavity"].min() * 1.05, rare_cancer["Concavity"].max() * 1.05, 50
)
pcgrid = np.array(np.meshgrid(per_grid, con_grid)).reshape(2, -1).T
pcgrid = pd.DataFrame(pcgrid, columns=["Perimeter", "Concavity"])
pcgrid

knnPredGrid = knn.predict(pcgrid)
prediction_table = pcgrid.copy()
prediction_table["Class"] = knnPredGrid
prediction_table

# create the scatter plot
rare_plot = (
    alt.Chart(
        rare_cancer,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
prediction_plot = (
    alt.Chart(
        prediction_table,
        title="Imbalanced data",
    )
    .mark_point(opacity=0.05, filled=True, size=300)
    .encode(
        x=alt.X(
            "Perimeter",
            title="Perimeter (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Perimeter"].min() * 1.05, rare_cancer["Perimeter"].max() * 1.05),
                nice=False
            ),
        ),
        y=alt.Y(
            "Concavity",
            title="Concavity (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Concavity"].min() * 1.05, rare_cancer["Concavity"].max() * 1.05),
                nice=False
            ),
        ),
        color=alt.Color("Class", title="Diagnosis"),
    )
)
#rare_plot + prediction_plot
glue("fig:05-upsample-2", (rare_plot + prediction_plot))
```

:::{glue:figure} fig:05-upsample-2
:name: fig:05-upsample-2

Imbalanced data with background color indicating the decision of the classifier and the points represent the labeled data.
:::

+++

```{index} oversampling, DataFrame; sample
```

Despite the simplicity of the problem, solving it in a statistically sound manner is actually
fairly nuanced, and a careful treatment would require a lot more detail and mathematics than we will cover in this textbook.
For the present purposes, it will suffice to rebalance the data by *oversampling* the rare class.
In other words, we will replicate rare observations multiple times in our data set to give them more
voting power in the K-nearest neighbors algorithm. In order to do this, we will
first separate the classes out into their own data frames by filtering.
Then, we will
use the `sample` method on the rare class data frame to increase the number of `Malignant` observations to be the same as the number
of `Benign` observations. We set the `n` argument to be the number of `Malignant` observations we want, and set `replace=True`
to indicate that we are sampling with replacement.
Finally, we use the `value_counts` method to see that our classes are now balanced.
Note that `sample` picks which data to replicate *randomly*; we will learn more about properly handling randomness
in data analysis in {numref}`Chapter %s <classification2>`.

```{code-cell} ipython3
:tags: [remove-cell]
# hidden seed call to make the below resample reproducible
# we haven't taught students about seeds / prngs yet, so
# for now just hide this.
np.random.seed(1)
```

```{code-cell} ipython3
malignant_cancer = rare_cancer[rare_cancer["Class"] == "Malignant"]
benign_cancer = rare_cancer[rare_cancer["Class"] == "Benign"]
malignant_cancer_upsample = malignant_cancer.sample(
    n=benign_cancer.shape[0], replace=True
)
upsampled_cancer = pd.concat((malignant_cancer_upsample, benign_cancer))
upsampled_cancer["Class"].value_counts()
```

Now suppose we train our K-nearest neighbors classifier with $K=7$ on this *balanced* data.
{numref}`fig:05-upsample-plot` shows what happens now when we set the background color
of each area of our scatter plot to the decision the K-nearest neighbors
classifier would make. We can see that the decision is more reasonable; when the points are close
to those labeled malignant, the classifier predicts a malignant tumor, and vice versa when they are
closer to the benign tumor observations.

```{code-cell} ipython3
:tags: [remove-cell]

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(
    X=upsampled_cancer[["Perimeter", "Concavity"]], y=upsampled_cancer["Class"]
)

# create a prediction pt grid
knnPredGrid = knn.predict(pcgrid)
prediction_table = pcgrid
prediction_table["Class"] = knnPredGrid

# create the scatter plot
rare_plot = (
    alt.Chart(rare_cancer)
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X(
            "Perimeter",
            title="Perimeter (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Perimeter"].min() * 1.05, rare_cancer["Perimeter"].max() * 1.05),
                nice=False
            ),
        ),
        y=alt.Y(
            "Concavity",
            title="Concavity (standardized)",
            scale=alt.Scale(
                domain=(rare_cancer["Concavity"].min() * 1.05, rare_cancer["Concavity"].max() * 1.05),
                nice=False
            ),
        ),
        color=alt.Color("Class", title="Diagnosis"),
    )
)

# add a prediction layer, also scatter plot
upsampled_plot = (
    alt.Chart(prediction_table)
    .mark_point(opacity=0.05, filled=True, size=300)
    .encode(
        x=alt.X("Perimeter", title="Perimeter (standardized)"),
        y=alt.Y("Concavity", title="Concavity (standardized)"),
        color=alt.Color("Class", title="Diagnosis"),
    )
)
#rare_plot + upsampled_plot
glue("fig:05-upsample-plot", (rare_plot + upsampled_plot))
```

:::{glue:figure} fig:05-upsample-plot
:name: fig:05-upsample-plot

Upsampled data with background color indicating the decision of the classifier.
:::

### Missing data

```{index} missing data
```

One of the most common issues in real data sets in the wild is *missing data*,
i.e., observations where the values of some of the variables were not recorded.
Unfortunately, as common as it is, handling missing data properly is very
challenging and generally relies on expert knowledge about the data, setting,
and how the data were collected. One typical challenge with missing data is
that missing entries can be *informative*: the very fact that an entries were
missing is related to the values of other variables.  For example, survey
participants from a marginalized group of people may be less likely to respond
to certain kinds of questions if they fear that answering honestly will come
with negative consequences. In that case, if we were to simply throw away data
with missing entries, we would bias the conclusions of the survey by
inadvertently removing many members of that group of respondents.  So ignoring
this issue in real problems can easily lead to misleading analyses, with
detrimental impacts.  In this book, we will cover only those techniques for
dealing with missing entries in situations where missing entries are just
"randomly missing", i.e., where the fact that certain entries are missing
*isn't related to anything else* about the observation.

Let's load and examine a modified subset of the tumor image data
that has a few missing entries:

```{code-cell} ipython3
missing_cancer = pd.read_csv("data/wdbc_missing.csv")[["Class", "Radius", "Texture", "Perimeter"]]
missing_cancer["Class"] = missing_cancer["Class"].replace({
   "M" : "Malignant",
   "B" : "Benign"
})
missing_cancer
```

Recall that K-nearest neighbors classification makes predictions by computing
the straight-line distance to nearby training observations, and hence requires
access to the values of *all* variables for *all* observations in the training
data.  So how can we perform K-nearest neighbors classification in the presence
of missing data?  Well, since there are not too many observations with missing
entries, one option is to simply remove those observations prior to building
the K-nearest neighbors classifier. We can accomplish this by using the
`dropna` method prior to working with the data.

```{index} missing data; dropna
```

```{code-cell} ipython3
no_missing_cancer = missing_cancer.dropna()
no_missing_cancer
```

However, this strategy will not work when many of the rows have missing
entries, as we may end up throwing away too much data. In this case, another
possible approach is to *impute* the missing entries, i.e., fill in synthetic
values based on the other observations in the data set. One reasonable choice
is to perform *mean imputation*, where missing entries are filled in using the
mean of the present entries in each variable. To perform mean imputation, we
use a `SimpleImputer` transformer with the default arguments, and use
`make_column_transformer` to indicate which columns need imputation.

```{index} scikit-learn; SimpleImputer, missing data;mean imputation
```

```{code-cell} ipython3
from sklearn.impute import SimpleImputer

preprocessor = make_column_transformer(
    (SimpleImputer(), ["Radius", "Texture", "Perimeter"]),
    verbose_feature_names_out=False
)
preprocessor
```

To visualize what mean imputation does, let's just apply the transformer directly to the `missing_cancer`
data frame using the `fit` and `transform` functions.  The imputation step fills in the missing
entries with the mean values of their corresponding variables.

```{code-cell} ipython3
preprocessor.fit(missing_cancer)
imputed_cancer = preprocessor.transform(missing_cancer)
imputed_cancer
```

Many other options for missing data imputation can be found in
[the `scikit-learn` documentation](https://scikit-learn.org/stable/modules/impute.html).  However
you decide to handle missing data in your data analysis, it is always crucial
to think critically about the setting, how the data were collected, and the
question you are answering.

+++

(08:puttingittogetherworkflow)=
## Putting it together in a `Pipeline`

```{index} scikit-learn; Pipeline
```

The `scikit-learn` package collection also provides the [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?highlight=pipeline#sklearn.pipeline.Pipeline),
a  way to chain together multiple data analysis steps without a lot of otherwise necessary code for intermediate steps.
To illustrate the whole workflow, let's start from scratch with the `wdbc_unscaled.csv` data.
First we will load the data, create a model, and specify a preprocessor for the data.

```{code-cell} ipython3
# load the unscaled cancer data, make Class readable
unscaled_cancer = pd.read_csv("data/wdbc_unscaled.csv")
unscaled_cancer["Class"] = unscaled_cancer["Class"].replace({
   "M" : "Malignant",
   "B" : "Benign"
})
unscaled_cancer

# create the K-NN model
knn = KNeighborsClassifier(n_neighbors=7)

# create the centering / scaling preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), ["Area", "Smoothness"]),
)
```

```{index} scikit-learn; make_pipeline, scikit-learn; fit
```

Next we place these steps in a `Pipeline` using
the [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) function.
The `make_pipeline` function takes a list of steps to apply in your data analysis; in this
case, we just have the `preprocessor` and `knn` steps.
Finally, we call `fit` on the pipeline.
Notice that we do not need to separately call `fit` and `transform` on the `preprocessor`; the
pipeline handles doing this properly for us.
Also notice that when we call `fit` on the pipeline, we can pass
the whole `unscaled_cancer` data frame to the `X` argument, since the preprocessing
step drops all the variables except the two we listed: `Area` and `Smoothness`.
For the `y` response variable argument, we pass the `unscaled_cancer["Class"]` series as before.

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline

knn_pipeline = make_pipeline(preprocessor, knn)
knn_pipeline.fit(
    X=unscaled_cancer,
    y=unscaled_cancer["Class"]
)
knn_pipeline
```

As before, the fit object lists the function that trains the model. But now the fit object also includes information about
the overall workflow, including the standardization preprocessing step.
In other words, when we use the `predict` function with the `knn_pipeline` object to make a prediction for a new
observation, it will first apply the same preprocessing steps to the new observation.
As an example, we will predict the class label of two new observations:
one with `Area = 500` and `Smoothness = 0.075`, and one with `Area = 1500` and `Smoothness = 0.1`.

```{code-cell} ipython3
new_observation = pd.DataFrame({"Area": [500, 1500], "Smoothness": [0.075, 0.1]})
prediction = knn_pipeline.predict(new_observation)
prediction
```

The classifier predicts that the first observation is benign, while the second is
malignant. {numref}`fig:05-workflow-plot` visualizes the predictions that this
trained K-nearest neighbors model will make on a large range of new observations.
Although you have seen colored prediction map visualizations like this a few times now,
we have not included the code to generate them, as it is a little bit complicated.
For the interested reader who wants a learning challenge, we now include it below.
The basic idea is to create a grid of synthetic new observations using the `meshgrid` function from `numpy`,
predict the label of each, and visualize the predictions with a colored scatter having a very high transparency
(low `opacity` value) and large point radius. See if you can figure out what each line is doing!

```{note}
Understanding this code is not required for the remainder of the
textbook. It is included for those readers who would like to use similar
visualizations in their own data analyses.
```

```{code-cell} ipython3
:tags: [remove-output]
import numpy as np

# create the grid of area/smoothness vals, and arrange in a data frame
are_grid = np.linspace(
    unscaled_cancer["Area"].min() * 0.95, unscaled_cancer["Area"].max() * 1.05, 50
)
smo_grid = np.linspace(
    unscaled_cancer["Smoothness"].min() * 0.95, unscaled_cancer["Smoothness"].max() * 1.05, 50
)
asgrid = np.array(np.meshgrid(are_grid, smo_grid)).reshape(2, -1).T
asgrid = pd.DataFrame(asgrid, columns=["Area", "Smoothness"])

# use the fit workflow to make predictions at the grid points
knnPredGrid = knn_pipeline.predict(asgrid)

# bind the predictions as a new column with the grid points
prediction_table = asgrid.copy()
prediction_table["Class"] = knnPredGrid

# plot:
# 1. the colored scatter of the original data
unscaled_plot = alt.Chart(unscaled_cancer).mark_point(
    opacity=0.6,
    filled=True,
    size=40
).encode(
    x=alt.X("Area")
        .scale(
            nice=False,
            domain=(
                unscaled_cancer["Area"].min() * 0.95,
                unscaled_cancer["Area"].max() * 1.05
            )
        ),
    y=alt.Y("Smoothness")
        .scale(
            nice=False,
            domain=(
                unscaled_cancer["Smoothness"].min() * 0.95,
                unscaled_cancer["Smoothness"].max() * 1.05
            )
        ),
    color=alt.Color("Class").title("Diagnosis")
)

# 2. the faded colored scatter for the grid points
prediction_plot = alt.Chart(prediction_table).mark_point(
    opacity=0.05,
    filled=True,
    size=300
).encode(
    x="Area",
    y="Smoothness",
    color=alt.Color("Class").title("Diagnosis")
)
unscaled_plot + prediction_plot
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("fig:05-workflow-plot", (unscaled_plot + prediction_plot))
```

:::{glue:figure} fig:05-workflow-plot
:name: fig:05-workflow-plot

Scatter plot of smoothness versus area where background color indicates the decision of the classifier.
:::

(classification2)=
# Classification II: evaluation & tuning

```{code-cell} ipython3
:tags: [remove-cell]

from chapter_preamble import *
```

## Overview
This chapter continues the introduction to predictive modeling through
classification. While the previous chapter covered training and data
preprocessing, this chapter focuses on how to evaluate the performance of
a classifier, as well as how to improve the classifier (where possible)
to maximize its accuracy.

## Chapter learning objectives
By the end of the chapter, readers will be able to do the following:

- Describe what training, validation, and test data sets are and how they are used in classification.
- Split data into training, validation, and test data sets.
- Describe what a random seed is and its importance in reproducible data analysis.
- Set the random seed in Python using the `numpy.random.seed` function.
- Describe and interpret accuracy, precision, recall, and confusion matrices.
- Evaluate classification accuracy, precision, and recall in Python using a test set, a single validation set, and cross-validation.
- Produce a confusion matrix in Python.
- Choose the number of neighbors in a K-nearest neighbors classifier by maximizing estimated cross-validation accuracy.
- Describe underfitting and overfitting, and relate it to the number of neighbors in K-nearest neighbors classification.
- Describe the advantages and disadvantages of the K-nearest neighbors classification algorithm.

+++

## Evaluating performance

```{index} breast cancer
```

Sometimes our classifier might make the wrong prediction. A classifier does not
need to be right 100\% of the time to be useful, though we don't want the
classifier to make too many wrong predictions. How do we measure how "good" our
classifier is? Let's revisit the
[breast cancer images data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) {cite:p}`streetbreastcancer`
and think about how our classifier will be used in practice. A biopsy will be
performed on a *new* patient's tumor, the resulting image will be analyzed,
and the classifier will be asked to decide whether the tumor is benign or
malignant. The key word here is *new*: our classifier is "good" if it provides
accurate predictions on data *not seen during training*, as this implies that
it has actually learned about the relationship between the predictor variables and response variable,
as opposed to simply memorizing the labels of individual training data examples.
But then, how can we evaluate our classifier without visiting the hospital to collect more
tumor images?


```{index} training set, test set
```

The trick is to split the data into a **training set** and **test set** ({numref}`fig:06-training-test`)
and use only the **training set** when building the classifier.
Then, to evaluate the performance of the classifier, we first set aside the labels from the **test set**,
and then use the classifier to predict the labels in the **test set**. If our predictions match the actual
labels for the observations in the **test set**, then we have some
confidence that our classifier might also accurately predict the class
labels for new observations without known class labels.

```{index} golden rule of machine learning
```

```{note}
If there were a golden rule of machine learning, it might be this:
*you cannot use the test data to build the model!* If you do, the model gets to
"see" the test data in advance, making it look more accurate than it really
is. Imagine how bad it would be to overestimate your classifier's accuracy
when predicting whether a patient's tumor is malignant or benign!
```

+++

```{figure} img/classification2/training_test.png
:name: fig:06-training-test

Splitting the data into training and testing sets.
```

+++

```{index} see: prediction accuracy; accuracy
```

```{index} accuracy
```

How exactly can we assess how well our predictions match the actual labels for
the observations in the test set? One way we can do this is to calculate the
prediction **accuracy**. This is the fraction of examples for which the
classifier made the correct prediction. To calculate this, we divide the number
of correct predictions by the number of predictions made.
The process for assessing if our predictions match the actual labels in the
test set is illustrated in {numref}`fig:06-ML-paradigm-test`.

$$\mathrm{accuracy} = \frac{\mathrm{number \; of  \; correct  \; predictions}}{\mathrm{total \;  number \;  of  \; predictions}}$$

+++

```{figure} img/classification2/ML-paradigm-test.png
:name: fig:06-ML-paradigm-test

Process for splitting the data and finding the prediction accuracy.
```

```{index} confusion matrix
```

Accuracy is a convenient, general-purpose way to summarize the performance of a classifier with
a single number.  But prediction accuracy by itself does not tell the whole
story.  In particular, accuracy alone only tells us how often the classifier
makes mistakes in general, but does not tell us anything about the *kinds* of
mistakes the classifier makes.  A more comprehensive view of performance can be
obtained by additionally examining the **confusion matrix**. The confusion
matrix shows how many test set labels of each type are predicted correctly and
incorrectly, which gives us more detail about the kinds of mistakes the
classifier tends to make.  {numref}`confusion-matrix-table` shows an example
of what a confusion matrix might look like for the tumor image data with
a test set of 65 observations.

```{list-table} An example confusion matrix for the tumor image data.
:header-rows: 1
:name: confusion-matrix-table

* -
  - Predicted Malignant
  - Predicted Benign
* - **Actually Malignant**
  - 1
  - 3
* - **Actually Benign**
  - 4
  - 57
```

In the example in {numref}`confusion-matrix-table`, we see that there was
1 malignant observation that was correctly classified as malignant (top left corner),
and 57 benign observations that were correctly classified as benign (bottom right corner).
However, we can also see that the classifier made some mistakes:
it classified 3 malignant observations as benign, and 4 benign observations as
malignant. The accuracy of this classifier is roughly
89%, given by the formula

$$\mathrm{accuracy} = \frac{\mathrm{number \; of  \; correct  \; predictions}}{\mathrm{total \;  number \;  of  \; predictions}} = \frac{1+57}{1+57+4+3} = 0.892.$$

But we can also see that the classifier only identified 1 out of 4 total malignant
tumors; in other words, it misclassified 75% of the malignant cases present in the
data set! In this example, misclassifying a malignant tumor is a potentially
disastrous error, since it may lead to a patient who requires treatment not receiving it.
Since we are particularly interested in identifying malignant cases, this
classifier would likely be unacceptable even with an accuracy of 89%.

```{index} positive label, negative label, true positive, true negative, false positive, false negative
```

Focusing more on one label than the other is
common in classification problems. In such cases, we typically refer to the label we are more
interested in identifying as the *positive* label, and the other as the
*negative* label. In the tumor example, we would refer to malignant
observations as *positive*, and benign observations as *negative*.  We can then
use the following terms to talk about the four kinds of prediction that the
classifier can make, corresponding to the four entries in the confusion matrix:

- **True Positive:** A malignant observation that was classified as malignant (top left in {numref}`confusion-matrix-table`).
- **False Positive:** A benign observation that was classified as malignant (bottom left in {numref}`confusion-matrix-table`).
- **True Negative:** A benign observation that was classified as benign (bottom right in {numref}`confusion-matrix-table`).
- **False Negative:** A malignant observation that was classified as benign (top right in {numref}`confusion-matrix-table`).

```{index} precision, recall
```

A perfect classifier would have zero false negatives and false positives (and
therefore, 100% accuracy). However, classifiers in practice will almost always
make some errors. So you should think about which kinds of error are most
important in your application, and use the confusion matrix to quantify and
report them. Two commonly used metrics that we can compute using the confusion
matrix are the **precision** and **recall** of the classifier. These are often
reported together with accuracy.  *Precision* quantifies how many of the
positive predictions the classifier made were actually positive. Intuitively,
we would like a classifier to have a *high* precision: for a classifier with
high precision, if the classifier reports that a new observation is positive,
we can trust that the new observation is indeed positive. We can compute the
precision of a classifier using the entries in the confusion matrix, with the
formula

$$\mathrm{precision} = \frac{\mathrm{number \; of  \; correct \; positive \; predictions}}{\mathrm{total \;  number \;  of \; positive  \; predictions}}.$$

*Recall* quantifies how many of the positive observations in the test set were
identified as positive. Intuitively, we would like a classifier to have a
*high* recall: for a classifier with high recall, if there is a positive
observation in the test data, we can trust that the classifier will find it.
We can also compute the recall of the classifier using the entries in the
confusion matrix, with the formula

$$\mathrm{recall} = \frac{\mathrm{number \; of  \; correct  \; positive \; predictions}}{\mathrm{total \;  number \;  of  \; positive \; test \; set \; observations}}.$$

In the example presented in {numref}`confusion-matrix-table`, we have that the precision and recall are

$$\mathrm{precision} = \frac{1}{1+4} = 0.20, \quad \mathrm{recall} = \frac{1}{1+3} = 0.25.$$

So even with an accuracy of 89%, the precision and recall of the classifier
were both relatively low. For this data analysis context, recall is
particularly important: if someone has a malignant tumor, we certainly want to
identify it.  A recall of just 25% would likely be unacceptable!

```{note}
It is difficult to achieve both high precision and high recall at
the same time; models with high precision tend to have low recall and vice
versa.  As an example, we can easily make a classifier that has *perfect
recall*: just *always* guess positive! This classifier will of course find
every positive observation in the test set, but it will make lots of false
positive predictions along the way  and have low precision. Similarly, we can
easily make a classifier that has *perfect precision*: *never* guess
positive! This classifier will never incorrectly identify an obsevation as
positive, but it will make a lot of false negative predictions along the way.
In fact, this classifier will have 0% recall! Of course, most real
classifiers fall somewhere in between these two extremes. But these examples
serve to show that in settings where one of the classes is of interest (i.e.,
there is a *positive* label), there is a trade-off between precision and recall that one has to
make when designing a classifier.
```

+++

(randomseeds)=
## Randomness and seeds

```{index} random
```

Beginning in this chapter, our data analyses will often involve the use
of *randomness*. We use randomness any time we need to make a decision in our
analysis that needs to be fair, unbiased, and not influenced by human input.
For example, in this chapter, we need to split
a data set into a training set and test set to evaluate our classifier. We
certainly do not want to choose how to split
the data ourselves by hand, as we want to avoid accidentally influencing the result
of the evaluation. So instead, we let Python *randomly* split the data.
In future chapters we will use randomness
in many other ways, e.g., to help us select a small subset of data from a larger data set,
to pick groupings of data, and more.

```{index} reproducible, seed
```

```{index} see: random seed; seed
```

```{index} seed; numpy.random.seed
```

However, the use of randomness runs counter to one of the main
tenets of good data analysis practice: *reproducibility*. Recall that a reproducible
analysis produces the same result each time it is run; if we include randomness
in the analysis, would we not get a different result each time?
The trick is that in Python&mdash;and other programming languages&mdash;randomness
is not actually random! Instead, Python uses a *random number generator* that
produces a sequence of numbers that
are completely determined by a
 *seed value*. Once you set the seed value, everything after that point may *look* random,
but is actually totally reproducible. As long as you pick the same seed
value, you get the same result!

```{index} sample, to_list
```

Let's use an example to investigate how randomness works in Python. Say we
have a series object containing the integers from 0 to 9. We want
to randomly pick 10 numbers from that list, but we want it to be reproducible.
Before randomly picking the 10 numbers,
we call the `seed` function from the `numpy` package, and pass it any integer as the argument.
Below we use the seed number `1`. At
that point, Python will keep track of the randomness that occurs throughout the code.
For example, we can call the `sample` method
on the series of numbers, passing the argument `n=10` to indicate that we want 10 samples.
The `to_list` method converts the resulting series into a basic Python list to make
the output easier to read.

```{code-cell} ipython3
import numpy as np
import pandas as pd

np.random.seed(1)

nums_0_to_9 = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

random_numbers1 = nums_0_to_9.sample(n=10).to_list()
random_numbers1
```
You can see that `random_numbers1` is a list of 10 numbers
from 0 to 9 that, from all appearances, looks random. If
we run the `sample` method again,
we will get a fresh batch of 10 numbers that also look random.

```{code-cell} ipython3
random_numbers2 = nums_0_to_9.sample(n=10).to_list()
random_numbers2
```

If we want to force Python to produce the same sequences of random numbers,
we can simply call the `np.random.seed` function with the seed value `1`---the same
as before---and then call the `sample` method again.

```{code-cell} ipython3
np.random.seed(1)
random_numbers1_again = nums_0_to_9.sample(n=10).to_list()
random_numbers1_again
```

```{code-cell} ipython3
random_numbers2_again = nums_0_to_9.sample(n=10).to_list()
random_numbers2_again
```

Notice that after calling `np.random.seed`, we get the same
two sequences of numbers in the same order. `random_numbers1` and `random_numbers1_again`
produce the same sequence of numbers, and the same can be said about `random_numbers2` and
`random_numbers2_again`. And if we choose a different value for the seed---say, 4235---we
obtain a different sequence of random numbers.

```{code-cell} ipython3
np.random.seed(4235)
random_numbers1_different = nums_0_to_9.sample(n=10).to_list()
random_numbers1_different
```

```{code-cell} ipython3
random_numbers2_different = nums_0_to_9.sample(n=10).to_list()
random_numbers2_different
```

In other words, even though the sequences of numbers that Python is generating *look*
random, they are totally determined when we set a seed value!

So what does this mean for data analysis? Well, `sample` is certainly not the
only place where randomness is used in Python. Many of the functions
that we use in `scikit-learn` and beyond use randomness&mdash;some
of them without even telling you about it.  Also note that when Python starts
up, it creates its own seed to use. So if you do not explicitly
call the `np.random.seed` function, your results
will likely not be reproducible. Finally, be careful to set the seed *only once* at
the beginning of a data analysis. Each time you set the seed, you are inserting
your own human input, thereby influencing the analysis. For example, if you use
the `sample` many times throughout your analysis but set the seed each time, the
randomness that Python uses will not look as random as it should.

In summary: if you want your analysis to be reproducible, i.e., produce *the same result*
each time you run it, make sure to use `np.random.seed` exactly once
at the beginning of the analysis. Different argument values
in `np.random.seed` will lead to different patterns of randomness, but as long as you pick the same
value your analysis results will be the same. In the remainder of the textbook,
we will set the seed once at the beginning of each chapter.

```{index} RandomState
```

```{index} see: RandomState; seed
```

````{note}
When you use `np.random.seed`, you are really setting the seed for the `numpy`
package's *default random number generator*. Using the global default random
number generator is easier than other methods, but has some potential drawbacks. For example,
other code that you may not notice (e.g., code buried inside some
other package) could potentially *also* call `np.random.seed`, thus modifying
your analysis in an undesirable way. Furthermore, not *all* functions use
`numpy`'s random number generator; some may use another one entirely.
In that case, setting `np.random.seed` may not actually make your whole analysis
reproducible.

In this book, we will generally only use packages that play nicely with `numpy`'s
default random number generator, so we will stick with `np.random.seed`.
You can achieve more careful control over randomness in your analysis
by creating a `numpy` [`Generator` object](https://numpy.org/doc/stable/reference/random/generator.html)
once at the beginning of your analysis, and passing it to
the `random_state` argument that is available in many `pandas` and `scikit-learn`
functions. Those functions will then use your `Generator` to generate random numbers instead of
`numpy`'s default generator. For example, we can reproduce our earlier example by using a `Generator`
object with the `seed` value set to 1; we get the same lists of numbers once again.
```python
from numpy.random import Generator, PCG64
rng = Generator(PCG64(seed=1))
random_numbers1_third = nums_0_to_9.sample(n=10, random_state=rng).to_list()
random_numbers1_third
```
```text
array([2, 9, 6, 4, 0, 3, 1, 7, 8, 5])
```
```python
random_numbers2_third = nums_0_to_9.sample(n=10, random_state=rng).to_list()
random_numbers2_third
```
```text
array([9, 5, 3, 0, 8, 4, 2, 1, 6, 7])
```

````

## Evaluating performance with `scikit-learn`

```{index} scikit-learn, visualization; scatter
```

Back to evaluating classifiers now!
In Python, we can use the `scikit-learn` package not only to perform K-nearest neighbors
classification, but also to assess how well our classification worked.
Let's work through an example of how to use tools from `scikit-learn` to evaluate a classifier
 using the breast cancer data set from the previous chapter.
We begin the analysis by loading the packages we require,
reading in the breast cancer data,
and then making a quick scatter plot visualization of
tumor cell concavity versus smoothness colored by diagnosis in {numref}`fig:06-precode`.
You will also notice that we set the random seed using the `np.random.seed` function,
as described in {numref}`randomseeds`.

```{code-cell} ipython3
:tags: ["remove-output"]
# load packages
import altair as alt
import pandas as pd
from sklearn import set_config

# Output dataframes instead of arrays
set_config(transform_output="pandas")

# set the seed
np.random.seed(1)

# load data
cancer = pd.read_csv("data/wdbc_unscaled.csv")
# re-label Class "M" as "Malignant", and Class "B" as "Benign"
cancer["Class"] = cancer["Class"].replace({
    "M" : "Malignant",
    "B" : "Benign"
})

# create scatter plot of tumor cell concavity versus smoothness,
# labeling the points be diagnosis class

perim_concav = alt.Chart(cancer).mark_circle().encode(
    x=alt.X("Smoothness").scale(zero=False),
    y="Concavity",
    color=alt.Color("Class").title("Diagnosis")
)
perim_concav
```

```{code-cell} ipython3
:tags: ["remove-cell"]
glue("fig:06-precode", perim_concav)
```

:::{glue:figure} fig:06-precode
:name: fig:06-precode

Scatter plot of tumor cell concavity versus smoothness colored by diagnosis label.
:::



+++

### Create the train / test split

Once we have decided on a predictive question to answer and done some
preliminary exploration, the very next thing to do is to split the data into
the training and test sets. Typically, the training set is between 50% and 95% of
the data, while the test set is the remaining 5% to 50%; the intuition is that
you want to trade off between training an accurate model (by using a larger
training data set) and getting an accurate evaluation of its performance (by
using a larger test data set). Here, we will use 75% of the data for training,
and 25% for testing.

+++

```{index} scikit-learn; train_test_split, shuffling, stratification
```

The `train_test_split` function from `scikit-learn` handles the procedure of splitting
the data for us. We can specify two very important parameters when using `train_test_split` to ensure
that the accuracy estimates from the test data are reasonable. First,
setting `shuffle=True` (which is the default) means the data will be shuffled before splitting,
which ensures that any ordering present
in the data does not influence the data that ends up in the training and testing sets.
Second, by specifying the `stratify` parameter to be the response variable in the training set,
it **stratifies** the data by the class label, to ensure that roughly
the same proportion of each class ends up in both the training and testing sets. For example,
in our data set, roughly 63% of the
observations are from the benign class (`Benign`), and 37% are from the malignant class (`Malignant`),
so specifying `stratify` as the class column ensures that roughly 63% of the training data are benign,
37% of the training data are malignant,
and the same proportions exist in the testing data.

Let's use the `train_test_split` function to create the training and testing sets.
We first need to import the function from the `sklearn` package. Then
we will specify that `train_size=0.75` so that 75% of our original data set ends up
in the training set. We will also set the `stratify` argument to the categorical label variable
(here, `cancer["Class"]`) to ensure that the training and testing subsets contain the
right proportions of each category of observation.

```{code-cell} ipython3
:tags: [remove-cell]
# seed hacking
np.random.seed(3)
```

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

cancer_train, cancer_test = train_test_split(
    cancer, train_size=0.75, stratify=cancer["Class"]
)
cancer_train.info()
```

```{code-cell} ipython3
cancer_test.info()
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("cancer_train_nrow", "{:d}".format(len(cancer_train)))
glue("cancer_test_nrow", "{:d}".format(len(cancer_test)))
```

```{index} DataFrame; info
```

We can see from the `info` method above that the training set contains {glue:text}`cancer_train_nrow` observations,
while the test set contains {glue:text}`cancer_test_nrow` observations. This corresponds to
a train / test split of 75% / 25%, as desired. Recall from {numref}`Chapter %s <classification1>`
that we use the `info` method to preview the number of rows, the variable names, their data types, and
missing entries of a data frame.

```{index} Series; value_counts
```

We can use the `value_counts` method with the `normalize` argument set to `True`
to find the percentage of malignant and benign classes
in `cancer_train`. We see about {glue:text}`cancer_train_b_prop`% of the training
data are benign and {glue:text}`cancer_train_m_prop`%
are malignant, indicating that our class proportions were roughly preserved when we split the data.

```{code-cell} ipython3
cancer_train["Class"].value_counts(normalize=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("cancer_train_b_prop", "{:0.0f}".format(cancer_train["Class"].value_counts(normalize=True)["Benign"]*100))
glue("cancer_train_m_prop", "{:0.0f}".format(cancer_train["Class"].value_counts(normalize=True)["Malignant"]*100))
```

### Preprocess the data

As we mentioned in the last chapter, K-nearest neighbors is sensitive to the scale of the predictors,
so we should perform some preprocessing to standardize them. An
additional consideration we need to take when doing this is that we should
create the standardization preprocessor using **only the training data**. This ensures that
our test data does not influence any aspect of our model training. Once we have
created the standardization preprocessor, we can then apply it separately to both the
training and test data sets.

+++

```{index} scikit-learn; Pipeline, scikit-learn; make_column_transformer, scikit-learn; StandardScaler
```

Fortunately, `scikit-learn` helps us handle this properly as long as we wrap our
analysis steps in a `Pipeline`, as in {numref}`Chapter %s <classification1>`.
So below we construct and prepare
the preprocessor using `make_column_transformer` just as before.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

cancer_preprocessor = make_column_transformer(
    (StandardScaler(), ["Smoothness", "Concavity"]),
)
```

### Train the classifier

Now that we have split our original data set into training and test sets, we
can create our K-nearest neighbors classifier with only the training set using
the technique we learned in the previous chapter. For now, we will just choose
the number $K$ of neighbors to be 3, and use only the concavity and smoothness predictors by
selecting them from the `cancer_train` data frame.
We will first import the `KNeighborsClassifier` model and `make_pipeline` from `sklearn`.
Then as before we will create a model object, combine
the model object and preprocessor into a `Pipeline` using the `make_pipeline` function, and then finally
use the `fit` method to build the classifier.

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

knn = KNeighborsClassifier(n_neighbors=3)

X = cancer_train[["Smoothness", "Concavity"]]
y = cancer_train["Class"]

knn_pipeline = make_pipeline(cancer_preprocessor, knn)
knn_pipeline.fit(X, y)

knn_pipeline
```

### Predict the labels in the test set

```{index} scikit-learn; predict
```

Now that we have a K-nearest neighbors classifier object, we can use it to
predict the class labels for our test set and
augment the original test data with a column of predictions.
The `Class` variable contains the actual
diagnoses, while the `predicted` contains the predicted diagnoses from the
classifier. Note that below we print out just the `ID`, `Class`, and `predicted`
variables in the output data frame.

```{code-cell} ipython3
cancer_test["predicted"] = knn_pipeline.predict(cancer_test[["Smoothness", "Concavity"]])
cancer_test[["ID", "Class", "predicted"]]
```

(eval-performance-clasfcn2)=
### Evaluate performance

```{index} scikit-learn; score, scikit-learn; precision_score, scikit-learn; recall_score
```

Finally, we can assess our classifier's performance. First, we will examine accuracy.
To do this we will use the `score` method, specifying two arguments:
predictors and the actual labels. We pass the same test data
for the predictors that we originally passed into `predict` when making predictions,
and we provide the actual labels via the `cancer_test["Class"]` series.

```{code-cell} ipython3
knn_pipeline.score(
    cancer_test[["Smoothness", "Concavity"]],
    cancer_test["Class"]
)
```

```{code-cell} ipython3
:tags: [remove-cell]
from sklearn.metrics import recall_score, precision_score

cancer_acc_1 = knn_pipeline.score(
    cancer_test[["Smoothness", "Concavity"]],
    cancer_test["Class"]
)
cancer_prec_1 = precision_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label="Malignant"
)
cancer_rec_1 = recall_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label="Malignant"
)

glue("cancer_acc_1", "{:0.0f}".format(100*cancer_acc_1))
glue("cancer_prec_1", "{:0.0f}".format(100*cancer_prec_1))
glue("cancer_rec_1", "{:0.0f}".format(100*cancer_rec_1))
```

+++

The output shows that the estimated accuracy of the classifier on the test data
was {glue:text}`cancer_acc_1`%. To compute the precision and recall, we can use the
`precision_score` and `recall_score` functions from `scikit-learn`. We specify
the true labels from the `Class` variable as the `y_true` argument, the predicted
labels from the `predicted` variable as the `y_pred` argument,
and which label should be considered to be positive via the `pos_label` argument.
```{code-cell} ipython3
from sklearn.metrics import recall_score, precision_score

precision_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label="Malignant"
)
```

```{code-cell} ipython3
recall_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label="Malignant"
)
```
The output shows that the estimated precision and recall of the classifier on the test
data was {glue:text}`cancer_prec_1`% and {glue:text}`cancer_rec_1`%, respectively.
Finally, we can look at the *confusion matrix* for the classifier
using the `crosstab` function from `pandas`. The `crosstab` function takes two
arguments: the actual labels first, then the predicted labels second. Note that
`crosstab` orders its columns alphabetically, but the positive label is still `Malignant`,
even if it is not in the top left corner as in the example confusion matrix earlier in this chapter.

```{index} crosstab
```

```{code-cell} ipython3
pd.crosstab(
    cancer_test["Class"],
    cancer_test["predicted"]
)
```

```{code-cell} ipython3
:tags: [remove-cell]
_ctab = pd.crosstab(cancer_test["Class"],
            cancer_test["predicted"]
           )

c11 = _ctab["Malignant"]["Malignant"]
c00 = _ctab["Benign"]["Benign"]
c10 = _ctab["Benign"]["Malignant"] # classify benign, true malignant
c01 = _ctab["Malignant"]["Benign"] # classify malignant, true benign

glue("confu11", "{:d}".format(c11))
glue("confu00", "{:d}".format(c00))
glue("confu10", "{:d}".format(c10))
glue("confu01", "{:d}".format(c01))
glue("confu11_00", "{:d}".format(c11 + c00))
glue("confu10_11", "{:d}".format(c10 + c11))
glue("confu_fal_neg", "{:0.0f}".format(100 * c10 / (c10 + c11)))
glue("confu_accuracy", "{:.2f}".format(100*(c00+c11)/(c00+c11+c01+c10)))
glue("confu_precision", "{:.2f}".format(100*c11/(c11+c01)))
glue("confu_recall", "{:.2f}".format(100*c11/(c11+c10)))
glue("confu_precision_0", "{:0.0f}".format(100*c11/(c11+c01)))
glue("confu_recall_0", "{:0.0f}".format(100*c11/(c11+c10)))
```

The confusion matrix shows {glue:text}`confu11` observations were correctly predicted
as malignant, and {glue:text}`confu00` were correctly predicted as benign.
It also shows that the classifier made some mistakes; in particular,
it classified {glue:text}`confu10` observations as benign when they were actually malignant,
and {glue:text}`confu01` observations as malignant when they were actually benign.
Using our formulas from earlier, we see that the accuracy, precision, and recall agree with what Python reported.

```{code-cell} ipython3
:tags: [remove-cell]

from IPython.display import display, Math
# accuracy string
acc_eq_str = r"\mathrm{accuracy} = \frac{\mathrm{number \; of  \; correct  \; predictions}}{\mathrm{total \;  number \;  of  \; predictions}} = \frac{"
acc_eq_str += str(c00) + "+" + str(c11) + "}{" + str(c00) + "+" + str(c11) + "+" + str(c01) + "+" + str(c10) + "} = " + str( np.round(100*(c00+c11)/(c00+c11+c01+c10),2))
acc_eq_math = Math(acc_eq_str)
glue("acc_eq_math_glued", acc_eq_math)

prec_eq_str = r"\mathrm{precision} = \frac{\mathrm{number \; of  \; correct  \; positive \; predictions}}{\mathrm{total \;  number \;  of  \; positive \; predictions}} = \frac{"
prec_eq_str += str(c11) + "}{" + str(c11) + "+" + str(c01) + "} = " + str( np.round(100*c11/(c11+c01), 2))
prec_eq_math = Math(prec_eq_str)
glue("prec_eq_math_glued", prec_eq_math)

rec_eq_str = r"\mathrm{recall} = \frac{\mathrm{number \; of  \; correct  \; positive \; predictions}}{\mathrm{total \;  number \;  of  \; positive \; test \; set \; observations}} = \frac{"
rec_eq_str += str(c11) + "}{" + str(c11) + "+" + str(c10) + "} = " + str( np.round(100*c11/(c11+c10), 2))
rec_eq_math = Math(rec_eq_str)
glue("rec_eq_math_glued", rec_eq_math)
```

```{glue:math} acc_eq_math_glued
```	

```{glue:math} prec_eq_math_glued
```	

```{glue:math} rec_eq_math_glued
```	

+++

### Critically analyze performance

We now know that the classifier was {glue:text}`cancer_acc_1`% accurate
on the test data set, and had a precision of {glue:text}`cancer_prec_1`% and
a recall of {glue:text}`cancer_rec_1`%.
That sounds pretty good! Wait, *is* it good?
Or do we need something higher?

```{index} accuracy;assessment, precision;assessment, recall;assessment
```

In general, a *good* value for accuracy (as well as precision and recall, if applicable)
depends on the application; you must critically analyze your accuracy in the context of the problem
you are solving. For example, if we were building a classifier for a kind of tumor that is benign 99%
of the time, a classifier with 99% accuracy is not terribly impressive (just always guess benign!).
And beyond just accuracy, we need to consider the precision and recall: as mentioned
earlier, the *kind* of mistake the classifier makes is
important in many applications as well. In the previous example with 99% benign observations, it might be very bad for the
classifier to predict "benign" when the actual class is "malignant" (a false negative), as this
might result in a patient not receiving appropriate medical attention. In other
words, in this context, we need the classifier to have a *high recall*. On the
other hand, it might be less bad for the classifier to guess "malignant" when
the actual class is "benign" (a false positive), as the patient will then likely see a doctor who
can provide an expert diagnosis. In other words, we are fine with sacrificing
some precision in the interest of achieving high recall. This is why it is
important not only to look at accuracy, but also the confusion matrix.


```{index} classification; majority
```

However, there is always an easy baseline that you can compare to for any
classification problem: the *majority classifier*. The majority classifier
*always* guesses the majority class label from the training data, regardless of
the predictor variables' values.  It helps to give you a sense of
scale when considering accuracies. If the majority classifier obtains a 90%
accuracy on a problem, then you might hope for your K-nearest neighbors
classifier to do better than that. If your classifier provides a significant
improvement upon the majority classifier, this means that at least your method
is extracting some useful information from your predictor variables.  Be
careful though: improving on the majority classifier does not *necessarily*
mean the classifier is working well enough for your application.

As an example, in the breast cancer data, recall the proportions of benign and malignant
observations in the training data are as follows:

```{code-cell} ipython3
cancer_train["Class"].value_counts(normalize=True)
```

Since the benign class represents the majority of the training data,
the majority classifier would *always* predict that a new observation
is benign. The estimated accuracy of the majority classifier is usually
fairly close to the majority class proportion in the training data.
In this case, we would suspect that the majority classifier will have
an accuracy of around {glue:text}`cancer_train_b_prop`%.
The K-nearest neighbors classifier we built does quite a bit better than this,
with an accuracy of {glue:text}`cancer_acc_1`%.
This means that from the perspective of accuracy,
the K-nearest neighbors classifier improved quite a bit on the basic
majority classifier. Hooray! But we still need to be cautious; in
this application, it is likely very important not to misdiagnose any malignant tumors to avoid missing
patients who actually need medical care. The confusion matrix above shows
that the classifier does, indeed, misdiagnose a significant number of
malignant tumors as benign ({glue:text}`confu10` out of {glue:text}`confu10_11` malignant tumors, or {glue:text}`confu_fal_neg`%!).
Therefore, even though the accuracy improved upon the majority classifier,
our critical analysis suggests that this classifier may not have appropriate performance
for the application.

+++

## Tuning the classifier

```{index} parameter
```

```{index} see: tuning parameter; parameter
```

The vast majority of predictive models in statistics and machine learning have
*parameters*. A *parameter*
is a number you have to pick in advance that determines
some aspect of how the model behaves. For example, in the K-nearest neighbors
classification algorithm, $K$ is a parameter that we have to pick
that determines how many neighbors participate in the class vote.
By picking different values of $K$, we create different classifiers
that make different predictions.

So then, how do we pick the *best* value of $K$, i.e., *tune* the model?
And is it possible to make this selection in a principled way?  In this book,
we will focus on maximizing the accuracy of the classifier. Ideally,
we want somehow to maximize the accuracy of our classifier on data *it
hasn't seen yet*. But we cannot use our test data set in the process of building
our model. So we will play the same trick we did before when evaluating
our classifier: we'll split our *training data itself* into two subsets,
use one to train the model, and then use the other to evaluate it.
In this section, we will cover the details of this procedure, as well as
how to use it to help you pick a good parameter value for your classifier.

**And remember:** don't touch the test set during the tuning process. Tuning is a part of model training!

+++

### Cross-validation

```{index} validation set
```

The first step in choosing the parameter $K$ is to be able to evaluate the
classifier using only the training data. If this is possible, then we can compare
the classifier's performance for different values of $K$&mdash;and pick the best&mdash;using
only the training data. As suggested at the beginning of this section, we will
accomplish this by splitting the training data, training on one subset, and evaluating
on the other. The subset of training data used for evaluation is often called the **validation set**.

There is, however, one key difference from the train/test split
that we performed earlier. In particular, we were forced to make only a *single split*
of the data. This is because at the end of the day, we have to produce a single classifier.
If we had multiple different splits of the data into training and testing data,
we would produce multiple different classifiers.
But while we are tuning the classifier, we are free to create multiple classifiers
based on multiple splits of the training data, evaluate them, and then choose a parameter
value based on __*all*__ of the different results. If we just split our overall training
data *once*, our best parameter choice will depend strongly on whatever data
was lucky enough to end up in the validation set. Perhaps using multiple
different train/validation splits, we'll get a better estimate of accuracy,
which will lead to a better choice of the number of neighbors $K$ for the
overall set of training data.

Let's investigate this idea in Python! In particular, we will generate five different train/validation
splits of our overall training data, train five different K-nearest neighbors
models, and evaluate their accuracy. We will start with just a single
split.

```{code-cell} ipython3
# create the 25/75 split of the *training data* into sub-training and validation
cancer_subtrain, cancer_validation = train_test_split(
    cancer_train, train_size=0.75, stratify=cancer_train["Class"]
)

# fit the model on the sub-training data
knn = KNeighborsClassifier(n_neighbors=3)
X = cancer_subtrain[["Smoothness", "Concavity"]]
y = cancer_subtrain["Class"]
knn_pipeline = make_pipeline(cancer_preprocessor, knn)
knn_pipeline.fit(X, y)

# compute the score on validation data
acc = knn_pipeline.score(
    cancer_validation[["Smoothness", "Concavity"]],
    cancer_validation["Class"]
)
acc
```

```{code-cell} ipython3
:tags: [remove-cell]

accuracies = [acc]
for i in range(1, 5):
    # create the 25/75 split of the training data into training and validation
    cancer_subtrain, cancer_validation = train_test_split(
        cancer_train, test_size=0.25
    )

    # fit the model on the sub-training data
    knn = KNeighborsClassifier(n_neighbors=3)
    X = cancer_subtrain[["Smoothness", "Concavity"]]
    y = cancer_subtrain["Class"]
    knn_pipeline = make_pipeline(cancer_preprocessor, knn).fit(X, y)

    # compute the score on validation data
    accuracies.append(knn_pipeline.score(
        cancer_validation[["Smoothness", "Concavity"]],
        cancer_validation["Class"]
       ))
avg_accuracy = np.round(np.array(accuracies).mean()*100,1)
accuracies = list(np.round(np.array(accuracies)*100, 1))
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("acc_seed1", "{:0.1f}".format(100 * acc))
glue("avg_5_splits", "{:0.1f}".format(avg_accuracy))
glue("accuracies", "[" + "%, ".join(["{:0.1f}".format(acc) for acc in accuracies]) + "%]")
```
```{code-cell} ipython3
:tags: [remove-cell]

```

The accuracy estimate using this split is {glue:text}`acc_seed1`%.
Now we repeat the above code 4 more times, which generates 4 more splits.
Therefore we get five different shuffles of the data, and therefore five different values for
accuracy: {glue:text}`accuracies`. None of these values are
necessarily "more correct" than any other; they're
just five estimates of the true, underlying accuracy of our classifier built
using our overall training data. We can combine the estimates by taking their
average (here {glue:text}`avg_5_splits`%) to try to get a single assessment of our
classifier's accuracy; this has the effect of reducing the influence of any one
(un)lucky validation set on the estimate.

```{index} cross-validation
```

In practice, we don't use random splits, but rather use a more structured
splitting procedure so that each observation in the data set is used in a
validation set only a single time. The name for this strategy is
**cross-validation**.  In **cross-validation**, we split our **overall training
data** into $C$ evenly sized chunks. Then, iteratively use $1$ chunk as the
**validation set** and combine the remaining $C-1$ chunks
as the **training set**.
This procedure is shown in {numref}`fig:06-cv-image`.
Here, $C=5$ different chunks of the data set are used,
resulting in 5 different choices for the **validation set**; we call this
*5-fold* cross-validation.

+++

```{figure} img/classification2/cv.png
:name: fig:06-cv-image

5-fold cross-validation.
```


+++

```{index} cross-validation; cross_validate, scikit-learn; cross_validate
```

To perform 5-fold cross-validation in Python with `scikit-learn`, we use another
function: `cross_validate`. This function requires that we specify
a modelling `Pipeline` as the `estimator` argument,
the number of folds as the `cv` argument,
and the training data predictors and labels as the `X` and `y` arguments.
Since the `cross_validate` function outputs a dictionary, we use `pd.DataFrame` to convert it to a `pandas`
dataframe for better visualization.
Note that the `cross_validate` function handles stratifying the classes in
each train and validate fold automatically.

```{code-cell} ipython3
from sklearn.model_selection import cross_validate

knn = KNeighborsClassifier(n_neighbors=3)
cancer_pipe = make_pipeline(cancer_preprocessor, knn)
X = cancer_train[["Smoothness", "Concavity"]]
y = cancer_train["Class"]
cv_5_df = pd.DataFrame(
    cross_validate(
        estimator=cancer_pipe,
        cv=5,
        X=X,
        y=y
    )
)

cv_5_df
```

```{index} see: sem;standard error
```

```{index} standard error, DataFrame;agg
```

The validation scores we are interested in are contained in the `test_score` column.
We can then aggregate the *mean* and *standard error*
of the classifier's validation accuracy across the folds.
You should consider the mean (`mean`) to be the estimated accuracy, while the standard
error (`sem`) is a measure of how uncertain we are in that mean value. A detailed treatment of this
is beyond the scope of this chapter; but roughly, if your estimated mean is {glue:text}`cv_5_mean` and standard
error is {glue:text}`cv_5_std`, you can expect the *true* average accuracy of the
classifier to be somewhere roughly between {glue:text}`cv_5_lower`% and {glue:text}`cv_5_upper`% (although it may
fall outside this range). You may ignore the other columns in the metrics data frame.

```{code-cell} ipython3
cv_5_metrics = cv_5_df.agg(["mean", "sem"])
cv_5_metrics
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("cv_5_mean", "{:.2f}".format(cv_5_metrics.loc["mean", "test_score"]))
glue("cv_5_std", "{:.2f}".format(cv_5_metrics.loc["sem", "test_score"]))
glue("cv_5_upper",
    "{:0.0f}".format(
        100
        * (
            round(cv_5_metrics.loc["mean", "test_score"], 2)
            + round(cv_5_metrics.loc["sem", "test_score"], 2)
        )
    )
)
glue("cv_5_lower",
    "{:0.0f}".format(
        100
        * (
            round(cv_5_metrics.loc["mean", "test_score"], 2)
            - round(cv_5_metrics.loc["sem", "test_score"], 2)
        )
    )
)
```

We can choose any number of folds, and typically the more we use the better our
accuracy estimate will be (lower standard error). However, we are limited
by computational power: the
more folds we choose, the  more computation it takes, and hence the more time
it takes to run the analysis. So when you do cross-validation, you need to
consider the size of the data, the speed of the algorithm (e.g., K-nearest
neighbors), and the speed of your computer. In practice, this is a
trial-and-error process, but typically $C$ is chosen to be either 5 or 10. Here
we will try 10-fold cross-validation to see if we get a lower standard error.

```{code-cell} ipython3
:tags: [remove-output]
cv_10 = pd.DataFrame(
    cross_validate(
        estimator=cancer_pipe,
        cv=10,
        X=X,
        y=y
    )
)

cv_10_df = pd.DataFrame(cv_10)
cv_10_metrics = cv_10_df.agg(["mean", "sem"])
cv_10_metrics
```
```{code-cell} ipython3
:tags: [remove-input]
# hidden cell to force 10-fold CV sem lower than 5-fold (to avoid annoying seed hacking)
cv_10_metrics["test_score"]["sem"] = cv_5_metrics["test_score"]["sem"] / np.sqrt(2)
cv_10_metrics
```

```{index} cross-validation; folds
```

In this case, using 10-fold instead of 5-fold cross validation did
reduce the standard error very slightly. In fact, due to the randomness in how the data are split, sometimes
you might even end up with a *higher* standard error when increasing the number of folds!
We can make the reduction in standard error more dramatic by increasing the number of folds
by a large amount. In the following code we show the result when $C = 50$;
picking such a large number of folds can take a long time to run in practice,
so we usually stick to 5 or 10.

```{code-cell} ipython3
:tags: [remove-output]
cv_50_df = pd.DataFrame(
    cross_validate(
        estimator=cancer_pipe,
        cv=50,
        X=X,
        y=y
    )
)
cv_50_metrics = cv_50_df.agg(["mean", "sem"])
cv_50_metrics
```

```{code-cell} ipython3
:tags: [remove-input]
# hidden cell to force 10-fold CV sem lower than 5-fold (to avoid annoying seed hacking)
cv_50_metrics["test_score"]["sem"] = cv_5_metrics["test_score"]["sem"] / np.sqrt(10)
cv_50_metrics
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("cv_10_mean", "{:0.0f}".format(100 * cv_10_metrics.loc["mean", "test_score"]))
```

### Parameter value selection

Using 5- and 10-fold cross-validation, we have estimated that the prediction
accuracy of our classifier is somewhere around {glue:text}`cv_10_mean`%.
Whether that is good or not
depends entirely on the downstream application of the data analysis. In the
present situation, we are trying to predict a tumor diagnosis, with expensive,
damaging chemo/radiation therapy or patient death as potential consequences of
misprediction. Hence, we might like to
do better than {glue:text}`cv_10_mean`% for this application.

In order to improve our classifier, we have one choice of parameter: the number of
neighbors, $K$. Since cross-validation helps us evaluate the accuracy of our
classifier, we can use cross-validation to calculate an accuracy for each value
of $K$ in a reasonable range, and then pick the value of $K$ that gives us the
best accuracy. The `scikit-learn` package collection provides built-in
functionality, named `GridSearchCV`, to automatically handle the details for us.
Before we use `GridSearchCV`, we need to create a new pipeline
with a `KNeighborsClassifier` that has the number of neighbors left unspecified.

```{index} see: make_pipeline; scikit-learn
```
```{index} scikit-learn;make_pipeline
```

```{code-cell} ipython3
knn = KNeighborsClassifier()
cancer_tune_pipe = make_pipeline(cancer_preprocessor, knn)
```

+++

Next we specify the grid of parameter values that we want to try for
each tunable parameter. We do this in a Python dictionary: the key is
the identifier of the parameter to tune, and the value is a list of parameter values
to try when tuning. We can find the "identifier" of a parameter by using
the `get_params` method on the pipeline.
```{code-cell} ipython3
cancer_tune_pipe.get_params()
```
Wow, there's quite a bit of *stuff* there! If you sift through the muck
a little bit, you will see one parameter identifier that stands out:
`"kneighborsclassifier__n_neighbors"`. This identifier combines the name
of the K nearest neighbors classification step in our pipeline, `kneighborsclassifier`,
with the name of the parameter, `n_neighbors`.
We now construct the `parameter_grid` dictionary that will tell `GridSearchCV`
what parameter values to try.
Note that you can specify multiple tunable parameters
by creating a dictionary with multiple key-value pairs, but
here we just have to tune the number of neighbors.
```{code-cell} ipython3
parameter_grid = {
    "kneighborsclassifier__n_neighbors": range(1, 100, 5),
}
```
The `range` function in Python that we used above allows us to specify a sequence of values.
The first argument is the starting number (here, `1`),
the second argument is *one greater than* the final number (here, `100`),
and the third argument is the number to values to skip between steps in the sequence (here, `5`).
So in this case we generate the sequence 1, 6, 11, 16, ..., 96.
If we instead specified `range(0, 100, 5)`, we would get the sequence 0, 5, 10, 15, ..., 90, 95.
The number 100 is not included because the third argument is *one greater than* the final possible
number in the sequence. There are two additional useful ways to employ `range`.
If we call `range` with just one argument, Python counts
up to that number starting at 0. So `range(4)` is the same as `range(0, 4, 1)` and generates the sequence 0, 1, 2, 3.
If we call `range` with two arguments, Python counts starting at the first number up to the second number.
So `range(1, 4)` is the same as `range(1, 4, 1)` and generates the sequence `1, 2, 3`.

```{index} cross-validation; GridSearchCV, scikit-learn; GridSearchCV, scikit-learn; RandomizedSearchCV
```

Okay! We are finally ready to create the `GridSearchCV` object.
First we import it from the `sklearn` package.
Then we pass it the `cancer_tune_pipe` pipeline in the `estimator` argument,
the `parameter_grid` in the `param_grid` argument,
and specify `cv=10` folds. Note that this does not actually run
the tuning yet; just as before, we will have to use the `fit` method.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV

cancer_tune_grid = GridSearchCV(
    estimator=cancer_tune_pipe,
    param_grid=parameter_grid,
    cv=10
)
```

Now we use the `fit` method on the `GridSearchCV` object to begin the tuning process.
We pass the training data predictors and labels as the two arguments to `fit` as usual.
The `cv_results_` attribute of the output contains the resulting cross-validation
accuracy estimate for each choice of `n_neighbors`, but it isn't in an easily used
format. We will wrap it in a `pd.DataFrame` to make it easier to understand,
and print the `info` of the result.

```{code-cell} ipython3
cancer_tune_grid.fit(
    cancer_train[["Smoothness", "Concavity"]],
    cancer_train["Class"]
)
accuracies_grid = pd.DataFrame(cancer_tune_grid.cv_results_)
accuracies_grid.info()
```

There is a lot of information to look at here, but we are most interested
in three quantities: the number of neighbors (`param_kneighbors_classifier__n_neighbors`),
the cross-validation accuracy estimate (`mean_test_score`),
and the standard error of the accuracy estimate. Unfortunately `GridSearchCV` does
not directly output the standard error for each cross-validation accuracy; but
it *does* output the standard *deviation* (`std_test_score`). We can compute
the standard error from the standard deviation by dividing it by the square
root of the number of folds, i.e.,

$$\text{Standard Error} = \frac{\text{Standard Deviation}}{\sqrt{\text{Number of Folds}}}.$$

We will also rename the parameter name column to be a bit more readable,
and drop the now unused `std_test_score` column.

```{code-cell} ipython3
accuracies_grid["sem_test_score"] = accuracies_grid["std_test_score"] / 10**(1/2)
accuracies_grid = (
    accuracies_grid[[
        "param_kneighborsclassifier__n_neighbors",
        "mean_test_score",
        "sem_test_score"
    ]]
    .rename(columns={"param_kneighborsclassifier__n_neighbors": "n_neighbors"})
)
accuracies_grid
```

We can decide which number of neighbors is best by plotting the accuracy versus $K$,
as shown in {numref}`fig:06-find-k`.
Here we are using the shortcut `point=True` to layer a point and line chart.

```{code-cell} ipython3
:tags: [remove-output]

accuracy_vs_k = alt.Chart(accuracies_grid).mark_line(point=True).encode(
    x=alt.X("n_neighbors").title("Neighbors"),
    y=alt.Y("mean_test_score")
        .scale(zero=False)
        .title("Accuracy estimate")
)

accuracy_vs_k
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:06-find-k", accuracy_vs_k)
glue("best_k_unique", "{:d}".format(accuracies_grid["n_neighbors"][accuracies_grid["mean_test_score"].idxmax()]))
glue("best_acc", "{:.1f}".format(accuracies_grid["mean_test_score"].max()*100))
```

:::{glue:figure} fig:06-find-k
:name: fig:06-find-k

Plot of estimated accuracy versus the number of neighbors.
:::

We can also obtain the number of neighbours with the highest accuracy programmatically by accessing
the `best_params_` attribute of the fit `GridSearchCV` object. Note that it is still useful to visualize
the results as we did above since this provides additional information on how the model performance varies.
```{code-cell} ipython3
cancer_tune_grid.best_params_
```

+++

Setting the number of
neighbors to $K =$ {glue:text}`best_k_unique`
provides the highest cross-validation accuracy estimate ({glue:text}`best_acc`%). But there is no exact or perfect answer here;
any selection from $K = 30$ to $80$ or so would be reasonably justified, as all
of these differ in classifier accuracy by a small amount. Remember: the
values you see on this plot are *estimates* of the true accuracy of our
classifier. Although the
$K =$ {glue:text}`best_k_unique` value is
higher than the others on this plot,
that doesn't mean the classifier is actually more accurate with this parameter
value! Generally, when selecting $K$ (and other parameters for other predictive
models), we are looking for a value where:

- we get roughly optimal accuracy, so that our model will likely be accurate;
- changing the value to a nearby one (e.g., adding or subtracting a small number) doesn't decrease accuracy too much, so that our choice is reliable in the presence of uncertainty;
- the cost of training the model is not prohibitive (e.g., in our situation, if $K$ is too large, predicting becomes expensive!).

We know that $K =$ {glue:text}`best_k_unique`
provides the highest estimated accuracy. Further, {numref}`fig:06-find-k` shows that the estimated accuracy
changes by only a small amount if we increase or decrease $K$ near $K =$ {glue:text}`best_k_unique`.
And finally, $K =$ {glue:text}`best_k_unique` does not create a prohibitively expensive
computational cost of training. Considering these three points, we would indeed select
$K =$ {glue:text}`best_k_unique` for the classifier.

+++

### Under/Overfitting

To build a bit more intuition, what happens if we keep increasing the number of
neighbors $K$? In fact, the cross-validation accuracy estimate actually starts to decrease!
Let's specify a much larger range of values of $K$ to try in the `param_grid`
argument of `GridSearchCV`. {numref}`fig:06-lots-of-ks` shows a plot of estimated accuracy as
we vary $K$ from 1 to almost the number of observations in the data set.

```{code-cell} ipython3
:tags: [remove-output]

large_param_grid = {
    "kneighborsclassifier__n_neighbors": range(1, 385, 10),
}

large_cancer_tune_grid = GridSearchCV(
    estimator=cancer_tune_pipe,
    param_grid=large_param_grid,
    cv=10
)

large_cancer_tune_grid.fit(
    cancer_train[["Smoothness", "Concavity"]],
    cancer_train["Class"]
)

large_accuracies_grid = pd.DataFrame(large_cancer_tune_grid.cv_results_)

large_accuracy_vs_k = alt.Chart(large_accuracies_grid).mark_line(point=True).encode(
    x=alt.X("param_kneighborsclassifier__n_neighbors").title("Neighbors"),
    y=alt.Y("mean_test_score")
        .scale(zero=False)
        .title("Accuracy estimate")
)

large_accuracy_vs_k
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("fig:06-lots-of-ks", large_accuracy_vs_k)
```

:::{glue:figure} fig:06-lots-of-ks
:name: fig:06-lots-of-ks

Plot of accuracy estimate versus number of neighbors for many K values.
:::

+++

```{index} underfitting; classification
```

**Underfitting:** What is actually happening to our classifier that causes
this? As we increase the number of neighbors, more and more of the training
observations (and those that are farther and farther away from the point) get a
"say" in what the class of a new observation is. This causes a sort of
"averaging effect" to take place, making the boundary between where our
classifier would predict a tumor to be malignant versus benign to smooth out
and become *simpler.* If you take this to the extreme, setting $K$ to the total
training data set size, then the classifier will always predict the same label
regardless of what the new observation looks like. In general, if the model
*isn't influenced enough* by the training data, it is said to **underfit** the
data.

```{index} overfitting; classification
```

**Overfitting:** In contrast, when we decrease the number of neighbors, each
individual data point has a stronger and stronger vote regarding nearby points.
Since the data themselves are noisy, this causes a more "jagged" boundary
corresponding to a *less simple* model.  If you take this case to the extreme,
setting $K = 1$, then the classifier is essentially just matching each new
observation to its closest neighbor in the training data set. This is just as
problematic as the large $K$ case, because the classifier becomes unreliable on
new data: if we had a different training set, the predictions would be
completely different.  In general, if the model *is influenced too much* by the
training data, it is said to **overfit** the data.

```{code-cell} ipython3
:tags: [remove-cell]
alt.data_transformers.disable_max_rows()

cancer_plot = (
    alt.Chart(
        cancer_train,
    )
    .mark_point(opacity=0.6, filled=True, size=40)
    .encode(
        x=alt.X(
            "Smoothness",
            scale=alt.Scale(
                domain=(
                    cancer_train["Smoothness"].min() * 0.95,
                    cancer_train["Smoothness"].max() * 1.05,
                )
            ),
        ),
        y=alt.Y(
            "Concavity",
            scale=alt.Scale(
                domain=(
                    cancer_train["Concavity"].min() -0.025,
                    cancer_train["Concavity"].max() * 1.05,
                )
            ),
        ),
        color=alt.Color("Class", title="Diagnosis"),
    )
)

X = cancer_train[["Smoothness", "Concavity"]]
y = cancer_train["Class"]

# create a prediction pt grid
smo_grid = np.linspace(
    cancer_train["Smoothness"].min() * 0.95, cancer_train["Smoothness"].max() * 1.05, 100
)
con_grid = np.linspace(
    cancer_train["Concavity"].min() - 0.025, cancer_train["Concavity"].max() * 1.05, 100
)
scgrid = np.array(np.meshgrid(smo_grid, con_grid)).reshape(2, -1).T
scgrid = pd.DataFrame(scgrid, columns=["Smoothness", "Concavity"])

plot_list = []
for k in [1, 7, 20, 300]:
    cancer_pipe = make_pipeline(cancer_preprocessor, KNeighborsClassifier(n_neighbors=k))
    cancer_pipe.fit(X, y)

    knnPredGrid = cancer_pipe.predict(scgrid)
    prediction_table = scgrid.copy()
    prediction_table["Class"] = knnPredGrid

    # add a prediction layer
    prediction_plot = (
        alt.Chart(
            prediction_table,
            title=f"K = {k}"
        )
        .mark_point(opacity=0.2, filled=True, size=20)
        .encode(
            x=alt.X(
                "Smoothness",
                scale=alt.Scale(
                    domain=(
                        cancer_train["Smoothness"].min() * 0.95,
                        cancer_train["Smoothness"].max() * 1.05
                    ),
                    nice=False
                )
            ),
            y=alt.Y(
                "Concavity",
                scale=alt.Scale(
                    domain=(
                        cancer_train["Concavity"].min() -0.025,
                        cancer_train["Concavity"].max() * 1.05
                    ),
                    nice=False
                )
            ),
            color=alt.Color("Class", title="Diagnosis"),
        )
    )
    plot_list.append(cancer_plot + prediction_plot)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue(
    "fig:06-decision-grid-K",
    ((plot_list[0] | plot_list[1])
    & (plot_list[2] | plot_list[3])).configure_legend(
        orient="bottom", titleAnchor="middle"
    ),
)
```

:::{glue:figure} fig:06-decision-grid-K
:name: fig:06-decision-grid-K

Effect of K in overfitting and underfitting.
:::

+++

Both overfitting and underfitting are problematic and will lead to a model that
does not generalize well to new data. When fitting a model, we need to strike a
balance between the two. You can see these two effects in
{numref}`fig:06-decision-grid-K`, which shows how the classifier changes as we
set the number of neighbors $K$ to 1, 7, 20, and 300.

+++

### Evaluating on the test set

Now that we have tuned the K-NN classifier and set $K =$ {glue:text}`best_k_unique`,
we are done building the model and it is time to evaluate the quality of its predictions on the held out 
test data, as we did earlier in {numref}`eval-performance-clasfcn2`.
We first need to retrain the K-NN classifier
on the entire training data set using the selected number of neighbors.
Fortunately we do not have to do this ourselves manually; `scikit-learn` does it for
us automatically. To make predictions and assess the estimated accuracy of the best model on the test data, we can use the
`score` and `predict` methods of the fit `GridSearchCV` object. We can then pass those predictions to
the `precision`, `recall`, and `crosstab` functions to assess the estimated precision and recall, and print a confusion matrix.

```{index} scikit-learn;predict, scikit-learn;score, scikit-learn;precision_score, scikit-learn;recall_score, crosstab
```

```{code-cell} ipython3
cancer_test["predicted"] = cancer_tune_grid.predict(
    cancer_test[["Smoothness", "Concavity"]]
)

cancer_tune_grid.score(
    cancer_test[["Smoothness", "Concavity"]],
    cancer_test["Class"]
)
```

```{code-cell} ipython3
precision_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label='Malignant'
)
```

```{code-cell} ipython3
recall_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label='Malignant'
)
```

```{code-cell} ipython3
pd.crosstab(
    cancer_test["Class"],
    cancer_test["predicted"]
)
```
```{code-cell} ipython3
:tags: [remove-cell]
cancer_prec_tuned = precision_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label='Malignant'
)
cancer_rec_tuned = recall_score(
    y_true=cancer_test["Class"],
    y_pred=cancer_test["predicted"],
    pos_label='Malignant'
)
cancer_acc_tuned = cancer_tune_grid.score(
    cancer_test[["Smoothness", "Concavity"]],
    cancer_test["Class"]
)
glue("cancer_acc_tuned", "{:0.0f}".format(100*cancer_acc_tuned))
glue("cancer_prec_tuned", "{:0.0f}".format(100*cancer_prec_tuned))
glue("cancer_rec_tuned", "{:0.0f}".format(100*cancer_rec_tuned))
glue("mean_acc_ks", "{:0.0f}".format(100*accuracies_grid["mean_test_score"].mean()))
glue("std3_acc_ks", "{:0.0f}".format(3*100*accuracies_grid["mean_test_score"].std()))
glue("mean_sem_acc_ks", "{:0.0f}".format(100*accuracies_grid["sem_test_score"].mean()))
glue("n_neighbors_max", "{:0.0f}".format(accuracies_grid["n_neighbors"].max()))
glue("n_neighbors_min", "{:0.0f}".format(accuracies_grid["n_neighbors"].min()))
```

At first glance, this is a bit surprising: the accuracy of the classifier
has not changed much despite tuning the number of neighbors! Our first model
with $K =$ 3 (before we knew how to tune) had an estimated accuracy of {glue:text}`cancer_acc_1`%, 
while the tuned model with $K =$ {glue:text}`best_k_unique` had an estimated accuracy
of {glue:text}`cancer_acc_tuned`%. Upon examining {numref}`fig:06-find-k` again to see the
cross validation accuracy estimates for a range of neighbors, this result
becomes much less surprising. From {glue:text}`n_neighbors_min` to around {glue:text}`n_neighbors_max` neighbors, the cross
validation accuracy estimate varies only by around {glue:text}`std3_acc_ks`%, with
each estimate having a standard error around {glue:text}`mean_sem_acc_ks`%.
Since the cross-validation accuracy estimates the test set accuracy,
the fact that the test set accuracy also doesn't change much is expected.
Also note that the $K =$ 3 model had a precision 
precision of {glue:text}`cancer_prec_1`% and recall of {glue:text}`cancer_rec_1`%,
while the tuned model had
a precision of {glue:text}`cancer_prec_tuned`% and recall of {glue:text}`cancer_rec_tuned`%.
Given that the recall decreased&mdash;remember, in this application, recall
is critical to making sure we find all the patients with malignant tumors&mdash;the tuned model may actually be *less* preferred
in this setting. In any case, it is important to think critically about the result of tuning. Models tuned to
maximize accuracy are not necessarily better for a given application.

## Summary

Classification algorithms use one or more quantitative variables to predict the
value of another categorical variable. In particular, the K-nearest neighbors
algorithm does this by first finding the $K$ points in the training data
nearest to the new observation, and then returning the majority class vote from
those training observations. We can tune and evaluate a classifier by splitting
the data randomly into a training and test data set. The training set is used
to build the classifier, and we can tune the classifier (e.g., select the number
of neighbors in K-nearest neighbors) by maximizing estimated accuracy via
cross-validation. After we have tuned the model, we can use the test set to
estimate its accuracy.  The overall process is summarized in
{numref}`fig:06-overview`.

+++

```{figure} img/classification2/train-test-overview.png
:name: fig:06-overview

Overview of K-NN classification.
```

+++

```{index} scikit-learn;Pipeline, cross-validation, K-nearest neighbors; classification, classification
```

The overall workflow for performing K-nearest neighbors classification using `scikit-learn` is as follows:

1. Use the `train_test_split` function to split the data into a training and test set. Set the `stratify` argument to the class label column of the dataframe. Put the test set aside for now.
2. Create a `Pipeline` that specifies the preprocessing steps and the classifier.
3. Define the parameter grid by passing the set of $K$ values that you would like to tune.
4. Use `GridSearchCV` to estimate the classifier accuracy for a range of $K$ values. Pass the pipeline and parameter grid defined in steps 2. and 3. as the `param_grid` argument and the `estimator` argument, respectively.
5. Execute the grid search by passing the training data to the `fit` method on the `GridSearchCV` instance created in step 4.
6. Pick a value of $K$ that yields a high cross-validation accuracy estimate that doesn't change much if you change $K$ to a nearby value.
7. Create a new model object for the best parameter value (i.e., $K$), and retrain the classifier by calling the `fit` method.
8. Evaluate the estimated accuracy of the classifier on the test set using the `score` method.

In these last two chapters, we focused on the K-nearest neighbors algorithm,
but there are many other methods we could have used to predict a categorical label.
All algorithms have their strengths and weaknesses, and we summarize these for
the K-NN here.

**Strengths:** K-nearest neighbors classification

1. is a simple, intuitive algorithm,
2. requires few assumptions about what the data must look like, and
3. works for binary (two-class) and multi-class (more than 2 classes) classification problems.

**Weaknesses:** K-nearest neighbors classification

1. becomes very slow as the training data gets larger,
2. may not perform well with a large number of predictors, and
3. may not perform well when classes are imbalanced.

+++

## Predictor variable selection

```{note}
This section is not required reading for the remainder of the textbook. It is included for those readers
interested in learning how irrelevant variables can influence the performance of a classifier, and how to
pick a subset of useful variables to include as predictors.
```

```{index} irrelevant predictors
```

Another potentially important part of tuning your classifier is to choose which
variables from your data will be treated as predictor variables. Technically, you can choose
anything from using a single predictor variable to using every variable in your
data; the K-nearest neighbors algorithm accepts any number of
predictors. However, it is **not** the case that using more predictors always
yields better predictions! In fact, sometimes including irrelevant predictors can
actually negatively affect classifier performance.

+++ {"toc-hr-collapsed": true}

### The effect of irrelevant predictors

Let's take a look at an example where K-nearest neighbors performs
worse when given more predictors to work with. In this example, we modified
the breast cancer data to have only the `Smoothness`, `Concavity`, and
`Perimeter` variables from the original data. Then, we added irrelevant
variables that we created ourselves using a random number generator.
The irrelevant variables each take a value of 0 or 1 with equal probability for each observation, regardless
of what the value `Class` variable takes. In other words, the irrelevant variables have
no meaningful relationship with the `Class` variable.

```{code-cell} ipython3
:tags: [remove-cell]

np.random.seed(4)
cancer_irrelevant = cancer[["Class", "Smoothness", "Concavity", "Perimeter"]]
d = {
    f"Irrelevant{i+1}": np.random.choice(
        [0, 1], size=len(cancer_irrelevant), replace=True
    )
    for i in range(40)  ## in R textbook, it is 500, but the downstream analysis only uses up to 40
}
cancer_irrelevant = pd.concat((cancer_irrelevant, pd.DataFrame(d)), axis=1)
```

```{code-cell} ipython3
cancer_irrelevant[
    ["Class", "Smoothness", "Concavity", "Perimeter", "Irrelevant1", "Irrelevant2"]
]
```

Next, we build a sequence of K-NN classifiers that include `Smoothness`,
`Concavity`, and `Perimeter` as predictor variables, but also increasingly many irrelevant
variables. In particular, we create 6 data sets with 0, 5, 10, 15, 20, and 40 irrelevant predictors.
Then we build a model, tuned via 5-fold cross-validation, for each data set.
{numref}`fig:06-performance-irrelevant-features` shows
the estimated cross-validation accuracy versus the number of irrelevant predictors.  As
we add more irrelevant predictor variables, the estimated accuracy of our
classifier decreases. This is because the irrelevant variables add a random
amount to the distance between each pair of observations; the more irrelevant
variables there are, the more (random) influence they have, and the more they
corrupt the set of nearest neighbors that vote on the class of the new
observation to predict.

```{code-cell} ipython3
:tags: [remove-cell]

# get accuracies after including k irrelevant features
ks = [0, 5, 10, 15, 20, 40]
fixedaccs = list()
accs = list()
nghbrs = list()

for i in range(len(ks)):
    cancer_irrelevant_subset = cancer_irrelevant.iloc[:, : (4 + ks[i])]
    cancer_preprocessor = make_column_transformer(
        (
            StandardScaler(),
            list(cancer_irrelevant_subset.drop(columns=["Class"]).columns),
        ),
    )
    cancer_tune_pipe = make_pipeline(cancer_preprocessor, KNeighborsClassifier())
    param_grid = {
        "kneighborsclassifier__n_neighbors": range(1, 21),
    }  
    cancer_tune_grid = GridSearchCV(
        estimator=cancer_tune_pipe,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )

    X = cancer_irrelevant_subset.drop(columns=["Class"])
    y = cancer_irrelevant_subset["Class"]

    cancer_model_grid = cancer_tune_grid.fit(X, y)
    accuracies_grid = pd.DataFrame(cancer_model_grid.cv_results_)
    sorted_accuracies = accuracies_grid.sort_values(
        by="mean_test_score", ascending=False
    )

    res = sorted_accuracies.iloc[0, :]
    accs.append(res["mean_test_score"])
    nghbrs.append(res["param_kneighborsclassifier__n_neighbors"])

    ## Use fixed n_neighbors=3
    cancer_fixed_pipe = make_pipeline(
        cancer_preprocessor, KNeighborsClassifier(n_neighbors=3)
    )

    cv_5 = cross_validate(estimator=cancer_fixed_pipe, X=X, y=y, cv=5)
    cv_5_metrics = pd.DataFrame(cv_5).agg(["mean", "sem"])
    fixedaccs.append(cv_5_metrics.loc["mean", "test_score"])
```

```{code-cell} ipython3
:tags: [remove-cell]

summary_df = pd.DataFrame(
    {"ks": ks, "nghbrs": nghbrs, "accs": accs, "fixedaccs": fixedaccs}
)
plt_irrelevant_accuracies = (
    alt.Chart(summary_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("ks", title="Number of Irrelevant Predictors"),
        y=alt.Y(
            "accs",
            title="Model Accuracy Estimate",
            scale=alt.Scale(zero=False),
        ),
    )
)
glue("fig:06-performance-irrelevant-features", plt_irrelevant_accuracies)
```

:::{glue:figure} fig:06-performance-irrelevant-features
:name: fig:06-performance-irrelevant-features

Effect of inclusion of irrelevant predictors.
:::

Although the accuracy decreases as expected, one surprising thing about
{numref}`fig:06-performance-irrelevant-features` is that it shows that the method
still outperforms the baseline majority classifier (with about {glue:text}`cancer_train_b_prop`% accuracy)
even with 40 irrelevant variables.
How could that be? {numref}`fig:06-neighbors-irrelevant-features` provides the answer:
the tuning procedure for the K-nearest neighbors classifier combats the extra randomness from the irrelevant variables
by increasing the number of neighbors. Of course, because of all the extra noise in the data from the irrelevant
variables, the number of neighbors does not increase smoothly; but the general trend is increasing. {numref}`fig:06-fixed-irrelevant-features` corroborates
this evidence; if we fix the number of neighbors to $K=3$, the accuracy falls off more quickly.

```{code-cell} ipython3
:tags: [remove-cell]

plt_irrelevant_nghbrs = (
    alt.Chart(summary_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("ks", title="Number of Irrelevant Predictors"),
        y=alt.Y(
            "nghbrs",
            title="Tuned number of neighbors",
        ),
    )
)
glue("fig:06-neighbors-irrelevant-features", plt_irrelevant_nghbrs)
```

:::{glue:figure} fig:06-neighbors-irrelevant-features
:name: fig:06-neighbors-irrelevant-features

Tuned number of neighbors for varying number of irrelevant predictors.
:::

```{code-cell} ipython3
:tags: [remove-cell]

melted_summary_df = summary_df.melt(
            id_vars=["ks", "nghbrs"], var_name="Type", value_name="Accuracy"
        )
melted_summary_df["Type"] = melted_summary_df["Type"].apply(lambda x: "Tuned K" if x=="accs" else "K = 3")

plt_irrelevant_nghbrs_fixed = (
    alt.Chart(
        melted_summary_df
    )
    .mark_line(point=True)
    .encode(
        x=alt.X("ks", title="Number of Irrelevant Predictors"),
        y=alt.Y(
            "Accuracy",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("Type"),
    )
)
glue("fig:06-fixed-irrelevant-features", plt_irrelevant_nghbrs_fixed)
```

:::{glue:figure} fig:06-fixed-irrelevant-features
:name: fig:06-fixed-irrelevant-features

Accuracy versus number of irrelevant predictors for tuned and untuned number of neighbors.
:::

+++

### Finding a good subset of predictors

So then, if it is not ideal to use all of our variables as predictors without consideration, how
do we choose which variables we *should* use?  A simple method is to rely on your scientific understanding
of the data to tell you which variables are not likely to be useful predictors. For example, in the cancer
data that we have been studying, the `ID` variable is just a unique identifier for the observation.
As it is not related to any measured property of the cells, the `ID` variable should therefore not be used
as a predictor. That is, of course, a very clear-cut case. But the decision for the remaining variables
is less obvious, as all seem like reasonable candidates. It
is not clear which subset of them will create the best classifier. One could use visualizations and
other exploratory analyses to try to help understand which variables are potentially relevant, but
this process is both time-consuming and error-prone when there are many variables to consider.
Therefore we need a more systematic and programmatic way of choosing variables.
This is a very difficult problem to solve in
general, and there are a number of methods that have been developed that apply
in particular cases of interest. Here we will discuss two basic
selection methods as an introduction to the topic. See the additional resources at the end of
this chapter to find out where you can learn more about variable selection, including more advanced methods.

```{index} variable selection; best subset
```

```{index} see: predictor selection; variable selection
```

The first idea you might think of for a systematic way to select predictors
is to try all possible subsets of predictors and then pick the set that results in the "best" classifier.
This procedure is indeed a well-known variable selection method referred to
as *best subset selection* {cite:p}`bealesubset,hockingsubset`.
In particular, you

1. create a separate model for every possible subset of predictors,
2. tune each one using cross-validation, and
3. pick the subset of predictors that gives you the highest cross-validation accuracy.

Best subset selection is applicable to any classification method (K-NN or otherwise).
However, it becomes very slow when you have even a moderate
number of predictors to choose from (say, around 10). This is because the number of possible predictor subsets
grows very quickly with the number of predictors, and you have to train the model (itself
a slow process!) for each one. For example, if we have 2 predictors&mdash;let's call
them A and B&mdash;then we have 3 variable sets to try: A alone, B alone, and finally A
and B together. If we have 3 predictors&mdash;A, B, and C&mdash;then we have 7
to try: A, B, C, AB, BC, AC, and ABC. In general, the number of models
we have to train for $m$ predictors is $2^m-1$; in other words, when we
get to 10 predictors we have over *one thousand* models to train, and
at 20 predictors we have over *one million* models to train!
So although it is a simple method, best subset selection is usually too computationally
expensive to use in practice.

```{index} variable selection; forward
```

Another idea is to iteratively build up a model by adding one predictor variable
at a time. This method&mdash;known as *forward selection* {cite:p}`forwardefroymson,forwarddraper`&mdash;is also widely
applicable and fairly straightforward. It involves the following steps:

1. Start with a model having no predictors.
2. Run the following 3 steps until you run out of predictors:
    1. For each unused predictor, add it to the model to form a *candidate model*.
    2. Tune all of the candidate models.
    3. Update the model to be the candidate model with the highest cross-validation accuracy.
3. Select the model that provides the best trade-off between accuracy and simplicity.

Say you have $m$ total predictors to work with. In the first iteration, you have to make
$m$ candidate models, each with 1 predictor. Then in the second iteration, you have
to make $m-1$ candidate models, each with 2 predictors (the one you chose before and a new one).
This pattern continues for as many iterations as you want. If you run the method
all the way until you run out of predictors to choose, you will end up training
$\frac{1}{2}m(m+1)$ separate models. This is a *big* improvement from the $2^m-1$
models that best subset selection requires you to train! For example, while best subset selection requires
training over 1000 candidate models with 10 predictors, forward selection requires training only 55 candidate models.
Therefore we will continue the rest of this section using forward selection.

```{note}
One word of caution before we move on. Every additional model that you train
increases the likelihood that you will get unlucky and stumble
on a model that has a high cross-validation accuracy estimate, but a low true
accuracy on the test data and other future observations.
Since forward selection involves training a lot of models, you run a fairly
high risk of this happening. To keep this risk low, only use forward selection
when you have a large amount of data and a relatively small total number of
predictors. More advanced methods do not suffer from this
problem as much; see the additional resources at the end of this chapter for
where to learn more about advanced predictor selection methods.
```

+++

### Forward selection in Python

```{index} variable selection; implementation
```

We now turn to implementing forward selection in Python.
First we will extract a smaller set of predictors to work with in this illustrative example&mdash;`Smoothness`,
`Concavity`, `Perimeter`, `Irrelevant1`, `Irrelevant2`, and `Irrelevant3`&mdash;as well as the `Class` variable as the label.
We will also extract the column names for the full set of predictors.

```{code-cell} ipython3
cancer_subset = cancer_irrelevant[
    [
        "Class",
        "Smoothness",
        "Concavity",
        "Perimeter",
        "Irrelevant1",
        "Irrelevant2",
        "Irrelevant3",
    ]
]

names = list(cancer_subset.drop(
    columns=["Class"]
).columns.values)

cancer_subset
```

To perform forward selection, we could use the
[`SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
from `scikit-learn`; but it is difficult to combine this approach with parameter tuning to find a good number of neighbors
for each set of features. Instead we will code the forward selection algorithm manually.
In particular, we need code that tries adding each available predictor to a model, finding the best, and iterating.
If you recall the end of the wrangling chapter, we mentioned
that sometimes one needs more flexible forms of iteration than what
we have used earlier, and in these cases one typically resorts to
a *for loop*; see
the [control flow section](https://wesmckinney.com/book/python-basics.html#control_for) in
*Python for Data Analysis* {cite:p}`mckinney2012python`.
Here we will use two for loops: one over increasing predictor set sizes
(where you see `for i in range(1, n_total + 1):` below),
and another to check which predictor to add in each round (where you see `for j in range(len(names))` below).
For each set of predictors to try, we extract the subset of predictors,
pass it into a preprocessor, build a `Pipeline` that tunes
a K-NN classifier using 10-fold cross-validation,
and finally records the estimated accuracy.

```{code-cell} ipython3
from sklearn.compose import make_column_selector

accuracy_dict = {"size": [], "selected_predictors": [], "accuracy": []}

# store the total number of predictors
n_total = len(names)

# start with an empty list of selected predictors
selected = []

# create the pipeline and CV grid search objects
param_grid = {
    "kneighborsclassifier__n_neighbors": range(1, 61, 5),
}
cancer_preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include="number"))
)
cancer_tune_pipe = make_pipeline(cancer_preprocessor, KNeighborsClassifier())
cancer_tune_grid = GridSearchCV(
    estimator=cancer_tune_pipe,
    param_grid=param_grid,
    cv=10,
    n_jobs=-1
)

# for every possible number of predictors
for i in range(1, n_total + 1):
    accs = np.zeros(len(names))
    # for every possible predictor to add
    for j in range(len(names)):
        # Add remaining predictor j to the model
        X = cancer_subset[selected + [names[j]]]
        y = cancer_subset["Class"]

        # Find the best K for this set of predictors
        cancer_tune_grid.fit(X, y)
        accuracies_grid = pd.DataFrame(cancer_tune_grid.cv_results_)

        # Store the tuned accuracy for this set of predictors
        accs[j] = accuracies_grid["mean_test_score"].max()

    # get the best new set of predictors that maximize cv accuracy
    best_set = selected + [names[accs.argmax()]]

    # store the results for this round of forward selection
    accuracy_dict["size"].append(i)
    accuracy_dict["selected_predictors"].append(", ".join(best_set))
    accuracy_dict["accuracy"].append(accs.max())

    # update the selected & available sets of predictors
    selected = best_set
    del names[accs.argmax()]

accuracies = pd.DataFrame(accuracy_dict)
accuracies
```

```{index} variable selection; elbow method
```

Interesting! The forward selection procedure first added the three meaningful variables `Perimeter`,
`Concavity`, and `Smoothness`, followed by the irrelevant variables. {numref}`fig:06-fwdsel-3`
visualizes the accuracy versus the number of predictors in the model. You can see that
as meaningful predictors are added, the estimated accuracy increases substantially; and as you add irrelevant
variables, the accuracy either exhibits small fluctuations or decreases as the model attempts to tune the number
of neighbors to account for the extra noise. In order to pick the right model from the sequence, you have
to balance high accuracy and model simplicity (i.e., having fewer predictors and a lower chance of overfitting).
The way to find that balance is to look for the *elbow*
in {numref}`fig:06-fwdsel-3`, i.e., the place on the plot where the accuracy stops increasing dramatically and
levels off or begins to decrease. The elbow in {numref}`fig:06-fwdsel-3` appears to occur at the model with
3 predictors; after that point the accuracy levels off. So here the right trade-off of accuracy and number of predictors
occurs with 3 variables: `Perimeter, Concavity, Smoothness`. In other words, we have successfully removed irrelevant
predictors from the model! It is always worth remembering, however, that what cross-validation gives you
is an *estimate* of the true accuracy; you have to use your judgement when looking at this plot to decide
where the elbow occurs, and whether adding a variable provides a meaningful increase in accuracy.

```{code-cell} ipython3
:tags: [remove-cell]

fwd_sel_accuracies_plot = (
    alt.Chart(accuracies)
    .mark_line(point=True)
    .encode(
        x=alt.X("size", title="Number of Predictors"),
        y=alt.Y(
            "accuracy",
            title="Estimated Accuracy",
            scale=alt.Scale(zero=False),
        ),
    )
)
glue("fig:06-fwdsel-3", fwd_sel_accuracies_plot)
```

:::{glue:figure} fig:06-fwdsel-3
:name: fig:06-fwdsel-3

Estimated accuracy versus the number of predictors for the sequence of models built using forward selection.
:::

+++

```{note}
Since the choice of which variables to include as predictors is
part of tuning your classifier, you *cannot use your test data* for this
process!
```

