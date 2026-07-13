# Methods used in Belkin's double descent experiments

**GOAL:** We want to recreate the double-descent curves seen in Belkin's paper using processes that match Belkin's as closely as possible. Once we have done this, we can make adjustments in the processes; subsequent changes in the curves can then be attributed to these adjustments.

## 0. Datasets overview
### MNIST: Collection of 60,000 training images and 10,000 testing images of handwritten digits. Task is to classify into one of 10 categories (the 10 digits)
### CIFAR-10: collection of 50,000 training images and 10,000 testing images of common items like airplane, car, bird, cat, etc. Task is to classify into one of 10 categories.
### 20-Newsgroups: collection of nearly 18,000 emails, each of which relates to exactly 1 of 20 topics. The emails contain headers, body text, and images. Task is to classify into one of the 20 topics.
### TIMIT: collection of audio recordings of spoken sentences. There were 630 speakers who each spoke 10 randomly selected sentences (from a pool of 2,342 sentences). Speakers were 70% male, 30% female. Task is to classify into one of eight major dialect regions: New England, Northern, North Midland, South Midland, Southern, New York City, Western, Army Brat (moved around). *Note that this dataset isn't freely available--must purchase LDC license*
### SVHN: collection of over 600,000 photos of house numbers from Google Street View. Images are 32x32. 73,257 digits for training, 26,032 digits for testing, and 531,131 additional, somewhat less difficult samples, to use as extra training data. Task is to classify each digit into 1 of 10 categories (the digits)

## 1. Random Fourier Features

### 1.1 MNIST
*Found in the main text, page 15851, fig. 2*

Data details:
* Classification problem
* Choose a subset of size $n=10^4$ as training data. It is of the form$$\{(x_1,y_1),(x_2,y_2),\dots(x_n,y_n)\}\subset\mathbb{R}^d\times\mathbb{R}$$
* *Is this subset randomly chosen for each $N$?????????*

Model details:
* 2-layer neural network with fixed weights in the first layer
  * *Are these fixed weights kept the same?????*
* RFF model family $\mathcal{H}_N$ consists of functions of the form $$h:\mathbb{R}^d\to\mathbb{C}\;\;\;\text{defined by}\;\;\;h(x)=\sum_{k=1}^Na_ke^{i\langle v_k,x\rangle},$$where $\{v_1,v_2,\dots v_N\}\subset\mathbb{R}^d$ are sampled independently from the standard normal distribution in $\mathbb{R}^d$. Thus there are $N$ parameters used: $(a_1,a_2,\dots a_N)$.
* *Is $\{v_1,v_2,\dots v_N\}\subset\mathbb{R}^d$ sampled exactly once in the experiement??????? Or multiple times? IDK how RFF is generally done*

Training details:
* Find the predictor $h_{n,N}$ by$$h_{n,N}=\argmin_{h\in\mathcal{H}_N}\frac{1}{n}\sum_{i=1}^N\left(h(x_i)-y_i\right)^2$$
* When $N>n$, the minimizer isn't unique. So choose minimizer whose coefficients $(a_1,a_2,\dots a_N)$ which has the minimum $\ell_2$ norm. Intended as an approximation of RKHS norm.
* *Done using OLS/Moore-Penrose pseudoinverse??????*

Experiment details:
* Train for the following values of $N$ ($\times10^3$):$$0.5, 1,1.5, 2,2.5,3,4,5,6,7,8,9,10,11,12,13,14,20,40,60$$
* Plot the *test risk* and *train risk* separately measured by zero-one loss and mean squared loss for each $N$.
* Also, plot the $\ell_2$-norm of the resulting coefficient vector for each $N$.

### 1.2 CIFAR-10
*Found in the appendix, p. 3, fig. S1*

Data details:
* Color images converted to grayscale. Appendix does not specify exactly how.
* "Maximum range of each feature is scaled to $[0,1]$." Again, doesn't specify exactly how. Do we divide each feature by the maximum value of each respective feature? Divide each value by 255?
* Choose subset of size $n=10^4$

Model details:
* I assume it's the same as in **1.1**.

Training details:
* Assumed to be the same as in **1.1**.

Experiment details:
* Train *once* for the following values of $N$ ($\times10^3$):$$0.5,1,1.5,2,2.5,3,3.5,5,6,7,8,9,10,11,12,13,14,20,40,60$$It's done only once for each $N$ because "the results were empirically highly consistent."
* Plot test risk and train risk, as well as $\ell_2$-norm of solution coefficient vectors for each $N$.

### 1.3 20-Newsgroups
*Found in the appendix, p. 3, fig.S1*

Data details:
* "Transform each sparse feature vector (bag of words) into a dense feature vector by summing up its corresponding word embeddings obtained from reference [14]"
* Randomly pick $\frac18$ of the dataset to use as test set. *I assume this means they used* $\frac78$ *of the dataset as the training set??????????????*

Model details:
* Assumed to be the same as **1.1**.

Training details:
* Assumed to be the same as **1.1**.

### 1.4 TIMIT


### 1.5 SVHN

## 2. Fully-connected neural net

### 2.1 MNIST (A)
*Found in main text pp. 15851-15852, fig. 3. Also seen in appendix, fig. S4*

Data details:
* Choose subset of MNIST with size $n=4{,}000$

Model details:
* Fully-connected net with exactly one hidden layer of $H$ units. Since each image in MNIST is 28x28, we get the total number of parameters to be $N=795H+10$ for any $H$.
 * Matrix form:$$h=\sigma\left(W^{(1)}x+b^{(1)}\right)$$ $$\hat{y}=W^{(2)}h+b^{(2)}$$*Doesn't specify which activation* $\sigma$ *is used but probably ReLU??????*


Training details:
* Mean squared loss as loss function.
* Find minimum with SGD.
* Use the following weight reuse scheme before interpolation, and random initialization after it.
  * For the first net, (minimum $N$-value and $H$-value, $H_0$), initialize using standard Glorot-uniform distribution.
  * For subsequent networks with $H_i$ hidden units, initialize the first $H_{i-1}$ hidden units to be the parameters/weights of the previous (trained) network. The remaining weights are initialized with a draw from $\mathcal{N}(0,0.01)$.
* Different training cutoffs:
  * For networks with $N$ less than interpolation threshold,  decay the step size by 10% after each of 500 epochs. Also stop after classification error reaches zero or 6000 epochs, whichever happens earlier.
  * For networks with $N$ greater than interpolation threshold, fixed step size is used, and training is stopped after 6000 epochs.

Experiment details:
* Train the model for the following $N$-values ($\times10^3$): $$4,5,7,10,12,20,24,27,30,33,35,37,38,39,39.5,40,45,50,70,85,200,250,800$$
* In reality, this means we train for the following $H$-values (roughly): $$5,6,9,13,15,25,30,34,38,41,44,47,48,49,50,51,57,63,88,107,252,314,1006$$
* Repeat the experiment 5 times and report average risk

### 2.2 CIFAR-10a
*Appendix fig. S4*

Data details:
* Choose subset with size $n=960$, with only 2 classes (cat, dog).
* Downsampled 8x8 image figures.

Model details:
* Fully-connected neural net with exactly one hidden layer of $H$ units
* Since each image is 8x8, we get the total number of parameters to be $N=67H+2$ for any $H$
* Matrix form same as **2.1**

Training details:
* Same as **2.1**

Experiment details:
* Train model for the following $N$-values ($\times 10^3$):$$69,136,203,270,337,404,471,739,940,1476,1744,1811,1878,1945,2079,2146,2481,2548,2682,2950,3486,7975,11995,19968,59967,249979,299961,650036$$
* In reality, this translates to $H$-values of$$1,2,3,4,5,6,7,11,14,22,26,27,28,29,31,32,37,38,40,44,52,119,179,298,895,3731,4477,9702$$
* The rest is the same as **2.1**

### 2.3 MNIST (appendix again)
*Appendix fig. S4*

Data details:
* Same as **2.1**

Model details:
* Same as **2.1**

Training details:
* No weight reuse (random initialization for all ranges of parameters)
* The rest is the same as **2.1**

Experiment details
* Same except cut off the first few $H$-values (start around $N=14$)


## 3. Random Forest

### 3.1 MNIST (classification)
*Found in main text p. 15852, fig. 4*

Data details:
* Choose subset of size $n=10^4$.

### 3.2 MNIST (regression)

## 4. Random ReLU features

### 4.1 MNIST
*Found in appendix, fig. S3, pp. 4-5* 

Data details
* yeet

Model details:
* Random ReLU features model family $\mathcal{H}_N$ with $N$ parameters consists of functions of the form$$h:\mathbb{R}^d\to\mathbb{R}\;\;\;\text{defined by}\;\;\;h(x)=\sum_{k=1}^Na_k\max\left(\langle v_k,x\rangle,0\right).$$
* Vectors $v_1,\dots,v_N$ sampled independently from uniform distribution over surface of unit sphere in $\mathbb{R}^d$.

Training details:
* Coefficients $a_1,\dots,a_N$ learned using linear regression

Experiment details:
