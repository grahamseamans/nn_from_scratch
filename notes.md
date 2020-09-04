largely informed by:

https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e

ways to go:
1) add more layers
2) use 3d to take in 2d image down to output vector. skips flatten step.
3) make it work when initialized as zero?


we know what the loss function is, it's:
sum(ground truth - nets output)
but how do we take the derivative of this across so 
many different parts of a net. like how do we know what the 
loss is of a single weight?

l_rate = leaning rate

so for the weights: w1 = w0 - error * vect_in * l_rate
and for biases: b1 = b0 - error * l_rate

for weights:
weights is [786, 10] error is [10,1] vect in is [786,1] l_rate is scalar

so mult vect in in some weird way by error? 
n vect * m vect = n x m matrix?? 
this is called the outer product, np.outer()
then all we need to do is mult by a scalar
weight1 = weight0 - (np.outer(vect_in, error) * l_rate)

for biases:
bases is [10,1] error is [10,1] l_rate is scalar
bias1 = bias0 - (error * l_rate)

leibnitz is really confusing. especially for the chain rule.
wish i had a math person to talk to.
