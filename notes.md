largely informed by:

https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e

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


--------------------------------------------------
##Coming back a few months later

now using:
https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf

Okay so...
how is this all going to go down...

--------------------------------------------------

calculating the values:

  xxxxxxxx

x xxxxxxxx
x xxxxxxxx

weights:
you put in the vector len 8
multiply that by a 8x2 matrix
get out a vector len 2

xx
xx
xx

biases:
add that vector of len 2 by another vector of len 2
get another vector of len 2

rinse and repeat

--------------------------------------------------

so then what's the backprop?

It's multivar calculus, which I know from youtube but I dont
feel that confident about. I wish I took a class, or a really 
wish that i just felt confident from what I knew...

so when you take a multivar derivative you make everything
else besides what you're solving for a constant...

so what is the formula for this?
1 layer:
out = sigmoid(b1 + w1(in))

2 layer:
out = sigmoid(b1 + w1(sigmoid(b2 + w2(in))))

3 layer:
out = sigmoid(b1 + w1(sigmoid(b2 + w2(sigmoid(b3 + w3(in))))))

and so forth?
so it's pretty repetitious
we can probably figure out how to do calculus on this...

but at the end of the day we want to find the slope of the 
error so we need to get these into their error form:

we're using sum squared error
E = 1/2 * sum(label - out)^2

an example from a weight:
	delta w1 ~ -1 * partial(E)/partial(w1)


##############

calculus intermission:

dy/dx * dx/dt
y = f(x)
x = g(t)

the way this works is that you just use x like a variable
nothing special, and then you do normal derivative stuff.

then after you're done you can substitute t back in through.
leibniz isn't so scary after all

##############

we're now going to use the notation from the link above and 
call (bx + wx*a(x-1)) "net"

1 layer:
out = error(sigmoid(b1 + w1(in)))

(s = sigmoid)
dE/db1 = dE/ds*ds/dnet*dnet/db1
dE/dw1 = dE/ds*ds/dnet*dnet/dw1

2 layer:
out = error(sigmoid(b1 + w1(sigmoid(b2 + w2(in)))))

dE/db1 = dE/ds*ds/dnet*dnet/db1
dE/dw1 = dE/ds*ds/dnet*dnet/dw1
dE/db2 = dE/ds*ds/dnet*dnet/ds*ds/dnet*dnet/db2
dE/dw2 = dE/ds*ds/dnet*dnet/ds*ds/dnet*dnet/db2

--------------------------------------------------

out_x = output of layer x
out = output of net
w_x = weights of layer x
in_x = input of layer x
label = labeled data, target

so the terms are:
dE/ds = -(label - out), -cost
ds/dnet = out_x(1 - out_x)
dnet/ds = w_x
dnet/dwx = in_x
dnet/dbx = 1

I feel like there's probably an eaiser way to deal with this
with how the ins and outs of things are all over the place

maybe the thing to do is to have an array that stores all the
inputs for each layer, and then have an output value
to store the total net output

--------------------------------------------------
done-
what we want is to just read it first and really understand what I did 


after that we want to try to implement another 2 layers to get a feeling 
for what needs to be repeated with each layer.


Then we want to generalize the layers so we can parameterize them.


Then we want to code up the generalization of the layers.


profit? - no :(

--------------------------------------------------


dE_dnet = cost * np.multiply(out1, np.subtract(1, out1))

    weights1 -= l_rate * np.outer(out2, dE_dnet)
    biases1 -= l_rate * dE_dnet

    dE_ds2 = np.matmul(weights1, dE_dnet)


    ds_dnet2 = np.multiply(out2, np.subtract(1, out2))

    dnet1/dnet2 = dnet1/ds2 * ds2/dnet2
    dnet1/dnet2 = w1 * out2(1 - out2) 

    each layer should:
        # set up current layer
        ds_dnet = np.multiply(layer_output, np.subtract(1, layer_output))
        layer_chain = prev_layer_chain * ds_dnet

        # adjust weights / biases
        weights -= l_rate * np.outer(layer_input, layer_chain)
        biases -= l_rate * layer_chain

        # setup for next layer
        layer_chain = np.matmul(self.weights, layer_chain)
        return layer_chain


    weights2 -= l_rate * np.outer(vect_in, dE_ds2 * ds_dnet2)
    biases2 -= l_rate * dE_ds2 * ds_dnet2

    for each layer I want:
        dsx/dnetx * dnetx/dsx-1


    dE/db1 = dE/ds1*ds1/dnet1 * dnet1/db1
    dE/dw1 = dE/ds1*ds1/dnet1 * dnet1/dw1

    dE/db2 = dE/ds1*ds1/dnet1 * dnet1/ds2*ds2/dnet2 * dnet2/db2
    dE/dw2 = dE/ds1*ds1/dnet1 * dnet1/ds2*ds2/dnet2 * dnet2/dw2

    dE/db3 = dE/ds1*ds1/dnet1 * dnet1/ds2*ds2/dnet2 * dnet2/ds3*ds3/dnet3 * dnet3/db3
    dE/dw3 = dE/ds1*ds1/dnet1 * dnet1/ds2*ds2/dnet2 * dnet2/ds3*ds3/dnet3 * dnet3/dw3


    out_x = output of layer x
    out = output of net
    w_x = weights of layer x
    in_x = input of layer x
    label = labeled data, target

    so the terms are:
    dE/ds = -(label - out), - cost
    ds/dnet = out_x(1 - out_x)
    dnet/ds = w_x
    dnet/dwx = in_x
    dnet/dbx = 1


--------------------------------------------------

so we want to try mini batches?

it seems like the thing to do for batches is to take the error
over batch_size and then average that, and then do gradient 
descent with that average error. 

This seems weird, but doable form this standpoint, but 
what about the layer_in and layer_out that's used in the 
derivative of the sigmoid function?

could i average these as well? If i do they will basically
become noise most likely, which I guess is okay?
But then why include them? I guess that they might have
an overall flavor that can be adjusted for, that is
the point of these mini batches, so yea lets try that out.

--------------------------------------------------

what's after mini-batches?

Things to think about:
	the activation derivative almost never helps, but
	I think I might keep it around just because
	relu derivative = 0 =< 0, 1 > 0

	I might move the layer in/out over to the net
	class becasue it's eating up space and processing
	so be storing things so much extra. I'm pretty 
	sure that encapsulation would say to stick the 
	i/o in the layers (each layer manages their own io)
	but this runs pretty slow as is and I'd like 
	it to be faster...

implement relu derivative
push io sotrage up to net class

--------------------------------------------------
