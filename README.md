# Self-assessment in neural networks

This is a new type of layer for neural networks in PyTorch: It gives the network the ability to assess the reliability of its own features, and to prioritize neurons that prove useful.

This invention is a spin-off of a private research project of mine. Preliminary results showed that using this layer lead to a minor improvement on the MNIST dataset. However, the improvement was too small to know for sure if it is useful.

Since this project does not go in the same direction as my primary research, I am publishing it here. Maybe someone else wants to have a look at it.


## Intuition behind the idea


The intuition behind this new type of layer is as follows:

When a normal neural network is used to solve a regression problem, it does not provide a measure of how certain it is of the correctness of its output. If the network has no idea what the output should actually be, a well trained network will simply give the value that minimizes the expected loss over all training examples, without giving any indication that this is effectively just a wild guess.

There is a difference between saying "I have checked every data source I can find and the answer is 0.3" and saying "I have no idea where to even start, but the average is 0.3 so let's go with that". Contemporary neural networks do not capture that difference.

It seems clear to me that this information ought to be useful to have. The question is: Is it useful enough to be worth the extra effort to calculate?

For more information, check out this blog entry I wrote about it: [https://floriandietz.me/neural_networks_self_assessment/](https://floriandietz.me/neural_networks_self_assessment/)


## Usage

Have a look at the code in selfAssessment.py

There are two ways this can be useful:

(1) To get an estimate of the reliability of the prediction of your network along with the prediction itself.

(2) To (try to) improve the networks total performance by giving the network the ability to rate each of its own features. This allows the network to model uncertainty over each of its own features, which can be used to improve performance.

### (1) self-assessment of outputs

To get a self-assessment at the end of a regression task, just use a layer like this at or near the end of the network:

```python
self.final_layer = SelfAssessment(500, num_outputs, sass_features=1)
```

and call it like this:

```python
output, self_assessment = self.final_layer(x)
```

### (2) self-assessment to improve performance

To use the self-assessment to improve training, just replace any code that looks like this

```python
self.fc1 = nn.Linear(200, 300)
```


and call it like this:

```python
self.fc1 = SelfAssessment(200, 270, sass_features=30, add_sass_features_to_output=True)
```

By the way: Since the sass component's output has a different scale than the output of the main component, I would recommend using a hyperparameter optimizer that uses a different learning rate for each parameter, such as the Adam optimizer.
