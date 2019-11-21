import math

import torch
from torch import autograd
import torch.nn as nn


class SelfAssessmentFunction(autograd.Function):
    """
    Implements two linear layers, one called 'main' and one called 'sass' (for 'self-assessment').

    The main layer behaves just like a regular linear layer.

    The sass layer behaves like a linear layer on the forward pass, but it uses a custom backward method to calculate its gradient:

    The gradient the sass layer receives from backpropagation is ignored completely. It is replaced with the result of a custom loss function: It uses MLEloss, with the absolute values of the gradient of the main layer as the target. This new loss function is then used to calculate the gradient of the sass layer.

    As a result, the sass layer learns to approximate the average absolute gradient of the main layer.

    Each neuron in the sass layer reacts to a different subset of the neurons in the main layer, which is controlled by output_to_sass_mean_compression.
    """

    @staticmethod
    def forward(ctx, input, weight_main, bias_main, weight_sass, bias_sass, output_to_sass_mean_compression):
        # Both feed-forward portions are just the result of applying the respective layer to the input
        output_main = input.mm(weight_main.t())
        output_main += bias_main.unsqueeze(0).expand_as(output_main)
        output_sass = input.mm(weight_sass.t())
        output_sass += bias_sass.unsqueeze(0).expand_as(output_sass)
        ctx.save_for_backward(input, weight_main, bias_main, weight_sass, bias_sass, output_sass, output_to_sass_mean_compression)
        return output_main, output_sass

    @staticmethod
    def backward(ctx, grad_main, grad_sass):
        input, weight_main, bias_main, weight_sass, bias_sass, output_sass, output_to_sass_mean_compression = ctx.saved_tensors
        grad_input = grad_weight_main = grad_bias_main = grad_weight_sass = grad_bias_sass = grad_output_to_sass_mean_compression = None
        # Perform normal gradient calculations on the main layer
        grad_weight_main = grad_main.t().mm(input)
        grad_bias_main = grad_main.sum(0)
        # For the sass layer, ignore the grad_sass and recompute it:
        # The grad_sass is computed through MLELoss, with the absolute of the gradient of the main neurons as the target.
        # Each neuron in sass measures a subset of the main neurons. This mapping is done by output_to_sass_mean_compression.
        target = grad_main.abs().mm(output_to_sass_mean_compression)
        grad_sass = (output_sass - target) * 2
        # Apply this new gradient
        grad_weight_sass = grad_sass.t().mm(input)
        grad_bias_sass = grad_sass.sum(0)
        # Calculate the gradient for the input
        grad_input = grad_main.mm(weight_main)
        return grad_input, grad_weight_main, grad_bias_main, grad_weight_sass, grad_bias_sass, grad_output_to_sass_mean_compression


class SelfAssessment(nn.Module):
    """
    Implements a linear layer as well as a self-assessment layer, which is a second linear layer that is trained to predict the gradient of the first linear layer.

    If add_sass_features_to_output=True, the results of both layers are combined into a single tensor. Otherwise they are returned separately.

    The parameter sass_features specifies the number of different self-assessment neurons, which must be a multiple of out_features.

    For example:

    If you set sass_features=1 and add_sass_features_to_output=False, this module behaves like a normal linear layer, but you will get a secondary output that predicts the average absolute gradient of the entire layer.

    If you set out_features=200, sass_features=10, add_sass_features_to_output=True, you end up with an output vector of size 210. 200 of those are normal results of the linear layer, while the remaining 10 are predictions of the mean absolute gradient of the 10 blocks of 20 neurons. These 10 additional predictions may improve network performance because subsequent layers can use them as an estimate for how reliable the 10*20 main neurons are for the given example.

    Note for interpretation: A higher value of the self-assessment means that the network is less sure of itself. (The value measures how much the network expects to learn from the new data.)
    """

    def __init__(self, in_features, out_features, sass_features=1, add_sass_features_to_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sass_features = sass_features
        self.add_sass_features_to_output = add_sass_features_to_output
        if float(out_features) % float(sass_features) != 0:
            raise ValueError("The number of output features (out_features) must be a multiple of the number of self-assessment features (sass_features).")
        # Create one layer for the calculation itself, and another for the self assessment
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_main = nn.Parameter(torch.Tensor(out_features))
        self.weight_sass = nn.Parameter(torch.Tensor(sass_features, in_features))
        self.bias_sass = nn.Parameter(torch.Tensor(sass_features))
        self.reset_parameters()
        # Create a mapping that compresses features from the main layer by taking the means of a subset of them.
        # This is needed to calculate the gradient of MSE-loss for all sass features at the same time.
        self.output_to_sass_mean_compression = torch.zeros(out_features, sass_features, requires_grad=False)
        main_per_sass = out_features / sass_features
        for i in range(out_features):
            j = int(i / main_per_sass)
            self.output_to_sass_mean_compression[i,j] = 1.0 / main_per_sass
        # Create mappings that combine output_main and output_sass into a single tensor
        self.output_combiner_main = torch.zeros(out_features, out_features + sass_features, requires_grad=False)
        self.output_combiner_sass = torch.zeros(sass_features, out_features + sass_features, requires_grad=False)
        for i in range(out_features):
            self.output_combiner_main[i,i] = 1.0
        for i in range(sass_features):
            self.output_combiner_sass[i,out_features+i] = 1.0

    def reset_parameters(self):
        # main
        nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_main)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_main, -bound, bound)
        # sass
        nn.init.kaiming_uniform_(self.weight_sass, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_sass)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_sass, -bound, bound)

    def forward(self, input):
        output_main, output_sass = SelfAssessmentFunction.apply(input, self.weight_main, self.bias_main, self.weight_sass, self.bias_sass, self.output_to_sass_mean_compression)
        if self.add_sass_features_to_output:
            combined_output = output_main.mm(self.output_combiner_main) + output_sass.mm(self.output_combiner_sass)
            return combined_output
        else:
            return output_main, output_sass

    def extra_repr(self):
        return 'in_features={}, out_features={}, sass_features={}'.format(
            self.in_features, self.out_features, self.sass_features
        )
