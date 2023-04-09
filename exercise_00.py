import torch as th


def design_matrix():
    # Start with a 7x7 matrix of ones
    m = th.full(size=(7, 7), fill_value=1, dtype=th.int16)

    # Add the 2 rows and columns
    m[[1, 5], :] = 2
    m[:, [1, 5]] = 2

    # Add the three
    m[3, 3] = 3

    # Print out the resulting matrix
    print(m)

    # Replace the surroundings of the 3 with fours
    m[2:5, 2:5] = 4
    m[3, 3] = 3
    print(m)


# Create a Perceptron Class
class Perceptron(th.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.weights = th.nn.Parameter(th.full(size=(input_size + 1,), fill_value=1.0), requires_grad=True)

    def forward(self, x: th.Tensor):
        return th.matmul(th.concat((x, th.full(size=(x.shape[0], 1), fill_value=1.0)), dim=1), self.weights)


def test_perceptron():
    # Generate some random inputs
    inputs = th.randn(size=(5, 3))
    print(inputs)

    # Create a Perceptron and call it with the inputs
    perceptron = Perceptron(input_size=3)
    outputs = perceptron(inputs)
    print(outputs)


if __name__ == '__main__':
    design_matrix()
    test_perceptron()
