import torch as th


def task1():
    # Start with a 13x13 matrix of ones
    matrix = th.full(size=(13, 13), fill_value=1, dtype=th.int8)

    # Fill in the columns and rows with twos
    matrix[[1, 6, -2], :] = 2
    matrix[:, [1, 6, -2]] = 2

    # Fill in the left threes
    matrix[3:5, 3:5] = 3
    matrix[-5:-3, 3:5] = 3

    # Fill in the right threes
    matrix[3:5, -5:-3] = 3
    matrix[-5:-3, -5:-3] = 3

    # Print out the matrix
    print(matrix)


def sigmoid(x):
    return 1.0 / (1.0 + th.exp(-x))


def softmax(x):
    return th.exp(x) / th.sum(th.exp(x))


def task3():
    # Generate a random vector
    random_vector = th.randint(low=-5, high=5, size=(5, ), dtype=th.float32)

    # Compare the outputs
    print(f'Inputs: {random_vector}')
    print(f'Sigmoid: {sigmoid(random_vector)}')
    print(f'Softmax: {softmax(random_vector)}')


if __name__ == '__main__':
    task1()
    task3()
