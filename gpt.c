#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define DIMS_MAX 3
#define OPS_ARGS_MAX 2

typedef struct tensor
{
	int shape[DIMS_MAX];
	float *data; // points to the 1d points of data
	float *grad;
	struct tensor *ops_args[OPS_ARGS_MAX]; // tensors used in an ops function args
} tensor;

void tensor_print_shape(tensor *t)
{
	printf("Shape:\t");
	for (int i = 0; i < DIMS_MAX; i++)
	{
		if (t->shape[i] != 0)
			printf("%d ", t->shape[i]);
	}
	printf("\n");
}
int tensor_flat_length(tensor *t)
{
	int total = 1;
	for (int i = 0; i < DIMS_MAX; i++)
	{
		if (t->shape[i] != 0)
			total *= t->shape[i];
	}
	return total;
}
void tensor_print_data(tensor *t)
{
	printf("Data:\t");
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		printf("%.2f ", t->data[i]);
	}
	printf("\n");
}
void tensor_print_grad(tensor *t)
{
	printf("Grad:\t");
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		printf("%.2f ", t->grad[i]);
	}
	printf("\n");
}
#define tensor_print(tensor_pointer) (                    \
	{                                                     \
		printf("=====Tensor=====\n");                     \
		printf("Variable Name: '" #tensor_pointer "'\n"); \
		tensor_print_shape(tensor_pointer);               \
		tensor_print_data(tensor_pointer);                \
		tensor_print_grad(tensor_pointer);                \
		printf("================\n\n");                   \
	})

float *tensor_malloc_data(int flat_length)
{
	float *d = malloc(flat_length * sizeof(float));
	return d;
}
tensor *tensor_malloc()
{
	tensor *t = malloc(sizeof(tensor));
	return t;
}
void tensor_free(tensor *t)
{
	free(t->data);
	free(t->grad);
	free(t);
}
tensor *tensor_zeros(int shape[DIMS_MAX])
{
	tensor *t = tensor_malloc();
	for (int i = 0; i < DIMS_MAX; i++)
	{
		t->shape[i] = shape[i];
	}
	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		t->ops_args[i] = NULL;
	}
	int flat_length = tensor_flat_length(t);
	t->data = tensor_malloc_data(flat_length);
	t->grad = tensor_malloc_data(flat_length);
	for (int i = 0; i < flat_length; i++)
	{
		t->grad[i] = 0.0;
		t->data[i] = 0.0;
	}
	return t;
}
tensor *tensor_arange(float start, float stop, float step)
{
	float flat_length = (int)((stop - start) / step) + 1;
	tensor *t = tensor_zeros((int[DIMS_MAX]){flat_length, 1});
	for (int i = 0; i < flat_length; i++)
	{
		t->data[i] = start + i * step;
	}
	return t;
}
tensor *tensor_ones(int shape[DIMS_MAX])
{
	tensor *t = tensor_zeros(shape);
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		t->data[i] = 1.0;
	}
	return t;
}

void tensor_assert_same_shape(tensor *a, tensor *b)
{
	for (int i = 0; i < DIMS_MAX; i++)
	{
		assert(a->shape[i] == b->shape[i]);
	}
}

tensor **tensor_malloc_ops_args(int num_ops)
{
	return malloc(num_ops * sizeof(tensor *));
}

tensor *ops_sum(tensor *t)
{
	tensor *output = tensor_zeros((int[DIMS_MAX]){1, 1});
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		output->data[0] += t->data[i]; // forward
		t->grad[i] = 1.0;			   // backward
	}

	output->ops_args[0] = t;

	return output;
}

tensor *ops_add(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);
	tensor *output = tensor_zeros(a->shape);
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		output->data[i] = a->data[i] + b->data[i]; // forward
		a->grad[i] = 1.0;						   // backward
		b->grad[i] = 1.0;						   // backward
	}

	output->ops_args[0] = a;
	output->ops_args[1] = b;

	return output;
}

tensor *ops_square(tensor *t)
{
	tensor *output = tensor_zeros(t->shape);
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		float t_i = t->data[i];
		output->data[i] = t_i * t_i; // forward
		t->grad[i] = 2 * t_i;		 // backward
	}

	output->ops_args[0] = t;

	return output;
}

tensor *ops_sub(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);
	tensor *output = tensor_zeros(a->shape);
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		output->data[i] = a->data[i] - b->data[i]; // forward
		a->grad[i] = 1.0;						   // backward
		b->grad[i] = -1.0;						   // backward
	}

	output->ops_args[0] = a;
	output->ops_args[1] = b;

	return output;
}

tensor *loss_mse(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);

	tensor *sub = ops_sub(a, b);
	tensor *sqr = ops_square(sub);
	tensor *sum = ops_sum(sqr);

	return sum;
}

void tensor_zero_grad(tensor *t)
{
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		t->grad[i] = 0.0;
	}
}

void graph_zero_grad(tensor *node)
{
	tensor_zero_grad(node);
	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		tensor *op = node->ops_args[i];
		if (op != NULL)
		{
			graph_zero_grad(op);
		}
	}
}
void graph_free(tensor *node)
{
	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		tensor *op = node->ops_args[i];
		if (op != NULL)
		{
			graph_free(op);
		}
	}
	tensor_free(node);
}

void graph_backprop(tensor *node)
{
	assert(false); // todo
	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		tensor *op = node->ops_args[i];
		if (op != NULL)
		{
			graph_backprop(op);
		}
	}
}

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	// tensor *x = tensor_arange(0, 5, 1);
	tensor *y = tensor_arange(0, 5, 1);
	tensor *y_hat = tensor_ones(y->shape);

	tensor *loss = loss_mse(y, y_hat);
	graph_backprop(loss);
	graph_free(loss);
}

int main()
{
	linear_regression_example();
	return 0;
}
