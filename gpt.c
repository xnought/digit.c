#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define DIMS_MAX 3

typedef struct tensor
{
	int shape[DIMS_MAX];
	float *data; // points to the 1d points of data
	float *grad;
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

tensor *ops_sum(tensor *t)
{
	tensor *sum = tensor_zeros((int[DIMS_MAX]){1, 1});
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		sum->data[0] += t->data[i]; // forward
		t->grad[i] = 1.0;			// backward
	}

	return sum;
}

void tensor_assert_same_shape(tensor *a, tensor *b)
{
	for (int i = 0; i < DIMS_MAX; i++)
	{
		assert(a->shape[i] == b->shape[i]);
	}
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

	return output;
}

tensor *loss_mse(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);

	tensor *sub = ops_sub(a, b);
	tensor *sqr = ops_square(sub);
	tensor *sum = ops_sum(sqr);

	tensor_free(sub);
	tensor_free(sqr);

	return sum;
}

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	tensor *x = tensor_arange(0, 5, 1);
	tensor *y = tensor_arange(0, 5, 1);
	tensor *y_hat = tensor_ones(y->shape);

	tensor_print(x);
	tensor_print(y);
	tensor_print(y_hat);

	tensor *loss = loss_mse(y, y_hat);
	tensor_print(loss);

	tensor_free(x);
	tensor_free(y);
	tensor_free(y_hat);
}

int main()
{
	linear_regression_example();
	return 0;
}
