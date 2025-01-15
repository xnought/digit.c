#include <stdio.h>
#include <stdlib.h>

#define DIMS_MAX 3

typedef struct tensor
{
	int shape[DIMS_MAX];
	float *data; // points to the 1d points of data
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

#define tensor_print(tensor_pointer) (                    \
	{                                                     \
		printf("=====Tensor=====\n");                     \
		printf("Variable Name: '" #tensor_pointer "'\n"); \
		tensor_print_shape(tensor_pointer);               \
		tensor_print_data(tensor_pointer);                \
		printf("================\n");                     \
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
	for (int i = 0; i < flat_length; i++)
	{
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

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	tensor *x = tensor_arange(0, 10, 1);
	tensor *y = tensor_arange(0, 10, 1);
	tensor *y_hat = tensor_ones(y->shape);

	tensor_print(x);
	tensor_print(y);
	tensor_print(y_hat);

	tensor_free(x);
	tensor_free(y);
	tensor_free(y_hat);
}

int main()
{
	linear_regression_example();
	return 0;
}
