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
void tensor_print(tensor *t)
{
	printf("=====Tensor=====\n");
	tensor_print_shape(t);
	tensor_print_data(t);
	printf("================\n");
}

float *tensor_malloc_data(int flat_length)
{
	float *d = (float *)malloc(sizeof(float) * flat_length);
	return d;
}
tensor *tensor_malloc()
{
	tensor *t = (tensor *)malloc(sizeof(tensor));
	return t;
}
void tensor_free(tensor *t)
{
	free(t->data);
	free(t);
}
tensor *tensor_empty(int shape[DIMS_MAX])
{
	tensor *t = tensor_malloc();
	t->data = tensor_malloc_data(tensor_flat_length(t));
	for (int i = 0; i < DIMS_MAX; i++)
	{
		t->shape[i] = shape[i];
	}
	return t;
}
tensor *tensor_arange(float start, float stop, float step)
{
	float flat_length = (int)((stop - start) / step) + 1;
	tensor *t = tensor_empty((int[DIMS_MAX]){flat_length});
	for (int i = 0; i < flat_length; i++)
	{
		t->data[i] = start + i * step;
	}
	return t;
}

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	tensor *x = tensor_arange(0, 10, 1);
	tensor_print(x);
}

int main()
{
	linear_regression_example();
	return 0;
}
