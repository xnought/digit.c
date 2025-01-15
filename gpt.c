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

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	float d[] = {1.0, 2.0, 3.0};
	tensor x = {.shape = {3}, .data = d};
	tensor_print(&x);
}

int main()
{
	linear_regression_example();
	return 0;
}
