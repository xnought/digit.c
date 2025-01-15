#include <stdio.h>
#include <stdlib.h>

#define DIMS_MAX 10

typedef struct tensor
{
	int shape[DIMS_MAX];
} tensor;

void tensor_print_shape(tensor *t)
{
	printf("Shape: ( ");
	for (int i = 0; i < DIMS_MAX; i++)
	{
		if (t->shape[i] != 0)
			printf("%d ", t->shape[i]);
	}
	printf(")");
}
void tensor_print(tensor *t)
{
	tensor_print_shape(t);
}

void linear_regression_example()
{
	printf("Linear Regression Example.\n");

	printf("1. Define the dataset to do lin reg on.\n");
	// Shaped (N points, D dimension). In this case D = 1. So just a vector.
	tensor x = {.shape = {100, 1}};
	tensor_print(&x);
}

int main()
{
	linear_regression_example();
	return 0;
}
