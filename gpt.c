#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define SHAPE_MAX 2
#define OPS_ARGS_MAX 2
#define t_shape int[SHAPE_MAX]

typedef enum
{
	NO_OP,
	SUM,
	SQUARE,
	SUB,
	MATMUL,
} op_t;

typedef struct tensor
{
	int shape[SHAPE_MAX];
	float *data; // points to the 1d points of data
	float *grad;
	struct tensor *ops_args[OPS_ARGS_MAX]; // tensors used in an ops function args
	op_t op;
	int transposed;
} tensor;

void tensor_print_op(tensor *t)
{
	printf("Op:\t");

	switch (t->op)
	{
	case MATMUL:
		printf("MATMUL");
		break;
	case SUB:
		printf("SUB");
		break;
	case SQUARE:
		printf("SQUARE");
		break;
	case SUM:
		printf("SUM");
		break;
	case NO_OP:
		printf("NO_OP");
		break;
	default:
		printf("ERROR");
		break;
	}

	printf("\n");
}
void tensor_print_shape(tensor *t)
{
	printf("Shape:\t");
	for (int i = 0; i < SHAPE_MAX; i++)
	{
		if (t->shape[i] != 0)
			printf("%d ", t->shape[i]);
	}
	printf("\n");
}
int tensor_flat_length(tensor *t)
{
	int total = 1;
	for (int i = 0; i < SHAPE_MAX; i++)
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
int tensor_index2d(tensor *t, int i, int j)
{
	return t->transposed ? j * t->shape[0] + i : i * t->shape[1] + j;
}
void tensor_print2d_grad(tensor *m)
{
	printf("Grad:\n");
	for (int i = 0; i < m->shape[0]; i++)
	{
		printf("\t");
		for (int j = 0; j < m->shape[1]; j++)
		{
			printf("%0.2f ", m->grad[tensor_index2d(m, i, j)]);
		}
		printf("\n");
	}
	printf("\n");
}
void tensor_print2d_data(tensor *m)
{
	printf("Data:\n");
	for (int i = 0; i < m->shape[0]; i++)
	{
		printf("\t");
		for (int j = 0; j < m->shape[1]; j++)
		{
			printf("%0.2f ", m->data[tensor_index2d(m, i, j)]);
		}
		printf("\n");
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
		tensor_print_op(tensor_pointer);                  \
		printf("================\n\n");                   \
	})

#define tensor_print2d(tensor_pointer) (                  \
	{                                                     \
		printf("=====Tensor=====\n");                     \
		printf("Variable Name: '" #tensor_pointer "'\n"); \
		tensor_print_shape(tensor_pointer);               \
		tensor_print2d_data(tensor_pointer);              \
		tensor_print2d_grad(tensor_pointer);              \
		tensor_print_op(tensor_pointer);                  \
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
	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		t->ops_args[i] = NULL;
	}
	t->transposed = 0;
	t->op = NO_OP;
	return t;
}
void tensor_free(tensor *t)
{
	free(t->data);
	free(t->grad);
	free(t);
}
tensor *tensor_zeros(int shape[SHAPE_MAX])
{
	tensor *t = tensor_malloc();
	for (int i = 0; i < SHAPE_MAX; i++)
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
	float flat_length = (int)((stop - start) / step);
	tensor *t = tensor_zeros((t_shape){flat_length, 1});
	for (int i = 0; i < flat_length; i++)
	{
		t->data[i] = start + i * step;
	}
	return t;
}
tensor *tensor_ones(int shape[SHAPE_MAX])
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
	for (int i = 0; i < SHAPE_MAX; i++)
	{
		assert(a->shape[i] == b->shape[i]);
	}
}
tensor **tensor_malloc_ops_args(int num_ops)
{
	return malloc(num_ops * sizeof(tensor *));
}
void tensor_transpose(tensor *t)
{
	t->transposed = 1;
	int temp = t->shape[1];
	t->shape[1] = t->shape[0];
	t->shape[0] = temp;
}

tensor *ops_sum(tensor *t)
{
	tensor *output = tensor_zeros((t_shape){1, 1});
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		output->data[0] += t->data[i];
	}

	output->op = SUM;
	output->ops_args[0] = t;

	return output;
}
tensor *ops_add(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);
	tensor *output = tensor_zeros(a->shape);
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		output->data[i] = a->data[i] + b->data[i];
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
		output->data[i] = t_i * t_i;
	}

	output->op = SQUARE;
	output->ops_args[0] = t;

	return output;
}
tensor *ops_sub(tensor *a, tensor *b)
{
	tensor_assert_same_shape(a, b);
	tensor *output = tensor_zeros(a->shape);
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		output->data[i] = a->data[i] - b->data[i];
	}

	output->op = SUB;
	output->ops_args[0] = a;
	output->ops_args[1] = b;

	return output;
}
tensor *ops_matmul(tensor *a, tensor *b)
{
	assert(a->shape[1] == b->shape[0]);

	tensor *output = tensor_zeros((t_shape){a->shape[0], b->shape[1]});
	for (int i = 0; i < a->shape[0]; i++)
	{
		for (int j = 0; j < b->shape[1]; j++)
		{
			for (int k = 0; k < a->shape[1]; k++)
			{
				output->data[tensor_index2d(output, i, j)] += a->data[tensor_index2d(a, i, k)] * b->data[tensor_index2d(b, k, j)];
			}
		}
	}

	output->op = MATMUL;
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

void ops_sum_backprop(tensor *output)
{
	tensor *a = output->ops_args[0];
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		a->grad[i] = 1.0;			   // local
		a->grad[i] *= output->grad[0]; // chain rule
	}
}
void ops_square_backprop(tensor *output)
{
	tensor *a = output->ops_args[0];
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		a->grad[i] = 2 * a->data[i];   // local
		a->grad[i] *= output->grad[i]; // chain rule
	}
}
void ops_sub_backprop(tensor *output)
{
	tensor *a = output->ops_args[0];
	tensor *b = output->ops_args[1];
	for (int i = 0; i < tensor_flat_length(a); i++)
	{
		a->grad[i] = 1.0;			   // local
		b->grad[i] = -1.0;			   // local
		a->grad[i] *= output->grad[i]; // chain rule
		b->grad[i] *= output->grad[i]; // chain rule
	}
}

void ops_matmul_backprop(tensor *output)
{
	tensor *a = output->ops_args[0];
	tensor *b = output->ops_args[1];
}

void graph_backprop(tensor *output)
{
	switch (output->op)
	{
	case MATMUL:
		ops_matmul_backprop(output);
		break;
	case SUB:
		ops_sub_backprop(output);
		break;
	case SQUARE:
		ops_square_backprop(output);
		break;
	case SUM:
		output->grad[0] = 1.0; // reduce grad
		ops_sum_backprop(output);
		break;
	case NO_OP:
		return;
	default:
		exit(1);
		break;
	}

	for (int i = 0; i < OPS_ARGS_MAX; i++)
	{
		tensor *child = output->ops_args[i];
		if (child != NULL)
		{
			graph_backprop(child);
		}
	}
}

float rand_0_1()
{
	return (float)rand() / (float)RAND_MAX;
}

float rand_between(float a, float b)
{
	return a + rand_0_1() * (b - a);
}
void tensor_seed_random(int seed)
{
	srand(seed);
}
tensor *tensor_random(float a, float b, int shape[SHAPE_MAX])
{
	tensor *t = tensor_zeros(shape);
	for (int i = 0; i < tensor_flat_length(t); i++)
	{
		t->data[i] = rand_between(a, b);
	}
	return t;
}

void linear_regression_example()
{
	tensor_seed_random(0);

	// tensor *x = tensor_arange(0, 6, 1);				   // (N, d)
	tensor *y = tensor_arange(0, 12, 1); // (N, 1)
	y->shape[0] = 6;
	y->shape[1] = 2;
	tensor_print2d(y);
	// tensor *w = tensor_random(-1, 1, (t_shape){1, 1}); // (d, 1)
	tensor *w = tensor_ones((t_shape){y->shape[1], 1}); // (d, 1)
	tensor_print2d(w);

	// tensor *w = tensor_ones((t_shape){1, 1}); // (d, 1)
	// tensor *yhat = ops_matmul(x, w);		  // (N, 1)
	// tensor *loss = loss_mse(y, yhat);
	// graph_backprop(loss);
	// tensor_print(loss);
	// tensor_print(yhat);
	// tensor_print(y);
}

int main()
{
	linear_regression_example();
	return 0;
}
