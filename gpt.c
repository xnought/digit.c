#include <stdio.h>
#include <stdlib.h>

typedef struct tensor
{
	int data;
} tensor;

int main()
{
	printf("Hello GPT!\n");
	tensor a = {.data = 1};
	printf("(%d)\n", a.data);
	return 0;
}
