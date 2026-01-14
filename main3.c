#include <math.h>
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include "base.h"
#include "prng.h"
#include "arena.h"

#include "arena.c"
#include "prng.c"


typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;

typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD  = (1<<0),
    MV_FLAG_PARAMETER      = (1<<1),
    MV_FLAG_INPUT          = (1<<2),
    MV_FLAG_OUTPUT         = (1<<3),
    MV_FLAG_DESIRED_OUTPUT = (1<<4),
    MV_FLAG_COST           = (1<<5),
} model_var_flags;

typedef enum {
    MV_OP_NULL=0,
    MV_OP_CREATE,
    _MV_OP_UNARY_START, // this tells us how many ops involved for gradients

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,
    
    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op; // type of operations, how the variables are created

typedef struct{
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;
    
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;

} model_training_desc;





// where it sits in the enum determines how many inputs, how clever!
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0: ((op) <_MV_OP_BINARY_START ? 1 :2)) 
#define MODEL_VAR_MAX_INPUTS 2
#define MNIST_PIXELS 784


typedef struct model_var{
    u32 index;  // for indexing operations
    u32 flags; // things that modify function
    matrix* val; //matrix for its value
    matrix* grad; //matrix for its gradient
    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS]; //contains a pointer to another model var? or its own pointer?
} model_var;



typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {

    u32 num_vars;

    model_var* input;
    model_var* output;

    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;

} model_context;



model_var* _mv_unary_impl(mem_arena* arena, model_context* model,
                          model_var* input, u32 flags, model_var_op op,
                          u32 rows, u32 cols);

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);


model_var* mv_softmax(

    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

model_var*  mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
);

model_var*  mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
);
model_var*  mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
);
model_var*  mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
);






matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);


void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_clear(matrix* mat);
b32 mat_copy(matrix*dst, matrix* src);
void mat_fill(matrix* mat, f32 x);
void mat_scale(matrix* mat, f32 scale);

matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
b32 mat_add(matrix* out, const matrix*a, const matrix* b); // boolean on success
f32 mat_sum(matrix* mat); // sums all entries in the matrix
b32 mat_sub(matrix* out, const matrix*a, const matrix* b); // boolean on success

b32 mat_mul(matrix* out, const matrix*a, const matrix* b,
            b8 zero_out, b8 transpose_a, b8 transpose_b); // boolean on success
u64 mat_argmax(matrix* mat);

b32 mat_relu(matrix* out, const matrix* in); //rectified linear unit, 0 if negative, equal if pos
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q); // this is our cost funtion
// takes in some expected probability distribution p and expected distribution q



b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad);

b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,      // Outputs: Where to accumulate the gradients
    const matrix* p, const matrix* q,    // Inputs: The original values
    const matrix* grad                   // Upstream: The gradient from the cost node
);

void draw_mnist_digit(f32* data);

//out var is who specifies output
model_program model_prog_create(mem_arena* arena, model_context* model,
                                model_var* out_var);

void model_prog_compute(model_program* program);
void model_prog_compute_grads(model_program* program);


model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
void model_feedforward(model_context* model);
void model_train(model_context* model,
                 const model_training_desc* training_desc);
void create_mnist_model(mem_arena* arena, model_context* model);













/*  --------------------------- MAIN ----------------------- */



int main(int argc, char* argv[]) {

    mem_arena* perm_arena = arena_create(MiB(500), MiB(500));  

    matrix* train_images = mat_load(perm_arena, 60000, MNIST_PIXELS, "train_images.mat");
    matrix* test_images = mat_load(perm_arena, 10000, MNIST_PIXELS, "test_images.mat");
    matrix* train_labels = mat_create(perm_arena, 60000, 10);
    matrix* test_labels = mat_create(perm_arena, 10000, 10);

    {
        matrix* train_labels_file = mat_load(perm_arena, 60000, 1, "train_labels.mat");
        matrix* test_labels_file = mat_load(perm_arena, 10000, 1, "test_labels.mat");
         
        // one-hot encoding
        for (u32 i=0; i< 60000; i++){
            u32 num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f; // each row of 10 is only 1 at target val
        }

        for (u32 i=0; i< 10000; i++){
            u32 num = test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f; // each row of 10 is only 1 at target val
        }

    }

    /*draw_mnist_digit(train_images->data);

    printf("Hot encoding of digit: \n");
    for(u32 i=0; i<10; i++){
        printf("%.0f ", train_labels->data[i]);
    }
    printf("\n");
    */


    model_context* model = model_create(perm_arena);

    create_mnist_model(perm_arena, model); // populate all the model variables

    model_compile(perm_arena, model);
    
    memcpy(model->input->val->data, train_images->data, sizeof(f32) * MNIST_PIXELS);

    model_feedforward(model);
    
    printf("pre-training output: \n");
    for (int i =0; i<10; i++){
        printf("%.4f ", model->output->val->data[i]);
    }
    printf("\n");
    
    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = 3,
        .batch_size = 50,
        .learning_rate = 0.01f,
    };
    model_train(model, &training_desc);

    printf("post-training output: \n");


    for (int i =0; i<10; i++){
        printf("%.4f ", model->output->val->data[i]);
    }

    arena_destroy(perm_arena);

    return 0;
}

/*  --------------------------- MAIN ----------------------- */











/* ANSI escape codes */
void draw_mnist_digit(f32* data){
    for(u32 y =0; y< 28; y++){
        for (u32 x =0; x<28; x++){
            f32 num = data[x + y*28];
            u32 col = 232 + (u32)(num*24);
            printf("\x1b[48;5;%dm  ", col); //x1b[ is the escape code
        }
        printf("\n");
    }
    printf("\x1b[0m");
}


void create_mnist_model(mem_arena* arena, model_context* model){
    model_var* input = mv_create(arena, model, MNIST_PIXELS, 1, MV_FLAG_INPUT);

    model_var* W0 = mv_create(arena, model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W1 = mv_create(arena, model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W2 = mv_create(arena, model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    
    f32 bound0 = sqrtf(6.0f / (784 + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16+ 10));

    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b1 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b2 = mv_create(arena, model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);



    model_var* z0_a = mv_matmul(arena, model, W0, input, false);
    model_var* z0_b = mv_add(arena, model, z0_a, b0, false);
    model_var* a0 = mv_relu(arena, model, z0_b, false);



    model_var* z1_a = mv_matmul(arena, model, W1, a0, false);
    model_var* z1_b = mv_add(arena, model, z1_a, b1, false);
    model_var* z1_c = mv_relu(arena, model, z1_b, false); //
    model_var* a1 = mv_add(arena, model, z1_c, a0, false); //  residual connection



    model_var* z2_a = mv_matmul(arena, model, W2, a1, false);
    model_var* z2_b = mv_add(arena, model, z2_a, b2, false);
    model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);



    model_var* y = mv_create(arena, model, 10, 1, MV_FLAG_DESIRED_OUTPUT);
    model_var* cost = mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
}



/*
Takes our memory areana, and then pushes a matrix (with dimensions and an 
array of f32 data) and then pushes an array onto that data of a predetermined shape
*/
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols){
    matrix* mat = PUSH_STRUCT(arena, matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows*cols);
    return mat;
}


/*
uses the memcpy function, but not before checking that the two are compatible to copy between each other
*/
b32 mat_copy(matrix* dst, matrix* src){
    if(dst->rows != src->rows || dst->cols != src->cols){
        return false;
   }
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);
    return true;
}


void mat_clear(matrix* mat){
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}


void mat_fill(matrix* mat, f32 x){
    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i =0; i<size; i++){
        mat->data[i] = x; // assume index i increments by f32?
    }
}


void mat_fill_rand(matrix* mat, f32 lower, f32 upper){
    
    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i =0; i<size; i++){
        mat->data[i] = prng_randf() * (upper - lower) + lower ; // assume index i increments by f32?
    }
}


void mat_scale(matrix* mat, f32 scale){

    u64 size = (u64)mat->rows * mat->cols;
    for(u64 i =0; i<size; i++){
        mat->data[i] *= scale; // assume index i increments by f32?
    }
}

f32 mat_sum(matrix* mat){

    f32 sum = 0.0f;

    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i =0; i<size; i++){
        sum += mat->data[i]; // assume index i increments by f32?
    }
    return sum;
}


u64 mat_argmax(matrix* mat){
    u64 size = (u64)mat->rows * mat->cols;


    f32 max = -INFINITY;
    u64 max_i = 0;
    for (u64 i =0; i< size; i++){
        if (mat->data[i] > mat->data[max_i]){
            max_i = i;
        }
    }
    return max_i;
}




b32 mat_add(matrix* out, const matrix*a, const matrix* b){
    
    if(a->rows != b->rows || a->cols != b->cols){
        return false;
    }
    if(out-> rows != a->rows || out->cols != a->cols){
        return false;
    }
    u64 size = (u64)a->rows * a->cols;


    for(u64 i=0; i<size; i++){
        out->data[i] = (a->data[i] + b->data[i]);
    }


    return true;
} // boolean on success


b32 mat_sub(matrix* out, const matrix*a, const matrix* b){

    
    if(a->rows != b->rows || a->cols != b->cols){
        return false;
    }
    if(out-> rows != a->rows || out->cols != a->cols){
        return false;
    }

    u64 size = (u64)a->rows * a->cols;


    for(u64 i=0; i<size; i++){
        out->data[i] = (a->data[i] - b->data[i]);
    }
    return true;
}


void _mat_mul_nn(matrix* out, const matrix* a, const matrix*b){
    for (u64 i=0; i < out->rows; i++){
        for (u64 k = 0; k<a->cols; k++){
            for (u64 j=0; j < out->cols; j++){
                out->data[j + i * out->cols]  +=// ith column, jth row
                    (a->data[k + i* a->cols] *  // to pick column we just add on an index
                     b->data[j + k * b->cols]); // to pick row we use total cols

            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix*b){
    for (u64 i=0; i < out->rows; i++){
        for (u64 j=0; j < out->cols; j++){
            for (u64 k = 0; k<a->cols; k++){
                out->data[j + i * out->cols]  +=// ith column, jth row
                    (a->data[k + i* a->cols] *  // to pick column we just add on an index
                     b->data[k + j * b->cols]); // to pick row we use total cols

            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix*b){

    for (u64 k = 0; k<a->rows; k++){
        for (u64 i=0; i < out->rows; i++){
            for (u64 j=0; j < out->cols; j++){
                out->data[j + i * out->cols]  +=// ith column, jth row
                    (a->data[i + k* a->cols] *  // to pick column we just add on an index
                     b->data[j + k * b->cols]); // to pick row we use total cols

            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix*b){
    for (u64 i=0; i < out->rows; i++){
       for (u64 j=0; j < out->cols; j++){
            for (u64 k = 0; k<a->rows; k++){
                out->data[j + i * out->cols]  +=// ith column, jth row
                    (a->data[i + k* a->cols] *  // to pick column we just add on an index
                     b->data[k + j * b->cols]); // to pick row we use total cols

            }
        }
    }
}



b32 mat_mul(matrix* out, const matrix*a, const matrix* b,
            b8 zero_out, b8 transpose_a, b8 transpose_b){
        
    // if the transpose argument is true, then takes a_cols otherwise takes a_rows
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;
    
    if (a_cols != b_rows) {return false;}
    if(out->rows != a_rows || out->cols != b_cols) {return false;}
    if (zero_out){
        mat_clear(out);
    }

    // depending on how much transposition we are doing, we call different functions
    u32 transpose = (transpose_a << 1) | transpose_b;
    
    // nice way to create double binary if statements
    switch (transpose){
        case 0b00: { _mat_mul_nn(out, a , b);} break;
        case 0b01: { _mat_mul_nt(out, a , b);} break;
        case 0b10: { _mat_mul_tn(out, a , b);} break;
        case 0b11: { _mat_mul_tt(out, a , b);} break;

    }

    return true;


}


b32 mat_relu(matrix* out, const matrix* in){
    if (out->rows != in->rows || out->cols != in->cols){
        return false;
    }
    u64 size = (u64)out->rows * out->cols;
    for(u64 i =0; i<size; i++){
        out->data[i] = MAX(0, in->data[i]);

    }
    return true;
}


b32 mat_softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    u64 size = (u64)out->rows * out->cols;

    // 1. Find the maximum value in the input vector
    f32 max_val = -INFINITY;
    for (u64 i = 0; i < size; i++) {
        if (in->data[i] > max_val) max_val = in->data[i];
    }

    // 2. Compute exp(x - max) and sum them
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max_val); // Always <= 1.0f
        sum += out->data[i];
    }

    // 3. Normalize
    mat_scale(out, 1.0f / sum);
    return true;
}




b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols || out->rows != p->rows || out->cols != p->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        // Clamp q to avoid log(0) while preserving high penalty
        float q_val = q->data[i] < 1e-7f ? 1e-7f : q->data[i]; 
        out->data[i] = -p->data[i] * logf(q_val);
    }
    return true;
}


b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad){
     if (out->rows != in->rows || out->cols != in->cols){
        return false;
    }

     if (out->rows != grad->rows || out->cols != grad->cols){
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i =0; i<size; i++){
        out->data[i] += in->data[i] >= 0.0f ? grad->data[i] : 0.0f;

    }
    return true;
}



b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out,
                         const matrix* grad){
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    if (softmax_out->rows != 1 && softmax_out->cols!=1){
        return false;
    }
    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    
    matrix* jacobian = mat_create(scratch.arena, size, size);
    for (u32 i=0; i<size; i++){
        for (u32 j =0; j<size; j++){
            // just the softmax defintion (which has kronecker delta)
            jacobian->data[j + i*size] =
                softmax_out->data[i] *((i==j) - softmax_out->data[j]);
        }
    }
    mat_mul(out, jacobian, grad, false, false, false);
    arena_scratch_release(scratch);

    return true;
}

b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,      // Outputs: Where to accumulate the gradients
    const matrix* p, const matrix* q,    // Inputs: The original values
    const matrix* grad){
    if(p->rows != q->rows || p->cols != q->cols){
        return false;
    } 


    u64 size = (u64)p->rows * p->cols;

    if (p_grad != NULL){
        if(p->rows != p_grad->rows || p->cols != p_grad->cols){
            return false;
        }
        for (u64 i =0; i< size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }

        if (q_grad != NULL){

            if(q->rows != q_grad->rows || q->cols != p_grad->cols){
                return false;
            }
            for (u64 i = 0; i< size; i++){    
                q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
            }
        }
    }
    return true;
}



/* worth looking into the apis behind the file reading */
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename){
    matrix* mat = mat_create(arena, rows, cols);
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);
    size = MIN(size, sizeof(f32) * rows * cols);
    fread(mat->data, 1, size, f);
    fclose(f);
    return mat;
}


model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
){
    model_var* out= PUSH_STRUCT(arena, model_var);
    out->index  = model->num_vars ++;
    out->flags = flags;
    out->val = mat_create(arena, rows, cols); // our value vector
    out->op = MV_OP_CREATE; 
    if (flags & MV_FLAG_REQUIRES_GRAD){
        out->grad = mat_create(arena, rows, cols); // our gradient vector
    }

    if(flags & MV_FLAG_INPUT){model->input = out;} // stores the input for manager
    if(flags & MV_FLAG_OUTPUT){model->output = out;} // stores the input for manager
    if(flags & MV_FLAG_DESIRED_OUTPUT){model->desired_output = out;} // stores the input for manager
    if(flags & MV_FLAG_COST){model->cost = out;} // stores the input for manager

    return out;
}

/* For unary, create a model variable, using arena model input flags and op, 
then set the new model var's input to the input. Set gradients correctly */
model_var* _mv_unary_impl(mem_arena* arena, model_context* model,
                          model_var* input,u32 rows, u32 cols, u32 flags, model_var_op op)
                          {
    if (input->flags & MV_FLAG_REQUIRES_GRAD){
        flags |= MV_FLAG_REQUIRES_GRAD; // also require gradient for future
    }
    model_var* out = mv_create(arena, model, rows, cols, flags);
    out->inputs[0] = input; // only one input here
    out-> op = op;
    return out;
}


model_var* _mv_binary_impl(mem_arena* arena, model_context* model,
                          model_var* a, model_var* b, u32 rows, u32 cols, u32 flags, model_var_op op){
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD))
    {flags |= MV_FLAG_REQUIRES_GRAD;} // also require gradient for future

    model_var* out = mv_create(arena, model, rows, cols, flags);
    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;
    return out;
}

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
){
    return _mv_unary_impl(arena, model, input, input->val->rows, input->val->cols,
                   flags, MV_OP_RELU);
}


model_var* mv_softmax(

    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
){
    return _mv_unary_impl(arena, model, input, input->val->rows, input->val->cols,
                   flags, MV_OP_SOFTMAX);
}

model_var*  mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
) {
        if(a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
            return NULL;
        }
        return _mv_binary_impl(arena, model, a, b, a->val->rows, a->val->cols,
                               flags, MV_OP_ADD);
}

model_var*  mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
) {
        if(a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
            return NULL;
        }
        return _mv_binary_impl(arena, model, a, b, a->val->rows, a->val->cols,
                               flags, MV_OP_SUB);
}



model_var*  mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var*b, u32 flags
) {
        if(a->val->cols != b->val->rows){
            return NULL;
        }
        return _mv_binary_impl(arena, model, a, b, a->val->rows, b->val->cols,
                               flags, MV_OP_MATMUL);
}


model_var*  mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var*q, u32 flags


) {
        if(p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
            return NULL;
        }
        return _mv_binary_impl(arena, model, p, q, p->val->rows, p->val->cols,
                               flags, MV_OP_CROSS_ENTROPY);
}


//out var is who specifies output
model_program model_prog_create(mem_arena* arena, model_context* model,
                                model_var* out_var){
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);
    
    u32 stack_size = 0;

    u32 out_size = 0;

    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    stack[stack_size++] = out_var;
    
    while (stack_size > 0){
        model_var* cur = stack[--stack_size];
        if(cur->index >= model->num_vars) {continue;} // not possible?

        if( visited[cur->index]){
            if(out_size < model->num_vars){
                out[out_size++] = cur;
            }

            continue;
        }

        visited[cur->index]= true;

        if (stack_size<model->num_vars){
            stack[stack_size++] = cur; // add it to the stack, like the visualisation
        }
        
        //expand out curr
        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for(u32 i =0; i< num_inputs; i++){
            model_var* input = cur->inputs[i];
            if (input->index >= model->num_vars || visited[input->index]) {
                continue;  // are these if statements due to loops?
            }
            
            for(u32 j=0; j<stack_size;j++){
                if (stack[j] == input){
                    for (u32 k = j; k<stack_size-1;k++){
                        stack[k] = stack[k+1];
                    }
                    stack_size --;
                }
            }


            stack[stack_size ++ ] = input;
        }
    }

    arena_scratch_release(scratch);
    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size),
    };
    memcpy(prog.vars, out, sizeof(model_var*) * out_size);
    return prog;

}

/* executes our program in the forward direction */
void model_prog_compute(model_program* prog){
    for (u32 i =0; i< prog->size; i++){
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch (cur->op){
            case MV_OP_NULL:
            case MV_OP_CREATE: break;
            case _MV_OP_UNARY_START: break;
            case MV_OP_RELU: {mat_relu(cur->val, a->val);}break;
            case MV_OP_SOFTMAX: {mat_softmax(cur->val, a->val);}break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: { mat_add(cur->val, a->val, b->val);} break;
            case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val);} break;
            case MV_OP_MATMUL: { mat_mul(cur->val, a->val, b->val,
                                         true, false, false);} break;
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            } break;

        }

    }
}


void model_prog_compute_grads(model_program* prog){
    for (u32 i=0; i<prog->size; i++){
        // iterate through program and zero out grads if we are computing
        model_var* cur = prog->vars[i];
        if((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD){
            continue; // cur->flags & ... is a bitmask, checks if grad in flags
        }

        if(cur->flags & MV_FLAG_PARAMETER){
            continue; // don't compute if not neeeded
        } 
        mat_clear(cur->grad);

    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);
    for (i64 i = (i64)prog->size-1; i>=0; i--){
    
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        // if we only have one input, that doesn't need grad, skip
        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (num_inputs == 1 && (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD){
            continue;
        } 

        if (num_inputs == 2 && 
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD){
            continue;
        }

        switch (cur->op){
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;
            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;
            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if(a->flags & MV_FLAG_REQUIRES_GRAD){
                    mat_add(a->grad, a->grad, cur->grad);
                } // our gradient doesnt change, so we 

                if(b->flags & MV_FLAG_REQUIRES_GRAD){
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;


            case MV_OP_SUB: {

                if(a->flags & MV_FLAG_REQUIRES_GRAD){
                    mat_add(a->grad, a->grad, cur->grad);
                } // our gradient doesnt change, so we 

                if(b->flags & MV_FLAG_REQUIRES_GRAD){
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break; 

            // in this case ORDER MATTERS AND A COMES FIRST. A= l*m B =m*n
            // cur = l*n. So to get gradient of A, we need cur * B, but for 
            // compat, B must be traposed. For B, (cur stays) we do A^T * cur

            case MV_OP_MATMUL: {

                if(a->flags & MV_FLAG_REQUIRES_GRAD){
                 // transpose b, just due to nature of chain rule
                    mat_mul(a->grad, cur->grad, b->val, false, false, true);
                }

                if(b->flags & MV_FLAG_REQUIRES_GRAD){
                    mat_mul(b->grad, a->val, cur->grad, false, true, false);
                } 
            } break;


            case MV_OP_CROSS_ENTROPY: {
                model_var* p = a;
                model_var* q = b;
                mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val,
                                           cur->grad);
            } break;
        }
    }
return;

}







model_context* model_create(mem_arena* arena){
    model_context* model = PUSH_STRUCT(arena, model_context);
    return model;
}


void model_compile(mem_arena* arena, model_context* model){
    if(model->output != NULL){
        model->forward_prog = model_prog_create(arena, model, model->output);
    }
    if (model->cost != NULL){
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

void model_feedforward(model_context* model){
    model_prog_compute(&model->forward_prog);
}


/* Stochastic gradient descent:
batch training into batches
find graidents for parameters wrt cost
average out
subtract gradients from model
*/
void model_train(model_context* model,
                 const model_training_desc* training_desc
) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 num_tests = test_images->rows;
    u32 output_size = train_labels->cols;

    u32 num_batches = num_examples / training_desc->batch_size;
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);


    for(u32 i = 0; i< num_examples; i++){
        training_order[i] = i;
    }


    for (u32 epoch = 0; epoch < training_desc->epochs; epoch ++){
        for (u32 i = 0 ; i< num_examples; i++){
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;
            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }
        /* At the start, clear gradients of parameters, end update */
        for (u32 batch = 0; batch<num_batches; batch++){
            /* loop through cost, clear gradients */
            for (u32 i = 0; i< model->cost_prog.size; i++){

                model_var* cur = model->cost_prog.vars[i];
                // We average over a batch 
                if (cur->flags & MV_FLAG_PARAMETER){
                    mat_clear(cur->grad);
                }
            }
            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < training_desc->batch_size ;i++){
                u32 order_index = batch * training_desc->batch_size + i;
                u32 index = training_order[order_index];

                // Copy a single training datapoint into the network
                memcpy(model->input->val->data,
                       train_images->data + index * input_size,
                       sizeof(f32) * input_size
                       );


                memcpy(model->desired_output->val->data,
                       train_labels->data + index * output_size,
                       sizeof(f32) * output_size
                       );

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);

                
            }
            avg_cost /= (f32)training_desc->batch_size;
            for (u32 i = 0; i< model->cost_prog.size; i++){

                model_var* cur = model->cost_prog.vars[i];
                if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER){
                    continue;
                }
                mat_scale(cur->grad,
                          training_desc->learning_rate /
                          training_desc->batch_size
                );
                // gradients point in direction of steepest ascent, so for desc
                mat_sub(cur->val, cur->val, cur->grad);

            }

            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r", 
                epoch + 1, training_desc->epochs, 
                batch + 1, num_batches, avg_cost
            );
            fflush(stdout);
    

        }   
        printf("\n");
        u32 num_correct= 0;
        f32 avg_cost = 0;

        for (u32 i =0; i<num_tests; i++){
             memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size
            );


            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size
            );

            model_prog_compute(&model->cost_prog);
            avg_cost += mat_sum(model->cost->val);
            num_correct += 
                mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val);

        }
        avg_cost /= (f32)num_tests;
        printf(
            "Test Completed. Accuracy %5d / %d (%.1f), Average Cost: %.4f\n", 
            num_correct, num_tests, (f32)num_correct/num_tests * 100.0f, 
            avg_cost
        );
    }

    arena_scratch_release(scratch);
}



