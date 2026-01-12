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


// where it sits in the enum determines how many inputs, how clever!
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0: ((op) <_MV_OP_BINARY_START ? 1 :2)) 
#define MODEL_VAR_MAX_INPUTS 2

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
    u32 rows, u32 cols, u32 flags, model_var_op op
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


b32 mat_relu(matrix* out, const matrix* in); //rectified linear unit, 0 if negative, equal if pos
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q); // this is our cost funtion
// takes in some expected probability distribution p and expected distribution q
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);

void draw_mnist_digit(f32* data);



int main(int argc, char* argv[]) {

    mem_arena* perm_arena = arena_create(GiB(1), GiB(1));  


    matrix* train_images = mat_load(perm_arena, 60000, 784, "train_images.mat");
    matrix* test_images = mat_load(perm_arena, 60000, 784, "test_images.mat");
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
    draw_mnist_digit(train_images->data);
    for(u32 i=0; i<10; i++){
        printf("%.0f", train_labels->data[i]);
    }

    
    arena_destroy(perm_arena);

    return 0;
}




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

    u64 size = sizeof(f32) * (u64)a->rows * a->cols;


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


b32 mat_softmax(matrix* out, const matrix* in){
    // o_i = e^a_i . sum(e^a_i)

    u64 size = (u64)out->rows * out->cols;

    f32 sum = 0.0f;
    for (u64 i =0; i< size; i++){
        out->data[i]=  expf(out->data[i]);
        sum += out->data[i];
    }
    mat_scale(out, 1.0f/sum);
    return true;
}


/* loss funtion */
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q){
    // size checks
    if(p->rows != q->rows || p->cols != q->cols){ return false; }
    else if (out->rows != p->rows || out->cols!=p->cols){ return false; }
    // p times -log (q) ??

    u64 size = (u64)out->rows * out->cols;
    for (u64 i=0; i<size; i++){
        // if p->data is 0, set out to 0, else set it to p->data * log(q->data)
        out->data[i] = p->data[i] == 0.0f
            ? 0.0f : p->data[i] * -logf(q->data[i]);
    }
    return true;
}


b32 mat_relu_add_grad(matrix* out, const matrix* in){
    return true;
}


b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);


b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);

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
    u32 rows, u32 cols, u32 flags, model_var_op op
){
    model_var* out= PUSH_STRUCT(arena, model_var);
    out->index  = model->num_vars ++;
    out->flags = flags;
    out->val = mat_create(arena, rows, cols); // our value vector
    out->op = op; 
    if (flags & MV_FLAG_REQUIRES_GRAD){
        out->grad = mat_create(arena, rows, cols); // our gradient vector
    }
    if(flags & MV_FLAG_INPUT){model->input = out;} // stores the input for manager

    if(flags & MV_FLAG_INPUT){model->input = out;} // stores the input for manager
    if(flags & MV_FLAG_OUTPUT){model->output = out;} // stores the input for manager
    if(flags & MV_FLAG_DESIRED_OUTPUT){model->desired_output = out;} // stores the input for manager
    if(flags & MV_FLAG_COST){model->cost = out;} // stores the input for manager

    return out;
}

/* For unary, create a model variable, using arena model input flags and op, 
then set the new model var's input to the input. Set gradients correctly */
model_var* _mv_unary_impl(mem_arena* arena, model_context* model,
                          model_var* input, u32 flags, model_var_op op,
                          u32 rows, u32 cols){
    if (input->flags & MV_FLAG_REQUIRES_GRAD){
        flags |= MV_FLAG_REQUIRES_GRAD; // also require gradient for future
    }
    model_var* out = mv_create(arena, model, rows, cols, flags, op);
    out->inputs[0] = input; // only one input here
    return out;
}


model_var* _mv_binary_impl(mem_arena* arena, model_context* model,
                          model_var* a, model_var* b, u32 flags, model_var_op op,
                          u32 rows, u32 cols){
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD))
    {flags |= MV_FLAG_REQUIRES_GRAD;} // also require gradient for future

    model_var* out = mv_create(arena, model, rows, cols, flags, op);
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

