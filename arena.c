#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i8 b8; // booleans
typedef i32 b32;

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) <<20)
#define GiB(n) ((u64)(n) <<30)

#define MIN(a,b) (((a) < (b)) ?(a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#define ALIGN_UP_POW2(n,p) (((u64)(n) + ((u64)(p) -1)) & (~((u64)(p) -1 )))
#define ARENA_BASE_POS (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void*)) // roughly 8bytes


typedef struct{
    u64 capacity; u64 pos;
} mem_arena;


// ------- function prototypes --------- //

mem_arena* arena_create(u64 capacity);
void arena_destroy(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size, b32 non_zero);
void arena_pop(mem_arena* arena, u64 size);
void arena_pop_to(mem_arena* arena, u64 pos);
void arena_clear(mem_arena* arena);

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T)*(n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T)*(n), true)

u32 plat_get_pagesize(void)
// plot = platform
void* plat_mem_reserve(u64 size); //reserves size bites in virtual address space
b32 plat_mem_commit(void* ptr, u64 size); // boolean indicating success, commits this portion
b32 plat_mem_decommit(void*ptr, u64 size); // decommits this portion
b32 plat_mem_release(void* ptr, u64 size);




// -------- main ---------------

int main(void){

    // foo* bar = PUSH_STRUCT(perm_arena, foo)

    mem_arena* perm_arena = arena_create(MiB(1));

    arena_destroy(perm_arena);


    return 0;
}



mem_arena* arena_create(u64 capacity){
    mem_arena* arena = (mem_arena*)malloc(capacity);
    arena->capacity = capacity;
    arena->pos = ARENA_BASE_POS;
    return arena;

}


void arena_destroy(mem_arena* arena){
    free(arena);
}


// nb -> nonzero is performance optimisation, if you are not
// immediate rewriting the code, then set to false, otherwise true
void* arena_push(mem_arena* arena, u64 size, b32 non_zero){
    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    u64 new_pos = pos_aligned + size;

    if(new_pos> arena->capacity) {return NULL;} //or just crash

    arena->pos = new_pos;
    u8* out = (u8*)arena + pos_aligned; // arena pointer is start of memory

    if(!non_zero){
        memset(out, 0, size);
    }
    return out;

}  // return pointer to current pos, increment pointer



//ensures we dont pop from outside the arena's bounds
// size is how much data we dont care about anymore

void arena_pop(mem_arena* arena, u64 size){
    size = MIN(size, arena->pos - ARENA_BASE_POS);
    arena->pos -= size;

}

// moves the arena back to a set index
void arena_pop_to(mem_arena* arena, u64 pos){
    // if/else statement. There is nothing past arena pos
    u64 size = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, size);
}

void arena_clear(mem_arena* arena){
    arena_pop_to(arena, ARENA_BASE_POS);
}
