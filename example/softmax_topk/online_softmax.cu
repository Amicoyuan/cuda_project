struct __align__(8) MD{
    float m; // 最大值
    float d; // 归一值
}


__device__ MD reduce_md_op(MD a, MD b){

    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD small_m = a_bigger ? a : b;
    MD res;
    
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;

    return res;
}


__global__ void online_softmax(const float * __restrict x, float * __restrict y, int V){



    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;


    // reposition x and y to data for the current vector
    int x += thread_id * V;
    int y += vector_id * V;


    __shared__ MD md_total;

    MD md_patial;
    md_patial.m = -FLT_MAX;
    md_patial.d = 0.0F;


    // 求 max and 归一值
    for(int elem_id = thread_id; elem_id < V; elem_id++){
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0f;
        md_patial = reduce_md_op(new_elem,md_patial);
    }


    // 要么直接掉cub库里面的blockReuce,要么自己用  __shfl_xor_sync() 手搓 BlockReduce
    MD md ;// blockReduce 最终的max与归一值
    
    if(thread_id == 0){
        md_total = md;
    }
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F,md_total.d);

    for(int elem_id = thread_id ; elem_id < V; i++){
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
    }
}