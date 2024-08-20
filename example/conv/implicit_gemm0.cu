




void launch_param(param_t param)
{
    unsigned int n = param.n;
    unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;


    int outh = (h-r + 2 * p) / u +1;
    int outw = (w-s + 2 * q) / v+ 1;

    param.Oh = outh;
    param.Ow = outw;


    int blockx = ((outh * outw +16 -1) /16);
    int blocky = (k+ 16-1)/16;
    int blockz = n;
    int threadx = 16;
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx,thready,threadz);
    dim3 grid(blockx,blocky,blockz);
}