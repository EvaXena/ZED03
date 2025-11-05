#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "ap_int.h"

#define IN_WIDTH 600
#define IN_HIGHT 600
#define OUT_WIDTH 50
#define OUT_HEIGHT 50
#define SCALE_FACTOR 12
#define POOL_SIZE (SCALE_FACTOR * SCALE_FACTOR)

typedef ap_axiu<48,1,1,1> video_pixel;
typedef ap_uint<32> sum_t;

void avg_pool_downscaler(hls::stream<video_pixel>& in_stream,hls::stream<video_pixel>& out_stream)
{
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream

    sum_t sum_buffer_r[OUT_WIDTH];
    sum_t sum_buffer_g[OUT_WIDTH];
    sum_t sum_buffer_b[OUT_WIDTH];

    #pragma HLS ARRAY_PARTITION variable=sum_buffer_r complete
    #pragma HLS ARRAY_PARTITION variable=sum_buffer_g complete
    #pragma HLS ARRAY_PARTITION variable=sum_buffer_b complete

    for (int i=0; i < OUT_WIDTH;++i)
    {
        #pragma HLS UNROLL
        sum_buffer_r[i] = 0;
        sum_buffer_g[i] = 0;
        sum_buffer_b[i] = 0;
    }

    for (int y = 0;y < IN_HEIGHT;++y)
    {
        for (int out_x = 0;out_x < OUT_WIDTH;++out_x)
        {
            sum_t h_sum_r = 0,h_sum_g = 0,h_sum_b = 0;
            for(int x_in_block = 0;x_in_block < SCALE_FACTOR;++x_in_block)
            {
                #pragma HLS PIPELINE II=1
                video_pixel current_pixel = in_stream.read();
                ap_uint<16> r = current_pixel.data(47,32);
                ap_uint<16> g = current_pixel.data(31,16);
                ap_uint<16> b = current_pixel.data(15,0);

                h_sum_r += r;
                h_sum_g += g;
                h_sum_b += b;
            }
            sum_buffer_r[out_x] += h_sum_r;
            sum_buffer_g[out_x] += h_sum_g;
            sum_buffer_b[out_x] += h_sum_b;
        }
        if ((y + 1) % SCALE_FACTOR == 0){
            for (int i = 0;i < OUT_WIDTH;++i)
            {
                ap_uint<16> avg_r = sum_buffer_r[i] / POOL_SIZE;
                ap_uint<16> avg_g = sum_buffer_g[i] / POOL_SIZE;
                ap_uint<16> avg_b = sum_buffer_b[i] / POOL_SIZE;

                video_pixel out_pixel;
                out_pixel.data(47,32) = avg_r;
                out_pixel.data(31,16) = avg_g;
                out_pixel.data(15,0) = avg_b;
                out_pixel.last = ((y == IN_HEIGHT - 1) && (i == OUT_WIDTH - 1));
                out_pixel.keep = -1;

                out_stream.write(out_pixel);

                sum_buffer_r[i] = 0;
                sum_buffer_g[i] = 0;
                sum_buffer_b[i] = 0;

            }
        }
    }


}

