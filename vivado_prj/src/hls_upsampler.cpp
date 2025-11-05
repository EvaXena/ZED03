#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "ap_int.h"

#define IN_WIDTH 50
#define IN_HIGHT 50
#define OUT_WIGHT 600
#define OUT_HIGHT 600
#define SCALE_FACTOR 12

typedef ap_axiu<48,1,1,1> video_pixel;

void nearest_neighbor_upsampler(hls::stream<video_pixel> &in_stream,hls::stream<video_pixel>&out_stream)
{
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    video_pixel line_buffer[IN_WIDTH];
    for (int y_in = 0;y_in < IN_HIGHT;++y_in)
    {
        for (int x_in = 0; x_in < IN_WIDTH;++x_in)
        {
            #pragma HLS PIPELINE II=1
            line_buffer[x_in] = in_stream.read();
        }
        
        for (int y_repeat = 0;y_repeat < SCALE_FACTOR;++y_repeat)
        {
            for (int x_in = 0;x_in < IN_WIDTH;++x_in)
            {
                video_pixel current_pixel = line_buffer[x_in]
                
                for (int x_repeat = 0;x_repeat < SCALE_FACTOR;++x_repeat)
                {
                    #pragma HLS PIPELINE II=1
                    bool is_last = (y_in == IN_HIGHT - 1) &&
                                   (y_repeat == SCALE_FACTOR -1) &&
                                   (x_in == IN_WIDTH - 1) &&
                                   (x_repeat == SCALE_FACTOR -1);
                    current_pixel.last = is_last;
                    current_pixel,keep = -1;

                    out_stream.write(current_pixel);
                }
            }
        }
    }
}