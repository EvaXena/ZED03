#ifndef NNET_TANHLU_H_
#define NNET_TANHLU_H_

#include "nnet_common.h" 
#include "nnet_activation.h" 

namespace nnet {

struct tanhlu_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    typedef ap_fixed<18, 8> table_t;
};

template <class data_T, class res_T, typename CONFIG_T, class alpha_T, class beta_T, class lambd_T>
void tanhlu(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    alpha_T alpha,
    beta_T beta,
    lambd_T lambd
)
{
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }


    for (int i = 0; i < (CONFIG_T::height * CONFIG_T::width * CONFIG_T::n_in)/ data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_pack = data.read();
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL

            typename CONFIG_T::table_t data_p = in_pack[j];
            
            typename CONFIG_T::table_t lambd_cast = lambd;
            typename CONFIG_T::table_t alpha_cast = alpha;
            typename CONFIG_T::table_t beta_cast = beta;

            typename CONFIG_T::table_t lambd_x = lambd_cast * data_p;
            
            int data_round = (double)lambd_x * (CONFIG_T::table_size / 8.0);
            int index = data_round + (CONFIG_T::table_size / 2);
            
            if (index < 0) index = 0;
            if (index >= CONFIG_T::table_size) index = CONFIG_T::table_size - 1;
                
            typename CONFIG_T::table_t tanh_val = tanh_table[index];
            
            typename CONFIG_T::table_t term1 = tanh_val * alpha_cast;
            typename CONFIG_T::table_t term2 = beta_cast * data_p;
            
            out_pack[j] = term1 + term2;
        }
        res.write(out_pack);
    }
}

} // namespace nnet

#endif // NNET_TANHLU_H_