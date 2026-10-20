具体函数在 `E:\Data\MCU_Project\Stm32_Project\BSP_MATH\FFT\mdeal\fftdeal.c`
## 1.去直流
"去直流"是指从信号中移除直流分量，也就是信号中的零频率成分。直流分量代表信号的偏移量，它的频率为零

函数实现
```c
void removeDC(void) {
  float32_t mean = 0.0;
    int i;
    for(i=0;i<fft_size;i++) {
      mean += (float)(ADC_Buffer[i]*3.3/4096);
    }
    mean /=fft_size;
    for(i=0;i<fft_size;i++) {
      input_signal[i] = (float)(ADC_Buffer[i]*3.3/4096) - mean;
    }
}
```
* 在时域上将偏移量完全减掉
## 3.FFT数据存入
```c
    for(i=0;i<FFT_SIZE;i++) {
      FFT_Input[2*i] = (float)(signal[i]);
      FFT_Input[2*i+1] = 0;
    }
```
* 实部存入，虚部为0
* [[为什么要人为做一个双倍大小的数组，并交错填 0的原因]]
## 4.使用 ARM CMSIS DSP 库中的函数来执行基于 ARM Cortex-M 处理器的快速傅里叶变换（FFT）操作
```c
arm_cfft_f32(&arm_cfft_sR_f32_len1024,FFT_Input,ifftFlag,doBitReverse);
```
* 示例结构体arm_cfft_sR_f32_len1024(1024个数)
* FFT_Input FFT数组
* `ifftFlag` 和 `doBitReverse` 是用于控制 FFT 变换的标志位(这里我`ifftFlag是0``doBitReverse是1`)
## 5.得到频域值
```c
arm_cmplx_mag_f32(FFT_Input, FFT_Output, fft_size);
```
## 6.计算电压幅值、频率大小
设ADC采样频率为f,采样点数为N,FFT_Output 对应最大值索引是k
```bash
电压幅值
FFT_Output/(N/2)
波形频率
f0=(k*f)/N
```
