/*******************************************************************************
* Copyright (C) 2019-2022 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// powernet
// Created using ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --prefix powernet --checkpoint-file trained/powernet_qat_best3-q.pth.tar --config-file networks/powernet.yaml --overwrite --device MAX78000 --timer 0 --display-checkpoint --verbose

// DO NOT EDIT - regenerate this file instead!

// Configuring 4 layers
// Input data: HWC
// Layer 0: 1x103x1 flattened to 103x1x1, no pooling, linear, ReLU, 64x1x1 output
// Layer 1: 64x1x1, no pooling, linear, ReLU, 144x1x1 output
// Layer 2: 144x1x1, no pooling, linear, ReLU, 36x1x1 output
// Layer 3: 36x1x1, no pooling, linear, no activation, 1x1x1 output

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "gcfr_regs.h"
#include "cnn.h"
#include "weights.h"

void CNN_ISR(void)
{
  // Acknowledge interrupt to all quadrants
  *((volatile uint32_t *) 0x50100000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50500000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50900000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50d00000) &= ~((1<<12) | 1);

  CNN_COMPLETE; // Signal that processing is complete
#ifdef CNN_INFERENCE_TIMER
  cnn_time = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
#else
  cnn_time = 1;
#endif
}

int cnn_continue(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) |= 1; // Re-enable quadrant 0

  return CNN_OK;
}

int cnn_stop(void)
{
  *((volatile uint32_t *) 0x50100000) &= ~1; // Disable quadrant 0

  return CNN_OK;
}

void memcpy32(uint32_t *dst, const uint32_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

static const uint32_t kernels[] = KERNELS;

int cnn_load_weights(void)
{
  uint32_t len;
  volatile uint32_t *addr;
  const uint32_t *ptr = kernels;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    *((volatile uint8_t *) ((uint32_t) addr | 1)) = 0x01; // Set address
    len = *ptr++;
    while (len-- > 0)
      *addr++ = *ptr++;
  }

  return CNN_OK;
}

static const uint8_t bias_0[] = BIAS_0;
static const uint8_t bias_1[] = BIAS_1;
static const uint8_t bias_2[] = BIAS_2;
static const uint8_t bias_3[] = BIAS_3;

static void memcpy_8to32(uint32_t *dst, const uint8_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

int cnn_load_bias(void)
{
  memcpy_8to32((uint32_t *) 0x50108000, bias_0, sizeof(uint8_t) * 64);
  memcpy_8to32((uint32_t *) 0x50508000, bias_1, sizeof(uint8_t) * 144);
  memcpy_8to32((uint32_t *) 0x50908000, bias_2, sizeof(uint8_t) * 36);
  memcpy_8to32((uint32_t *) 0x50d08000, bias_3, sizeof(uint8_t) * 1);

  return CNN_OK;
}

int cnn_init(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x50100000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50100004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50100008) = 0x00000003; // Layer count
  *((volatile uint32_t *) 0x50500000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50500004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50500008) = 0x00000003; // Layer count
  *((volatile uint32_t *) 0x50900000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50900004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50900008) = 0x00000003; // Layer count
  *((volatile uint32_t *) 0x50d00000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50d00004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50d00008) = 0x00000003; // Layer count

  return CNN_OK;
}

int cnn_configure(void)
{
  // Layer 0 quadrant 0
  *((volatile uint32_t *) 0x50100310) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100390) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50100a10) = 0x0001f806; // Layer control 2
  *((volatile uint32_t *) 0x50100610) = 0x0000cdf8; // Mask offset and count
  *((volatile uint32_t *) 0x50100110) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100690) = 0x006600cd; // TRAM ptr max
  *((volatile uint32_t *) 0x50100790) = 0x081a3000; // Post processing register
  *((volatile uint32_t *) 0x50100710) = 0x00010001; // Mask and processor enables

  // Layer 0 quadrant 1
  *((volatile uint32_t *) 0x50500310) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500390) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a10) = 0x0001f806; // Layer control 2
  *((volatile uint32_t *) 0x50500610) = 0x0000cdf8; // Mask offset and count
  *((volatile uint32_t *) 0x50500110) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50500690) = 0x006600cd; // TRAM ptr max
  *((volatile uint32_t *) 0x50500790) = 0x081a2000; // Post processing register

  // Layer 0 quadrant 2
  *((volatile uint32_t *) 0x50900310) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900390) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a10) = 0x0001f806; // Layer control 2
  *((volatile uint32_t *) 0x50900610) = 0x0000cdf8; // Mask offset and count
  *((volatile uint32_t *) 0x50900110) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900690) = 0x006600cd; // TRAM ptr max
  *((volatile uint32_t *) 0x50900790) = 0x081a2000; // Post processing register

  // Layer 0 quadrant 3
  *((volatile uint32_t *) 0x50d00310) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00390) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a10) = 0x0001f806; // Layer control 2
  *((volatile uint32_t *) 0x50d00610) = 0x0000cdf8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00110) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d00690) = 0x006600cd; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00790) = 0x081a2000; // Post processing register

  // Layer 1 quadrant 0
  *((volatile uint32_t *) 0x50100394) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100494) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50100514) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50100594) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a14) = 0x00017820; // Layer control 2
  *((volatile uint32_t *) 0x50100614) = 0xcf00d378; // Mask offset and count
  *((volatile uint32_t *) 0x50100114) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100794) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100714) = 0xffffffff; // Mask and processor enables

  // Layer 1 quadrant 1
  *((volatile uint32_t *) 0x50500394) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500494) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50500514) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50500594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a14) = 0x00017820; // Layer control 2
  *((volatile uint32_t *) 0x50500614) = 0xcf00d378; // Mask offset and count
  *((volatile uint32_t *) 0x50500114) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50500794) = 0x00023000; // Post processing register
  *((volatile uint32_t *) 0x50500714) = 0xffffffff; // Mask and processor enables

  // Layer 1 quadrant 2
  *((volatile uint32_t *) 0x50900394) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900494) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50900514) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50900594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a14) = 0x00017820; // Layer control 2
  *((volatile uint32_t *) 0x50900614) = 0xcf00d378; // Mask offset and count
  *((volatile uint32_t *) 0x50900114) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900794) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900714) = 0xffffffff; // Mask and processor enables

  // Layer 1 quadrant 3
  *((volatile uint32_t *) 0x50d00394) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00494) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d00514) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a14) = 0x00017820; // Layer control 2
  *((volatile uint32_t *) 0x50d00614) = 0xcf00d378; // Mask offset and count
  *((volatile uint32_t *) 0x50d00114) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d00794) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00714) = 0xffffffff; // Mask and processor enables

  // Layer 2 quadrant 0
  *((volatile uint32_t *) 0x50100318) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100398) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100598) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a18) = 0x00011802; // Layer control 2
  *((volatile uint32_t *) 0x50100618) = 0xd380d6d8; // Mask offset and count
  *((volatile uint32_t *) 0x50100118) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100718) = 0xffffffff; // Mask and processor enables

  // Layer 2 quadrant 1
  *((volatile uint32_t *) 0x50500318) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500398) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500598) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a18) = 0x00011802; // Layer control 2
  *((volatile uint32_t *) 0x50500618) = 0xd380d6d8; // Mask offset and count
  *((volatile uint32_t *) 0x50500118) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50500798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500718) = 0xffffffff; // Mask and processor enables

  // Layer 2 quadrant 2
  *((volatile uint32_t *) 0x50900318) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900398) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900598) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a18) = 0x00011802; // Layer control 2
  *((volatile uint32_t *) 0x50900618) = 0xd380d6d8; // Mask offset and count
  *((volatile uint32_t *) 0x50900118) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900798) = 0x00023000; // Post processing register
  *((volatile uint32_t *) 0x50900718) = 0xffffffff; // Mask and processor enables

  // Layer 2 quadrant 3
  *((volatile uint32_t *) 0x50d00318) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00398) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00598) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a18) = 0x00011802; // Layer control 2
  *((volatile uint32_t *) 0x50d00618) = 0xd380d6d8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00118) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d00798) = 0x00022000; // Post processing register

  // Layer 3 quadrant 0
  *((volatile uint32_t *) 0x5010039c) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5010041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5010059c) = 0x0001e920; // Layer control
  *((volatile uint32_t *) 0x5010061c) = 0xd6e0d6e0; // Mask offset and count
  *((volatile uint32_t *) 0x5010011c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x5010079c) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x5010071c) = 0xffffffff; // Mask and processor enables

  // Layer 3 quadrant 1
  *((volatile uint32_t *) 0x5050039c) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5050041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5050059c) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x5050061c) = 0xd6e0d6e0; // Mask offset and count
  *((volatile uint32_t *) 0x5050011c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x5050079c) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x5050071c) = 0xffffffff; // Mask and processor enables

  // Layer 3 quadrant 2
  *((volatile uint32_t *) 0x5090039c) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5090041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5090059c) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x5090061c) = 0xd6e0d6e0; // Mask offset and count
  *((volatile uint32_t *) 0x5090011c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x5090079c) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x5090071c) = 0x000f000f; // Mask and processor enables

  // Layer 3 quadrant 3
  *((volatile uint32_t *) 0x50d0039c) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d0041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d0059c) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50d0061c) = 0xd6e0d6e0; // Mask offset and count
  *((volatile uint32_t *) 0x50d0011c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d0079c) = 0x00027000; // Post processing register


  return CNN_OK;
}

int cnn_start(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) = 0x00100808; // Enable quadrant 0
  *((volatile uint32_t *) 0x50500000) = 0x00100809; // Enable quadrant 1
  *((volatile uint32_t *) 0x50900000) = 0x00100809; // Enable quadrant 2
  *((volatile uint32_t *) 0x50d00000) = 0x00100809; // Enable quadrant 3

#ifdef CNN_INFERENCE_TIMER
  MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
#endif

  CNN_START; // Allow capture of processing time
  *((volatile uint32_t *) 0x50100000) = 0x00100009; // Master enable quadrant 0

  return CNN_OK;
}

int cnn_unload(uint32_t *out_buf)
{
  volatile uint32_t *addr;

  // Custom unload for this network, layer 3: 32-bit data, shape: (1, 1, 1)
  addr = (volatile uint32_t *) 0x50400000;
  *out_buf++ = *addr++;

  return CNN_OK;
}

int cnn_enable(uint32_t clock_source, uint32_t clock_divider)
{
  // Reset all domains, restore power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg1 = 0xf; // Mask memory
  MXC_GCFR->reg0 = 0xf; // Power
  MXC_GCFR->reg2 = 0x0; // Iso
  MXC_GCFR->reg3 = 0x0; // Reset

  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | clock_divider | clock_source;
  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); // Enable CNN clock

  MXC_NVIC_SetVector(CNN_IRQn, CNN_ISR); // Set CNN complete vector

  return CNN_OK;
}

int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutClr(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_disable(void)
{
  // Disable CNN clock
  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

  // Disable power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg2 |= 0xf; // Iso
  MXC_GCFR->reg0 = 0x0; // Power
  MXC_GCFR->reg1 = 0x0; // Mask memory
  MXC_GCFR->reg3 = 0x0; // Reset

  return CNN_OK;
}

