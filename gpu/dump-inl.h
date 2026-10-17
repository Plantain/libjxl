// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include "gpu/spike_format.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/dec_xyb.h"

namespace jxl {
inline Status DumpGpuSpike(const FrameHeader& fh, const Image3F& original,
                           const ImageBundle& decoded, const ImageF& diff,
                           const ImageF& dist, PassesEncoderState* enc,
                           const JxlCmsInterface& cms, int iteration) {
  const char* prefix = std::getenv("JXL_WP3_DUMP");
  if (!prefix) return true;
  JXL_ENSURE(original.xsize() == 512 && original.ysize() == 512);
  JXL_ENSURE(enc->coeffs.size() == 1 && fh.chroma_subsampling.Is444());
  auto in = std::make_unique<jxl_wp3::Input>();
  auto out = std::make_unique<jxl_wp3::Oracle>();
  auto& s = enc->shared; auto& p = in->p; const auto& lf = fh.loop_filter;
  const auto qp = s.quantizer.GetParams();
  p.global_scale=qp.global_scale; p.quant_dc=qp.quant_dc;
  p.scale=s.quantizer.Scale(); p.inverse=s.quantizer.InvGlobalScale();
  p.xmul=std::pow(.8f, fh.x_qm_scale-2.f); p.bmul=std::pow(.8f, fh.b_qm_scale-2.f);
  p.gab_enabled=lf.gab; p.epf_iters=lf.epf_iters;
  const float gab[]={lf.gab_x_weight1,lf.gab_x_weight2,lf.gab_y_weight1,
                     lf.gab_y_weight2,lf.gab_b_weight1,lf.gab_b_weight2};
  std::copy(gab,gab+6,p.gab);
  std::copy(lf.epf_channel_scale,lf.epf_channel_scale+3,p.channel);
  std::copy(lf.epf_sharp_lut,lf.epf_sharp_lut+8,p.sharp);
  p.border=lf.epf_border_sad_mul; p.epf_quant=lf.epf_quant_mul;
  p.skip_dc_smoothing=(fh.flags & FrameHeader::kSkipAdaptiveDCSmoothing)!=0;
  p.dc_precision=jxl_wp3::DCPrecision();
  std::copy(jxl_wp3::CapturedDC(),jxl_wp3::CapturedDC()+3*jxl_wp3::kBlocks,in->dc);
  OutputEncodingInfo info;
  JXL_RETURN_IF_ERROR(info.SetFromMetadata(*s.metadata));
  for (int c=0;c<3;++c) {
    p.dc_mul[c]=s.quantizer.MulDC()[c]; p.dc_cfl[c]=s.cmap.base().DCFactors()[c];
    p.opsin_bias[c]=info.opsin_params.opsin_biases[c];
    p.opsin_cbrt[c]=info.opsin_params.opsin_biases_cbrt[c];
    for(int j=0;j<3;++j) p.matrix[3*c+j]=info.opsin_params.inverse_opsin_matrix[4*(3*c+j)];
  }
  std::copy(info.opsin_params.quant_biases,info.opsin_params.quant_biases+4,p.biases);
  p.intensity=80.f; p.distance=enc->cparams.butteraugli_distance;
  for (int g=0;g<4;++g) {
    int offset=0;
    for(int y=(g/2)*32;y<(g/2+1)*32;++y) for(int x=(g%2)*32;x<(g%2+1)*32;++x) {
      const auto a=s.ac_strategy.ConstRow(y)[x]; const int anchor=y*64+x;
      in->kind[anchor]=a.RawStrategy(); in->first[anchor]=a.IsFirstBlock();
      in->quant[anchor]=s.raw_quant_field.ConstRow(y)[x];
      in->sharp[anchor]=s.epf_sharpness.ConstRow(y)[x];
      if(!a.IsFirstBlock()) continue;
      const int w=8*a.covered_blocks_x(),h=8*a.covered_blocks_y(),n=w*h;
      in->cx[anchor]=w/8; in->cy[anchor]=h/8;
      in->cfl[anchor]=s.cmap.base().YtoXRatio(s.cmap.ytox_map.ConstRow(y/8)[x/8]);
      in->cfl[jxl_wp3::kBlocks+anchor]=s.cmap.base().YtoBRatio(s.cmap.ytob_map.ConstRow(y/8)[x/8]);
      for(int c=0;c<3;++c) {
        const auto* q=enc->coeffs[0]->PlaneRow(c,g,offset).ptr32;
        const float* m=s.quantizer.DequantMatrix(a.Strategy(),c);
        for(int k=0;k<n;++k) {
          const int at=c*jxl_wp3::kPixels+(y*8+k/w)*512+x*8+k%w;
          in->ac[at]=q[k]; in->matrices[at]=m[k];
        }
      }
      offset+=n;
    }
    JXL_ENSURE(offset==256*256);
  }
  ImageMetadata metadata=*decoded.metadata();
  ImageBundle store(decoded.memory_manager(),&metadata); const ImageBundle* linear;
  JXL_RETURN_IF_ERROR(TransformIfNeeded(decoded,ColorEncoding::LinearSRGB(false),cms,
                                       nullptr,&store,&linear));
  for(int c=0;c<3;++c) for(int y=0;y<512;++y) {
    std::copy_n(original.ConstPlaneRow(c,y),512,out->original+c*jxl_wp3::kPixels+y*512);
    std::copy_n(linear->color().ConstPlaneRow(c,y),512,out->linear+c*jxl_wp3::kPixels+y*512);
  }
  for(int y=0;y<512;++y) std::copy_n(diff.ConstRow(y),512,out->diff+y*512);
  for(int y=0;y<64;++y) std::copy_n(dist.ConstRow(y),64,out->dist+y*64);
  const std::string name=std::string(prefix)+".i"+std::to_string(iteration)+".wp3";
  FILE* f=std::fopen(name.c_str(),"wb"); JXL_ENSURE(f);
  const bool ok=std::fwrite(in.get(),sizeof(*in),1,f)==1 && std::fwrite(out.get(),sizeof(*out),1,f)==1;
  const int close=std::fclose(f); JXL_ENSURE(ok && close==0);
  return true;
}
}
