// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <jxl/encode.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
static void Check(bool ok) { if(!ok) { std::fprintf(stderr,"WP3 dump failed\n"); std::exit(1); } }
int main(int argc,char** argv) {
  if(argc<4) { std::fprintf(stderr,"usage: wp3_dump DISTANCE OUTDIR tile.rgb...\n"); return 2; }
  std::filesystem::create_directories(argv[2]); size_t bytes=0;
  for(int arg=3;arg<argc;++arg) {
    const std::string prefix=std::string(argv[2])+"/"+std::filesystem::path(argv[arg]).stem().string();
    Check(setenv("JXL_WP3_DUMP",prefix.c_str(),1)==0);
    std::vector<uint8_t> rgb(512*512*3);
    std::ifstream f(argv[arg],std::ios::binary); f.read(reinterpret_cast<char*>(rgb.data()),rgb.size());
    Check(f.gcount()==static_cast<std::streamsize>(rgb.size()) && f.peek()==EOF);
    JxlEncoder* enc=JxlEncoderCreate(nullptr); Check(enc);
    JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
    info.xsize=info.ysize=512; info.bits_per_sample=8; info.num_color_channels=3;
    info.uses_original_profile=JXL_FALSE;
    Check(JxlEncoderSetBasicInfo(enc,&info)==JXL_ENC_SUCCESS);
    JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
    Check(JxlEncoderSetColorEncoding(enc,&color)==JXL_ENC_SUCCESS);
    auto* frame=JxlEncoderFrameSettingsCreate(enc,nullptr);
    Check(JxlEncoderFrameSettingsSetOption(frame,JXL_ENC_FRAME_SETTING_EFFORT,8)==JXL_ENC_SUCCESS);
    Check(JxlEncoderSetFrameDistance(frame,std::stof(argv[1]))==JXL_ENC_SUCCESS);
    JxlPixelFormat pf{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
    Check(JxlEncoderAddImageFrame(frame,&pf,rgb.data(),rgb.size())==JXL_ENC_SUCCESS);
    JxlEncoderCloseInput(enc);
    uint8_t output[65536]; JxlEncoderStatus status;
    do { uint8_t* p=output; size_t avail=sizeof(output);
      status=JxlEncoderProcessOutput(enc,&p,&avail); bytes+=sizeof(output)-avail;
    } while(status==JXL_ENC_NEED_MORE_OUTPUT);
    Check(status==JXL_ENC_SUCCESS); JxlEncoderDestroy(enc);
    std::fprintf(stderr,"dumped %s\n",argv[arg]);
  }
  std::printf("{\"tiles\":%d,\"distance\":%s,\"encoded_bytes\":%zu}\n",argc-3,argv[1],bytes);
}
