/*!
 * \file chunk_gated_delta_rule_fwd_h_def.cpp
 * \brief ChunkGatedDeltaRuleFwdH 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class ChunkGatedDeltaRuleFwdH : public OpDef {
public:
    explicit ChunkGatedDeltaRuleFwdH(const char* name) : OpDef(name)
    {
    this->Input("k")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("w")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("u")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("g")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("initial_state")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("cu_seqlens")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("chunk_indices")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("h")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("v")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("final_state")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("chunk_size")
        .AttrType(OPTIONAL)
        .Int(64);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ChunkGatedDeltaRuleFwdH);
} // namespace ops
