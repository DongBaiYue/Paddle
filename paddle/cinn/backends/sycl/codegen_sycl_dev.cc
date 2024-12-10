// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include <glog/logging.h>
#include <paddle/cinn/utils/string.h>

#include <fstream>
#include <set>
#include <unordered_set>

#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/hlir/op/op_util.h"

#include "paddle/cinn/backends/sycl/compiler_sycl.h"
using cinn::backends::syclrtc::NUM;

namespace cinn {
namespace backends {

const std::string &CodeGenSYCL_Dev::GetSourceHeader() {
  static std::string source_header =
      R"(#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
)";
  return source_header;
}

CodeGenSYCL_Dev::CodeGenSYCL_Dev(Target target) : CodeGenC(target) {
  CHECK(target.language == Target::Language::sycl)
      << "The target language is not sycl";
}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module,
                                     bool for_syclrtc) {
  for_syclrtc_ = for_syclrtc;
  auto source = Compile(module, OutputKind::CImpl);

  return source;
}

void CodeGenSYCL_Dev::Compile(const ir::Module &module,
                              const Outputs &outputs) {
  PADDLE_THROW(::common::errors::Fatal("CINN_SYCL_codegen_NOT_IMPLEMENTED"));
}

void CodeGenSYCL_Dev::Compile(const ir::LoweredFunc &func) {
  IrPrinter::Visit(Expr(func));
}

std::vector<Expr> CodeGenSYCL_Dev::GenerateBufferAliasExprs(
    const ir::_LoweredFunc_ *op, const std::vector<ir::Buffer> &temp_buffers) {
  std::set<ir::Buffer> temp_buffer_set(temp_buffers.begin(),
                                       temp_buffers.end());
  // prepare temp buffer alias
  std::vector<Expr> buffer_alias;
  auto tensors = ir::ir_utils::CollectIRNodes(op->body, [&](const Expr *x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() &&
           temp_buffer_set.count(x->as_tensor()->buffer);
  });

  // unique tensors
  std::set<ir::Tensor> unique_tensors;
  for (auto &e : tensors) {
    unique_tensors.insert(e.as_tensor_ref());
  }

  for (auto &t : unique_tensors) {
    auto data_type = t->type();
    auto data_ptr_type = data_type;
    data_ptr_type.set_cpp_handle();

    Var t_var(t->name, data_ptr_type);
    Var buf_var(t->buffer->name, data_ptr_type);
    buffer_alias.push_back(ir::Let::Make(t_var, buf_var));
  }

  return buffer_alias;
}

void CodeGenSYCL_Dev::Visit(const ir::_LoweredFunc_ *op) {
  // clear names valid within scope when enter a new function
  local_var_names_.clear();

  // Print the packed function
  str_ += "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  str_ += "void ";
  str_ += op->name;
  str_ +=
      "(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, "
      "void** void_args) {\n";
  IncIndent();
  // read void_args
  PrintFunctionDeclaration(op);
  DoIndent();
  str_ += "Q.submit([&](sycl::handler &h) {\n";
  IncIndent();
  DoIndent();
  str_ += "h.parallel_for<class " + GenerateKernelName(op) +
          ">(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), "
          "[=](sycl::nd_item<3> item) "
          "[[intel::kernel_args_restrict]]";
  if (op->cuda_axis_info.valid()) {
    bool has_symbol_in_thread_num = false;
    std::string launch_bounds_max_work_group_size =
        "[[intel::max_work_group_size(";
    for (int i = 0; i < 3; i++) {
      ir::Expr block_dim = op->cuda_axis_info.block_dim(i);
      if (block_dim.is_constant()) {
        launch_bounds_max_work_group_size +=
            std::to_string(block_dim.as_int64());
        if (i < 2) {
          launch_bounds_max_work_group_size += ", ";
        }
      } else {
        has_symbol_in_thread_num = true;
        break;
      }
    }
    launch_bounds_max_work_group_size += ")]]";
    if (!has_symbol_in_thread_num) {
      str_ += launch_bounds_max_work_group_size;
    }
  }
  str_ += "\n";
  // function body
  PrintFunctionBody(op);

  str_ += ");\n";
  DecIndent();
  DoIndent();
  str_ += "});\n";
  DecIndent();
  str_ += "}\n";
}

void CodeGenSYCL_Dev::Visit(const ir::_Var_ *op) {
  if (utils::StartsWith(op->name, "threadIdx") ||
      utils::StartsWith(op->name, "blockIdx")) {
    if (utils::StartsWith(op->name, "threadIdx")) {
      str_ += "(int)item.get_local_id(";
    } else {
      str_ += "(int)item.get_group(";
    }
    if (utils::EndsWith(op->name, "x")) {
      str_ += std::to_string(2);
    } else if (utils::EndsWith(op->name, "y")) {
      str_ += std::to_string(1);
    } else if (utils::EndsWith(op->name, "z")) {
      str_ += std::to_string(0);
    }
    str_ += ")";
  } else {
    str_ += op->name;
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Alloc *op) {
  PADDLE_ENFORCE_NE(
      op->destination.as_buffer(),
      nullptr,
      ::common::errors::InvalidArgument("ir::Alloc's buffer cannot nullptr."));
  PrintTempBufferCreation(op->destination.as_buffer_ref());
}

void CodeGenSYCL_Dev::Visit(const ir::Min *op) {
  str_ += hlir::GetExternFuncName(target_, op->type(), "min");
  str_ += "(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Max *op) {
  str_ += hlir::GetExternFuncName(target_, op->type(), "max");
  str_ += "(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSYCL_Dev::PrintFunctionBody(const ir::_LoweredFunc_ *op) {
  DoIndent();

  std::vector<Expr> new_body;

  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alis_var_exprs = op->CudaAliasVarExprs();

#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alis_var_exprs)

  new_body.push_back(op->body);

  Expr func_body = ir::Block::Make(new_body);

  optim::SimplifyBlocks(&func_body);
  // Make sure that the function's body is wrapped by a block
  if (!func_body.As<ir::Block>()) {
    func_body = ir::Block::Make({func_body});
  }
  IrPrinter::Visit(func_body);
}

void CodeGenSYCL_Dev::PrintFunctionDeclaration(const ir::_LoweredFunc_ *op) {
  for (int i = 0; i < op->args.size(); i++) {
    DoIndent();
    auto &arg = op->args[i];
    if (arg.is_buffer()) {
      // In CUDA kernel, only primitive type is supported, so we replace the
      // buffer with T*j
      if (arg.is_input()) str_ += "const ";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
      // str_ += kCKeywordRestrict;
      str_ += " ";
      str_ += ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
      str_ += " = (";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
    } else if (arg.is_var()) {
      if (arg.var_arg()->type().is_cpp_handle()) {
        // str_ += kCKeywordRestrict;
      }
      str_ += GetTypeRepr(arg.type());
      str_ += " ";
      str_ += arg.name();
      str_ += " = (";
      str_ += GetTypeRepr(arg.type());
    } else {
      CINN_NOT_IMPLEMENTED
    }
    str_ += ")(*(void **)(void_args[";
    str_ += std::to_string(i);
    str_ += "]));\n";
  }
}

void CodeGenSYCL_Dev::PrintBuiltinCodes() {}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module,
                                     CodeGenC::OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (for_syclrtc_) {
      str_ += "#ifdef __cplusplus\n";
      str_ += "extern \"C\" {\n";
      str_ += "#endif\n";
    }

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    PADDLE_THROW(::common::errors::Fatal("Not supported OutputKind"));
  }

  if (for_syclrtc_) {
    str_ += "\n#ifdef __cplusplus\n";
    str_ += "}\n";
    str_ += "#endif\n";
  }
  return str_;
}

void CodeGenSYCL_Dev::PrintIncludes() { str_ += GetSourceHeader(); }

void CodeGenSYCL_Dev::PrintTempBufferCreation(const ir::Buffer &buffer) {
  VLOG(3) << "PrintTempBufferCreation: " << buffer->name;
  VLOG(3) << "buffer->memory_type: " << buffer->memory_type;
  PADDLE_ENFORCE_NE(buffer->type(),
                    Void(),
                    ::common::errors::InvalidArgument(
                        "Buffer type cannot be void in CodeGenSYCL_Dev"));
  auto print_gpu_memory = [&](const std::string &mark) {
    str_ += mark;
    str_ += GetTypeRepr(buffer->dtype);
    str_ += " ";
    str_ += buffer->name;
    str_ += " ";

    str_ += "[ ";
    Expr buffer_size(1);
    for (int i = 0; i < buffer->shape.size(); i++) {
      buffer_size = buffer_size * buffer->shape[i];
    }
    optim::Simplify(&buffer_size);
    IrPrinter::Visit(buffer_size);
    str_ += " ]";
  };
  switch (buffer->memory_type) {
    case ir::MemoryType::GPUShared: {
      str_ += "auto ";
      str_ += buffer->name;
      str_ += " = *sycl::group_local_memory<";
      str_ += GetTypeRepr(buffer->dtype);
      str_ += "[ ";
      Expr buffer_size(1);
      for (int i = 0; i < buffer->shape.size(); i++) {
        buffer_size = buffer_size * buffer->shape[i];
      }
      optim::Simplify(&buffer_size);
      IrPrinter::Visit(buffer_size);
      str_ += " ]>(item.get_group())";
      break;
    }

    case ir::MemoryType::GPULocal: {
      break;
    }

    default:
      PADDLE_THROW(::common::errors::Fatal(
          "SYCL device codegen not support memory %s, %s",
          buffer->name,
          buffer->memory_type));
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Call *op) {
  VLOG(3) << "CodeGenSYCL visiting call op: " << op->name;
  VLOG(3) << "op->read_args.size(): " << op->read_args.size();
  VLOG(3) << "op->write_args.size(): " << op->write_args.size();
  if (op->name == "__syncthreads") {
    str_ += "sycl::group_barrier(item.get_group())";
    return;
  }
  str_ += op->name;
  str_ += "(";

  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      auto &arg = op->read_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->read_args.back().as_tensor()) {
      str_ += op->read_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->read_args.back());
    }
  }

  if (!op->write_args.empty()) {
    str_ += ", ";
    for (int i = 0; i < op->write_args.size() - 1; i++) {
      auto &arg = op->write_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->write_args.back().as_tensor()) {
      str_ += op->write_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->write_args.back());
    }
  }
  // sycl need parameter nd_item
  if ((op->name.find("cinn_block_reduce") != std::string::npos) ||
      (op->name.find("cinn_warp_reduce") != std::string::npos)) {
    str_ += ", item";
  }

  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Let *op) {
  VLOG(3) << "CodeGenSYCL visiting let op: " << op->symbol;
  PADDLE_ENFORCE_EQ(
      op->type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "ir::Let's op type cannot be valid in CodeGenSYCL_Dev"));

  local_var_names_.insert(op->symbol.as_var()->name);
}

void CodeGenSYCL_Dev::Visit(const ir::Load *op) {
  VLOG(3) << "CodeGenSYCL visiting load op: " << op->name();
  ir::Expr offset = [&] {
    if (load_to_offset_.count(op) == 0) {
      load_to_offset_[op] = op->index();
    }
    return load_to_offset_.at(op);
  }();
  if (local_var_names_.count(op->tensor.As<ir::_Tensor_>()->name)) {
    str_ += op->tensor.As<ir::_Tensor_>()->name;
    return;
  }

  if (offset.type().is_vector()) {
    CHECK(op->type().is_vector());
    Expr dense_strided_ramp = detail::StridedRampBase(offset, 1);
    PrintStackVecType(op->type().ElementOf(), offset.type().lanes());
    str_ += "::Load(";
    str_ += op->tensor.As<ir::_Tensor_>()->name;
    str_ += ", ";
    if (dense_strided_ramp.defined()) {
      // Loading a continuous Ramp address.
      IrPrinter::Visit(dense_strided_ramp);
    } else {
      IrPrinter::Visit(offset);
    }
    str_ += ")";
  } else if (op->is_addr_tensor()) {
    auto *tensor = op->tensor.As<ir::_Tensor_>();
    str_ += tensor->name;
    str_ += "[";
    IrPrinter::Visit(offset);
    str_ += "]";
  } else {
    CINN_RUNTIME_NOT_IMPLEMENTED
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Store *op) {
  VLOG(3) << "CodeGenSYCL visiting store op: " << op->name();
  CHECK(op->is_addr_tensor());
  ir::Expr offset = [&] {
    if (store_to_offset_.count(op) == 0) {
      store_to_offset_[op] = op->index();
    }
    return store_to_offset_.at(op);
  }();
  auto *tensor = op->tensor.As<ir::_Tensor_>();
  CHECK(tensor);
  if (local_var_names_.count(tensor->name)) {
    str_ += "auto ";
    str_ += tensor->name;
    str_ += " = ";
    IrPrinter::Visit(op->value);
    return;
  }
  str_ += "cinn_sycl_store(";
  str_ += tensor->name;
  str_ += ", ";
  IrPrinter::Visit(offset);
  str_ += ", ";
  IrPrinter::Visit(op->value);
  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Ramp *op) {
  if (op->stride.as_int32() != 1)
    CINN_RUNTIME_NOT_IMPLEMENTED
  str_ += "IndexVec<";
  str_ += std::to_string(op->lanes);
  str_ += ">::Ramp(";
  IrPrinter::Visit(op->base);
  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Broadcast *op) {
  IrPrinter::Visit(op->value);
}

void CodeGenSYCL_Dev::Visit(const ir::Select *op) {
  str_ += "cinn_sycl_select(";
  IrPrinter::Visit(op->condition);
  str_ += ", ";
  IrPrinter::Visit(op->true_value);
  str_ += ", ";
  IrPrinter::Visit(op->false_value);
  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Cast *op) {
  VLOG(3) << "CodeGenSYCL visiting cast op: " << op;
  VLOG(3) << op->v().type() << " to " << op->type();
  if (op->v().type().is_vector()) {
    if (op->v().type().is_bool()) {
      IrPrinter::Visit(op->v());
    } else {
      str_ += "cinn_sycl_cast<";
      str_ += GetTypeRepr(op->type());
      str_ += ">(";
      IrPrinter::Visit(op->v());
      str_ += ")";
    }
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenSYCL_Dev::PrintStackVecType(Type type, int lanes) {
  str_ += "DataVec<";
  str_ += GetTypeRepr(type);
  str_ += ", ";
  str_ += std::to_string(lanes);
  str_ += ">";
}

std::string CodeGenSYCL_Dev::GenerateKernelName(const ir::_LoweredFunc_ *op) {
  std::string kernel_name = "space" + std::to_string(NUM::getNum());
  kernel_name += "_";
  kernel_name += op->name;
  return kernel_name;
}

}  // namespace backends
}  // namespace cinn
