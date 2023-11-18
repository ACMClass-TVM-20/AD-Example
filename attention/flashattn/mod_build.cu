; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macos"

%0 = type { double }
%1 = type { ptr, %2, i32, %3, ptr, ptr, i64 }
%2 = type { i32, i32 }
%3 = type { i8, i8, i16 }

@__tvm_module_ctx = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMFuncCall = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMBackendGetFuncFromEnv = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMAPISetLastError = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str = private constant [62 x i8] c"Assert fail: num_args == 9, backward_op: num_args should be 9\00", align 1
@.str.1 = private constant [145 x i8] c"Assert fail: dO_handle_code == 3 or dO_handle_code == 13 or dO_handle_code == 7 or dO_handle_code == 4, backward_op: Expect arg[0] to be pointer\00", align 1
@.str.2 = private constant [141 x i8] c"Assert fail: Q_handle_code == 3 or Q_handle_code == 13 or Q_handle_code == 7 or Q_handle_code == 4, backward_op: Expect arg[1] to be pointer\00", align 1
@.str.3 = private constant [141 x i8] c"Assert fail: K_handle_code == 3 or K_handle_code == 13 or K_handle_code == 7 or K_handle_code == 4, backward_op: Expect arg[2] to be pointer\00", align 1
@.str.4 = private constant [141 x i8] c"Assert fail: V_handle_code == 3 or V_handle_code == 13 or V_handle_code == 7 or V_handle_code == 4, backward_op: Expect arg[3] to be pointer\00", align 1
@.str.5 = private constant [141 x i8] c"Assert fail: O_handle_code == 3 or O_handle_code == 13 or O_handle_code == 7 or O_handle_code == 4, backward_op: Expect arg[4] to be pointer\00", align 1
@.str.6 = private constant [141 x i8] c"Assert fail: L_handle_code == 3 or L_handle_code == 13 or L_handle_code == 7 or L_handle_code == 4, backward_op: Expect arg[5] to be pointer\00", align 1
@.str.7 = private constant [145 x i8] c"Assert fail: dQ_handle_code == 3 or dQ_handle_code == 13 or dQ_handle_code == 7 or dQ_handle_code == 4, backward_op: Expect arg[6] to be pointer\00", align 1
@.str.8 = private constant [145 x i8] c"Assert fail: dK_handle_code == 3 or dK_handle_code == 13 or dK_handle_code == 7 or dK_handle_code == 4, backward_op: Expect arg[7] to be pointer\00", align 1
@.str.9 = private constant [145 x i8] c"Assert fail: dV_handle_code == 3 or dV_handle_code == 13 or dV_handle_code == 7 or dV_handle_code == 4, backward_op: Expect arg[8] to be pointer\00", align 1
@.str.10 = private constant [112 x i8] c"Assert fail: 4 == T.tvm_struct_get(dO_handle, 0, 4, \22int32\22), backward_op.dO_handle.ndim is expected to equal 4\00", align 1
@.str.11 = private constant [250 x i8] c"Assert fail: T.tvm_struct_get(dO_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(dO_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(dO_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.dO_handle.dtype is expected to be float16\00", align 1
@.str.12 = private constant [191 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dO_handle_shape[0]) == 6, Argument backward_op.dO_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_dO_handle_shape[0])\00", align 1
@.str.13 = private constant [193 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dO_handle_shape[1]) == 32, Argument backward_op.dO_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_dO_handle_shape[1])\00", align 1
@.str.14 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dO_handle_shape[2]) == 512, Argument backward_op.dO_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_dO_handle_shape[2])\00", align 1
@.str.15 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dO_handle_shape[3]) == 128, Argument backward_op.dO_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_dO_handle_shape[3])\00", align 1
@.str.16 = private constant [318 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_dO_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_dO_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_dO_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_dO_handle_strides[0]), backward_op.dO_handle.strides: expected to be compact array\00", align 1
@.str.17 = private constant [206 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(dO_handle, 0, 8, \22uint64\22), Argument backward_op.dO_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(dO_handle, 0, 8, \22uint64\22)\00", align 1
@.str.18 = private constant [186 x i8] c"Assert fail: T.tvm_struct_get(dO_handle, 0, 10, \22int32\22) == 8, Argument backward_op.dO_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(dO_handle, 0, 10, \22int32\22)\00", align 1
@.str.19 = private constant [110 x i8] c"Assert fail: 4 == T.tvm_struct_get(Q_handle, 0, 4, \22int32\22), backward_op.Q_handle.ndim is expected to equal 4\00", align 1
@.str.20 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(Q_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(Q_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(Q_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.Q_handle.dtype is expected to be float16\00", align 1
@.str.21 = private constant [188 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_Q_handle_shape[0]) == 6, Argument backward_op.Q_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_Q_handle_shape[0])\00", align 1
@.str.22 = private constant [190 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_Q_handle_shape[1]) == 32, Argument backward_op.Q_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_Q_handle_shape[1])\00", align 1
@.str.23 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_Q_handle_shape[2]) == 512, Argument backward_op.Q_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_Q_handle_shape[2])\00", align 1
@.str.24 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_Q_handle_shape[3]) == 128, Argument backward_op.Q_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_Q_handle_shape[3])\00", align 1
@.str.25 = private constant [313 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_Q_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_Q_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_Q_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_Q_handle_strides[0]), backward_op.Q_handle.strides: expected to be compact array\00", align 1
@.str.26 = private constant [203 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, \22uint64\22), Argument backward_op.Q_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, \22uint64\22)\00", align 1
@.str.27 = private constant [183 x i8] c"Assert fail: T.tvm_struct_get(Q_handle, 0, 10, \22int32\22) == 8, Argument backward_op.Q_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(Q_handle, 0, 10, \22int32\22)\00", align 1
@.str.28 = private constant [189 x i8] c"Assert fail: dev_id == T.tvm_struct_get(Q_handle, 0, 9, \22int32\22), Argument backward_op.Q_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(Q_handle, 0, 9, \22int32\22)\00", align 1
@.str.29 = private constant [110 x i8] c"Assert fail: 4 == T.tvm_struct_get(K_handle, 0, 4, \22int32\22), backward_op.K_handle.ndim is expected to equal 4\00", align 1
@.str.30 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(K_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(K_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(K_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.K_handle.dtype is expected to be float16\00", align 1
@.str.31 = private constant [188 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_K_handle_shape[0]) == 6, Argument backward_op.K_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_K_handle_shape[0])\00", align 1
@.str.32 = private constant [190 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_K_handle_shape[1]) == 32, Argument backward_op.K_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_K_handle_shape[1])\00", align 1
@.str.33 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_K_handle_shape[2]) == 512, Argument backward_op.K_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_K_handle_shape[2])\00", align 1
@.str.34 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_K_handle_shape[3]) == 128, Argument backward_op.K_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_K_handle_shape[3])\00", align 1
@.str.35 = private constant [313 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_K_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_K_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_K_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_K_handle_strides[0]), backward_op.K_handle.strides: expected to be compact array\00", align 1
@.str.36 = private constant [203 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, \22uint64\22), Argument backward_op.K_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, \22uint64\22)\00", align 1
@.str.37 = private constant [183 x i8] c"Assert fail: T.tvm_struct_get(K_handle, 0, 10, \22int32\22) == 8, Argument backward_op.K_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(K_handle, 0, 10, \22int32\22)\00", align 1
@.str.38 = private constant [189 x i8] c"Assert fail: dev_id == T.tvm_struct_get(K_handle, 0, 9, \22int32\22), Argument backward_op.K_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(K_handle, 0, 9, \22int32\22)\00", align 1
@.str.39 = private constant [110 x i8] c"Assert fail: 4 == T.tvm_struct_get(V_handle, 0, 4, \22int32\22), backward_op.V_handle.ndim is expected to equal 4\00", align 1
@.str.40 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(V_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(V_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(V_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.V_handle.dtype is expected to be float16\00", align 1
@.str.41 = private constant [188 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_V_handle_shape[0]) == 6, Argument backward_op.V_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_V_handle_shape[0])\00", align 1
@.str.42 = private constant [190 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_V_handle_shape[1]) == 32, Argument backward_op.V_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_V_handle_shape[1])\00", align 1
@.str.43 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_V_handle_shape[2]) == 512, Argument backward_op.V_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_V_handle_shape[2])\00", align 1
@.str.44 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_V_handle_shape[3]) == 128, Argument backward_op.V_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_V_handle_shape[3])\00", align 1
@.str.45 = private constant [313 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_V_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_V_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_V_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_V_handle_strides[0]), backward_op.V_handle.strides: expected to be compact array\00", align 1
@.str.46 = private constant [203 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, \22uint64\22), Argument backward_op.V_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, \22uint64\22)\00", align 1
@.str.47 = private constant [183 x i8] c"Assert fail: T.tvm_struct_get(V_handle, 0, 10, \22int32\22) == 8, Argument backward_op.V_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(V_handle, 0, 10, \22int32\22)\00", align 1
@.str.48 = private constant [189 x i8] c"Assert fail: dev_id == T.tvm_struct_get(V_handle, 0, 9, \22int32\22), Argument backward_op.V_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(V_handle, 0, 9, \22int32\22)\00", align 1
@.str.49 = private constant [110 x i8] c"Assert fail: 4 == T.tvm_struct_get(O_handle, 0, 4, \22int32\22), backward_op.O_handle.ndim is expected to equal 4\00", align 1
@.str.50 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(O_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(O_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(O_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.O_handle.dtype is expected to be float16\00", align 1
@.str.51 = private constant [188 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_O_handle_shape[0]) == 6, Argument backward_op.O_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_O_handle_shape[0])\00", align 1
@.str.52 = private constant [190 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_O_handle_shape[1]) == 32, Argument backward_op.O_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_O_handle_shape[1])\00", align 1
@.str.53 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_O_handle_shape[2]) == 512, Argument backward_op.O_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_O_handle_shape[2])\00", align 1
@.str.54 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_O_handle_shape[3]) == 128, Argument backward_op.O_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_O_handle_shape[3])\00", align 1
@.str.55 = private constant [313 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_O_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_O_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_O_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_O_handle_strides[0]), backward_op.O_handle.strides: expected to be compact array\00", align 1
@.str.56 = private constant [203 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(O_handle, 0, 8, \22uint64\22), Argument backward_op.O_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(O_handle, 0, 8, \22uint64\22)\00", align 1
@.str.57 = private constant [183 x i8] c"Assert fail: T.tvm_struct_get(O_handle, 0, 10, \22int32\22) == 8, Argument backward_op.O_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(O_handle, 0, 10, \22int32\22)\00", align 1
@.str.58 = private constant [189 x i8] c"Assert fail: dev_id == T.tvm_struct_get(O_handle, 0, 9, \22int32\22), Argument backward_op.O_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(O_handle, 0, 9, \22int32\22)\00", align 1
@.str.59 = private constant [110 x i8] c"Assert fail: 3 == T.tvm_struct_get(L_handle, 0, 4, \22int32\22), backward_op.L_handle.ndim is expected to equal 3\00", align 1
@.str.60 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(L_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(L_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(L_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.L_handle.dtype is expected to be float16\00", align 1
@.str.61 = private constant [188 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_L_handle_shape[0]) == 6, Argument backward_op.L_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_L_handle_shape[0])\00", align 1
@.str.62 = private constant [190 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_L_handle_shape[1]) == 32, Argument backward_op.L_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_L_handle_shape[1])\00", align 1
@.str.63 = private constant [192 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_L_handle_shape[2]) == 512, Argument backward_op.L_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_L_handle_shape[2])\00", align 1
@.str.64 = private constant [249 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_L_handle_strides[2]) and 512 == T.Cast(\22int32\22, backward_op_L_handle_strides[1]) and 16384 == T.Cast(\22int32\22, backward_op_L_handle_strides[0]), backward_op.L_handle.strides: expected to be compact array\00", align 1
@.str.65 = private constant [203 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(L_handle, 0, 8, \22uint64\22), Argument backward_op.L_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(L_handle, 0, 8, \22uint64\22)\00", align 1
@.str.66 = private constant [183 x i8] c"Assert fail: T.tvm_struct_get(L_handle, 0, 10, \22int32\22) == 8, Argument backward_op.L_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(L_handle, 0, 10, \22int32\22)\00", align 1
@.str.67 = private constant [189 x i8] c"Assert fail: dev_id == T.tvm_struct_get(L_handle, 0, 9, \22int32\22), Argument backward_op.L_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(L_handle, 0, 9, \22int32\22)\00", align 1
@.str.68 = private constant [112 x i8] c"Assert fail: 4 == T.tvm_struct_get(dQ_handle, 0, 4, \22int32\22), backward_op.dQ_handle.ndim is expected to equal 4\00", align 1
@.str.69 = private constant [250 x i8] c"Assert fail: T.tvm_struct_get(dQ_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(dQ_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(dQ_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.dQ_handle.dtype is expected to be float16\00", align 1
@.str.70 = private constant [191 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dQ_handle_shape[0]) == 6, Argument backward_op.dQ_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_dQ_handle_shape[0])\00", align 1
@.str.71 = private constant [193 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dQ_handle_shape[1]) == 32, Argument backward_op.dQ_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_dQ_handle_shape[1])\00", align 1
@.str.72 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dQ_handle_shape[2]) == 512, Argument backward_op.dQ_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_dQ_handle_shape[2])\00", align 1
@.str.73 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dQ_handle_shape[3]) == 128, Argument backward_op.dQ_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_dQ_handle_shape[3])\00", align 1
@.str.74 = private constant [318 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_dQ_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_dQ_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_dQ_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_dQ_handle_strides[0]), backward_op.dQ_handle.strides: expected to be compact array\00", align 1
@.str.75 = private constant [206 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(dQ_handle, 0, 8, \22uint64\22), Argument backward_op.dQ_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(dQ_handle, 0, 8, \22uint64\22)\00", align 1
@.str.76 = private constant [186 x i8] c"Assert fail: T.tvm_struct_get(dQ_handle, 0, 10, \22int32\22) == 8, Argument backward_op.dQ_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(dQ_handle, 0, 10, \22int32\22)\00", align 1
@.str.77 = private constant [192 x i8] c"Assert fail: dev_id == T.tvm_struct_get(dQ_handle, 0, 9, \22int32\22), Argument backward_op.dQ_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(dQ_handle, 0, 9, \22int32\22)\00", align 1
@.str.78 = private constant [112 x i8] c"Assert fail: 4 == T.tvm_struct_get(dK_handle, 0, 4, \22int32\22), backward_op.dK_handle.ndim is expected to equal 4\00", align 1
@.str.79 = private constant [250 x i8] c"Assert fail: T.tvm_struct_get(dK_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(dK_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(dK_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.dK_handle.dtype is expected to be float16\00", align 1
@.str.80 = private constant [191 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dK_handle_shape[0]) == 6, Argument backward_op.dK_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_dK_handle_shape[0])\00", align 1
@.str.81 = private constant [193 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dK_handle_shape[1]) == 32, Argument backward_op.dK_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_dK_handle_shape[1])\00", align 1
@.str.82 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dK_handle_shape[2]) == 512, Argument backward_op.dK_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_dK_handle_shape[2])\00", align 1
@.str.83 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dK_handle_shape[3]) == 128, Argument backward_op.dK_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_dK_handle_shape[3])\00", align 1
@.str.84 = private constant [318 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_dK_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_dK_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_dK_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_dK_handle_strides[0]), backward_op.dK_handle.strides: expected to be compact array\00", align 1
@.str.85 = private constant [206 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(dK_handle, 0, 8, \22uint64\22), Argument backward_op.dK_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(dK_handle, 0, 8, \22uint64\22)\00", align 1
@.str.86 = private constant [186 x i8] c"Assert fail: T.tvm_struct_get(dK_handle, 0, 10, \22int32\22) == 8, Argument backward_op.dK_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(dK_handle, 0, 10, \22int32\22)\00", align 1
@.str.87 = private constant [192 x i8] c"Assert fail: dev_id == T.tvm_struct_get(dK_handle, 0, 9, \22int32\22), Argument backward_op.dK_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(dK_handle, 0, 9, \22int32\22)\00", align 1
@.str.88 = private constant [112 x i8] c"Assert fail: 4 == T.tvm_struct_get(dV_handle, 0, 4, \22int32\22), backward_op.dV_handle.ndim is expected to equal 4\00", align 1
@.str.89 = private constant [250 x i8] c"Assert fail: T.tvm_struct_get(dV_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(dV_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(dV_handle, 0, 7, \22uint16\22) == T.uint16(1), backward_op.dV_handle.dtype is expected to be float16\00", align 1
@.str.90 = private constant [191 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dV_handle_shape[0]) == 6, Argument backward_op.dV_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, backward_op_dV_handle_shape[0])\00", align 1
@.str.91 = private constant [193 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dV_handle_shape[1]) == 32, Argument backward_op.dV_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, backward_op_dV_handle_shape[1])\00", align 1
@.str.92 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dV_handle_shape[2]) == 512, Argument backward_op.dV_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, backward_op_dV_handle_shape[2])\00", align 1
@.str.93 = private constant [195 x i8] c"Assert fail: T.Cast(\22int32\22, backward_op_dV_handle_shape[3]) == 128, Argument backward_op.dV_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, backward_op_dV_handle_shape[3])\00", align 1
@.str.94 = private constant [318 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, backward_op_dV_handle_strides[3]) and 128 == T.Cast(\22int32\22, backward_op_dV_handle_strides[2]) and 65536 == T.Cast(\22int32\22, backward_op_dV_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, backward_op_dV_handle_strides[0]), backward_op.dV_handle.strides: expected to be compact array\00", align 1
@.str.95 = private constant [206 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(dV_handle, 0, 8, \22uint64\22), Argument backward_op.dV_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(dV_handle, 0, 8, \22uint64\22)\00", align 1
@.str.96 = private constant [186 x i8] c"Assert fail: T.tvm_struct_get(dV_handle, 0, 10, \22int32\22) == 8, Argument backward_op.dV_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(dV_handle, 0, 10, \22int32\22)\00", align 1
@.str.97 = private constant [192 x i8] c"Assert fail: dev_id == T.tvm_struct_get(dV_handle, 0, 9, \22int32\22), Argument backward_op.dV_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(dV_handle, 0, 9, \22int32\22)\00", align 1
@.str.98 = private constant [17 x i8] c"__tvm_set_device\00", align 1
@__TVMBackendAllocWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str.99 = private constant [19 x i8] c"backward_op_kernel\00", align 1
@.str.100 = private constant [21 x i8] c"backward_op_kernel_1\00", align 1
@.str.101 = private constant [21 x i8] c"backward_op_kernel_2\00", align 1
@__TVMBackendFreeWorkspace = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str.102 = private constant [61 x i8] c"Assert fail: num_args == 5, forward_op: num_args should be 5\00", align 1
@.str.103 = private constant [140 x i8] c"Assert fail: Q_handle_code == 3 or Q_handle_code == 13 or Q_handle_code == 7 or Q_handle_code == 4, forward_op: Expect arg[0] to be pointer\00", align 1
@.str.104 = private constant [140 x i8] c"Assert fail: K_handle_code == 3 or K_handle_code == 13 or K_handle_code == 7 or K_handle_code == 4, forward_op: Expect arg[1] to be pointer\00", align 1
@.str.105 = private constant [140 x i8] c"Assert fail: V_handle_code == 3 or V_handle_code == 13 or V_handle_code == 7 or V_handle_code == 4, forward_op: Expect arg[2] to be pointer\00", align 1
@.str.106 = private constant [140 x i8] c"Assert fail: O_handle_code == 3 or O_handle_code == 13 or O_handle_code == 7 or O_handle_code == 4, forward_op: Expect arg[3] to be pointer\00", align 1
@.str.107 = private constant [140 x i8] c"Assert fail: L_handle_code == 3 or L_handle_code == 13 or L_handle_code == 7 or L_handle_code == 4, forward_op: Expect arg[4] to be pointer\00", align 1
@.str.108 = private constant [109 x i8] c"Assert fail: 4 == T.tvm_struct_get(Q_handle, 0, 4, \22int32\22), forward_op.Q_handle.ndim is expected to equal 4\00", align 1
@.str.109 = private constant [245 x i8] c"Assert fail: T.tvm_struct_get(Q_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(Q_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(Q_handle, 0, 7, \22uint16\22) == T.uint16(1), forward_op.Q_handle.dtype is expected to be float16\00", align 1
@.str.110 = private constant [185 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_Q_handle_shape[0]) == 6, Argument forward_op.Q_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, forward_op_Q_handle_shape[0])\00", align 1
@.str.111 = private constant [187 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_Q_handle_shape[1]) == 32, Argument forward_op.Q_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, forward_op_Q_handle_shape[1])\00", align 1
@.str.112 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_Q_handle_shape[2]) == 512, Argument forward_op.Q_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, forward_op_Q_handle_shape[2])\00", align 1
@.str.113 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_Q_handle_shape[3]) == 128, Argument forward_op.Q_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, forward_op_Q_handle_shape[3])\00", align 1
@.str.114 = private constant [308 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, forward_op_Q_handle_strides[3]) and 128 == T.Cast(\22int32\22, forward_op_Q_handle_strides[2]) and 65536 == T.Cast(\22int32\22, forward_op_Q_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, forward_op_Q_handle_strides[0]), forward_op.Q_handle.strides: expected to be compact array\00", align 1
@.str.115 = private constant [202 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, \22uint64\22), Argument forward_op.Q_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, \22uint64\22)\00", align 1
@.str.116 = private constant [182 x i8] c"Assert fail: T.tvm_struct_get(Q_handle, 0, 10, \22int32\22) == 8, Argument forward_op.Q_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(Q_handle, 0, 10, \22int32\22)\00", align 1
@.str.117 = private constant [109 x i8] c"Assert fail: 4 == T.tvm_struct_get(K_handle, 0, 4, \22int32\22), forward_op.K_handle.ndim is expected to equal 4\00", align 1
@.str.118 = private constant [245 x i8] c"Assert fail: T.tvm_struct_get(K_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(K_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(K_handle, 0, 7, \22uint16\22) == T.uint16(1), forward_op.K_handle.dtype is expected to be float16\00", align 1
@.str.119 = private constant [185 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_K_handle_shape[0]) == 6, Argument forward_op.K_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, forward_op_K_handle_shape[0])\00", align 1
@.str.120 = private constant [187 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_K_handle_shape[1]) == 32, Argument forward_op.K_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, forward_op_K_handle_shape[1])\00", align 1
@.str.121 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_K_handle_shape[2]) == 512, Argument forward_op.K_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, forward_op_K_handle_shape[2])\00", align 1
@.str.122 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_K_handle_shape[3]) == 128, Argument forward_op.K_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, forward_op_K_handle_shape[3])\00", align 1
@.str.123 = private constant [308 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, forward_op_K_handle_strides[3]) and 128 == T.Cast(\22int32\22, forward_op_K_handle_strides[2]) and 65536 == T.Cast(\22int32\22, forward_op_K_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, forward_op_K_handle_strides[0]), forward_op.K_handle.strides: expected to be compact array\00", align 1
@.str.124 = private constant [202 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, \22uint64\22), Argument forward_op.K_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, \22uint64\22)\00", align 1
@.str.125 = private constant [182 x i8] c"Assert fail: T.tvm_struct_get(K_handle, 0, 10, \22int32\22) == 8, Argument forward_op.K_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(K_handle, 0, 10, \22int32\22)\00", align 1
@.str.126 = private constant [188 x i8] c"Assert fail: dev_id == T.tvm_struct_get(K_handle, 0, 9, \22int32\22), Argument forward_op.K_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(K_handle, 0, 9, \22int32\22)\00", align 1
@.str.127 = private constant [109 x i8] c"Assert fail: 4 == T.tvm_struct_get(V_handle, 0, 4, \22int32\22), forward_op.V_handle.ndim is expected to equal 4\00", align 1
@.str.128 = private constant [245 x i8] c"Assert fail: T.tvm_struct_get(V_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(V_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(V_handle, 0, 7, \22uint16\22) == T.uint16(1), forward_op.V_handle.dtype is expected to be float16\00", align 1
@.str.129 = private constant [185 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_V_handle_shape[0]) == 6, Argument forward_op.V_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, forward_op_V_handle_shape[0])\00", align 1
@.str.130 = private constant [187 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_V_handle_shape[1]) == 32, Argument forward_op.V_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, forward_op_V_handle_shape[1])\00", align 1
@.str.131 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_V_handle_shape[2]) == 512, Argument forward_op.V_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, forward_op_V_handle_shape[2])\00", align 1
@.str.132 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_V_handle_shape[3]) == 128, Argument forward_op.V_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, forward_op_V_handle_shape[3])\00", align 1
@.str.133 = private constant [308 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, forward_op_V_handle_strides[3]) and 128 == T.Cast(\22int32\22, forward_op_V_handle_strides[2]) and 65536 == T.Cast(\22int32\22, forward_op_V_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, forward_op_V_handle_strides[0]), forward_op.V_handle.strides: expected to be compact array\00", align 1
@.str.134 = private constant [202 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, \22uint64\22), Argument forward_op.V_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, \22uint64\22)\00", align 1
@.str.135 = private constant [182 x i8] c"Assert fail: T.tvm_struct_get(V_handle, 0, 10, \22int32\22) == 8, Argument forward_op.V_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(V_handle, 0, 10, \22int32\22)\00", align 1
@.str.136 = private constant [188 x i8] c"Assert fail: dev_id == T.tvm_struct_get(V_handle, 0, 9, \22int32\22), Argument forward_op.V_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(V_handle, 0, 9, \22int32\22)\00", align 1
@.str.137 = private constant [109 x i8] c"Assert fail: 4 == T.tvm_struct_get(O_handle, 0, 4, \22int32\22), forward_op.O_handle.ndim is expected to equal 4\00", align 1
@.str.138 = private constant [245 x i8] c"Assert fail: T.tvm_struct_get(O_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(O_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(O_handle, 0, 7, \22uint16\22) == T.uint16(1), forward_op.O_handle.dtype is expected to be float16\00", align 1
@.str.139 = private constant [185 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_O_handle_shape[0]) == 6, Argument forward_op.O_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, forward_op_O_handle_shape[0])\00", align 1
@.str.140 = private constant [187 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_O_handle_shape[1]) == 32, Argument forward_op.O_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, forward_op_O_handle_shape[1])\00", align 1
@.str.141 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_O_handle_shape[2]) == 512, Argument forward_op.O_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, forward_op_O_handle_shape[2])\00", align 1
@.str.142 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_O_handle_shape[3]) == 128, Argument forward_op.O_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\22int32\22, forward_op_O_handle_shape[3])\00", align 1
@.str.143 = private constant [308 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, forward_op_O_handle_strides[3]) and 128 == T.Cast(\22int32\22, forward_op_O_handle_strides[2]) and 65536 == T.Cast(\22int32\22, forward_op_O_handle_strides[1]) and 2097152 == T.Cast(\22int32\22, forward_op_O_handle_strides[0]), forward_op.O_handle.strides: expected to be compact array\00", align 1
@.str.144 = private constant [202 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(O_handle, 0, 8, \22uint64\22), Argument forward_op.O_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(O_handle, 0, 8, \22uint64\22)\00", align 1
@.str.145 = private constant [182 x i8] c"Assert fail: T.tvm_struct_get(O_handle, 0, 10, \22int32\22) == 8, Argument forward_op.O_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(O_handle, 0, 10, \22int32\22)\00", align 1
@.str.146 = private constant [188 x i8] c"Assert fail: dev_id == T.tvm_struct_get(O_handle, 0, 9, \22int32\22), Argument forward_op.O_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(O_handle, 0, 9, \22int32\22)\00", align 1
@.str.147 = private constant [109 x i8] c"Assert fail: 3 == T.tvm_struct_get(L_handle, 0, 4, \22int32\22), forward_op.L_handle.ndim is expected to equal 3\00", align 1
@.str.148 = private constant [245 x i8] c"Assert fail: T.tvm_struct_get(L_handle, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(L_handle, 0, 6, \22uint8\22) == T.uint8(16) and T.tvm_struct_get(L_handle, 0, 7, \22uint16\22) == T.uint16(1), forward_op.L_handle.dtype is expected to be float16\00", align 1
@.str.149 = private constant [185 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_L_handle_shape[0]) == 6, Argument forward_op.L_handle.shape[0] has an unsatisfied constraint: 6 == T.Cast(\22int32\22, forward_op_L_handle_shape[0])\00", align 1
@.str.150 = private constant [187 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_L_handle_shape[1]) == 32, Argument forward_op.L_handle.shape[1] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, forward_op_L_handle_shape[1])\00", align 1
@.str.151 = private constant [189 x i8] c"Assert fail: T.Cast(\22int32\22, forward_op_L_handle_shape[2]) == 512, Argument forward_op.L_handle.shape[2] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, forward_op_L_handle_shape[2])\00", align 1
@.str.152 = private constant [245 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, forward_op_L_handle_strides[2]) and 512 == T.Cast(\22int32\22, forward_op_L_handle_strides[1]) and 16384 == T.Cast(\22int32\22, forward_op_L_handle_strides[0]), forward_op.L_handle.strides: expected to be compact array\00", align 1
@.str.153 = private constant [202 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(L_handle, 0, 8, \22uint64\22), Argument forward_op.L_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(L_handle, 0, 8, \22uint64\22)\00", align 1
@.str.154 = private constant [182 x i8] c"Assert fail: T.tvm_struct_get(L_handle, 0, 10, \22int32\22) == 8, Argument forward_op.L_handle.device_type has an unsatisfied constraint: 8 == T.tvm_struct_get(L_handle, 0, 10, \22int32\22)\00", align 1
@.str.155 = private constant [188 x i8] c"Assert fail: dev_id == T.tvm_struct_get(L_handle, 0, 9, \22int32\22), Argument forward_op.L_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(L_handle, 0, 9, \22int32\22)\00", align 1
@.str.156 = private constant [18 x i8] c"forward_op_kernel\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer
@_MergedGlobals = internal global <{ ptr, ptr, ptr, ptr, ptr }> zeroinitializer, align 8

define dllexport i32 @backward_op(ptr nocapture readonly %args, ptr nocapture readonly %arg_type_ids, i32 %num_args, ptr nocapture readnone %out_ret_value, ptr nocapture readnone %out_ret_tcode, ptr nocapture readnone %resource_handle) local_unnamed_addr #0 !dbg !5 {
entry:
  %0 = alloca ptr, align 8
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  call void @llvm.dbg.value(metadata ptr %args, metadata !12, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata ptr %arg_type_ids, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata ptr %out_ret_value, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata ptr %out_ret_tcode, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata ptr %resource_handle, metadata !17, metadata !DIExpression()), !dbg !18
  %3 = alloca ptr, align 8, !dbg !18
  %stack_value235 = alloca [14 x %0], align 8, !dbg !18
  %stack_tcode236 = alloca [14 x i32], align 8, !dbg !18
  %4 = icmp eq i32 %num_args, 9, !dbg !18
  br i1 %4, label %assert_end, label %assert_fail, !dbg !18, !prof !19

common.ret:                                       ; preds = %backward_op_compute_.exit, %handle_init_end, %handle_init, %assert_fail227, %assert_fail225, %assert_fail223, %assert_fail221, %assert_fail217, %assert_fail215, %assert_fail213, %assert_fail211, %assert_fail209, %assert_fail205, %assert_fail203, %assert_fail201, %assert_fail199, %assert_fail197, %assert_fail193, %assert_fail191, %assert_fail189, %assert_fail187, %assert_fail185, %assert_fail181, %assert_fail179, %assert_fail177, %assert_fail175, %assert_fail173, %assert_fail169, %assert_fail167, %assert_fail165, %assert_fail163, %assert_fail161, %assert_fail157, %assert_fail155, %assert_fail153, %assert_fail151, %assert_fail149, %assert_fail145, %assert_fail143, %assert_fail141, %assert_fail139, %assert_fail135, %assert_fail133, %assert_fail131, %assert_fail129, %assert_fail127, %assert_fail123, %assert_fail121, %assert_fail119, %assert_fail117, %assert_fail115, %assert_fail111, %assert_fail109, %assert_fail107, %assert_fail105, %assert_fail103, %assert_fail99, %assert_fail97, %assert_fail95, %assert_fail93, %assert_fail91, %assert_fail87, %assert_fail85, %assert_fail83, %assert_fail81, %assert_fail79, %assert_fail75, %assert_fail73, %assert_fail71, %assert_fail69, %assert_fail67, %assert_fail63, %assert_fail61, %assert_fail59, %assert_fail57, %assert_fail55, %assert_fail51, %assert_fail49, %assert_fail47, %assert_fail45, %assert_fail43, %assert_fail39, %assert_fail37, %assert_fail35, %assert_fail33, %assert_fail31, %assert_fail29, %assert_fail27, %assert_fail25, %assert_fail23, %assert_fail19, %assert_fail17, %assert_fail15, %assert_fail13, %assert_fail11, %assert_fail9, %assert_fail7, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail7 ], [ -1, %assert_fail9 ], [ -1, %assert_fail11 ], [ -1, %assert_fail13 ], [ -1, %assert_fail15 ], [ -1, %assert_fail17 ], [ -1, %assert_fail19 ], [ -1, %assert_fail23 ], [ -1, %assert_fail25 ], [ -1, %assert_fail27 ], [ -1, %assert_fail29 ], [ -1, %assert_fail31 ], [ -1, %assert_fail33 ], [ -1, %assert_fail35 ], [ -1, %assert_fail37 ], [ -1, %assert_fail39 ], [ -1, %assert_fail43 ], [ -1, %assert_fail45 ], [ -1, %assert_fail47 ], [ -1, %assert_fail49 ], [ -1, %assert_fail51 ], [ -1, %assert_fail55 ], [ -1, %assert_fail57 ], [ -1, %assert_fail59 ], [ -1, %assert_fail61 ], [ -1, %assert_fail63 ], [ -1, %assert_fail67 ], [ -1, %assert_fail69 ], [ -1, %assert_fail71 ], [ -1, %assert_fail73 ], [ -1, %assert_fail75 ], [ -1, %assert_fail79 ], [ -1, %assert_fail81 ], [ -1, %assert_fail83 ], [ -1, %assert_fail85 ], [ -1, %assert_fail87 ], [ -1, %assert_fail91 ], [ -1, %assert_fail93 ], [ -1, %assert_fail95 ], [ -1, %assert_fail97 ], [ -1, %assert_fail99 ], [ -1, %assert_fail103 ], [ -1, %assert_fail105 ], [ -1, %assert_fail107 ], [ -1, %assert_fail109 ], [ -1, %assert_fail111 ], [ -1, %assert_fail115 ], [ -1, %assert_fail117 ], [ -1, %assert_fail119 ], [ -1, %assert_fail121 ], [ -1, %assert_fail123 ], [ -1, %assert_fail127 ], [ -1, %assert_fail129 ], [ -1, %assert_fail131 ], [ -1, %assert_fail133 ], [ -1, %assert_fail135 ], [ -1, %assert_fail139 ], [ -1, %assert_fail141 ], [ -1, %assert_fail143 ], [ -1, %assert_fail145 ], [ -1, %assert_fail149 ], [ -1, %assert_fail151 ], [ -1, %assert_fail153 ], [ -1, %assert_fail155 ], [ -1, %assert_fail157 ], [ -1, %assert_fail161 ], [ -1, %assert_fail163 ], [ -1, %assert_fail165 ], [ -1, %assert_fail167 ], [ -1, %assert_fail169 ], [ -1, %assert_fail173 ], [ -1, %assert_fail175 ], [ -1, %assert_fail177 ], [ -1, %assert_fail179 ], [ -1, %assert_fail181 ], [ -1, %assert_fail185 ], [ -1, %assert_fail187 ], [ -1, %assert_fail189 ], [ -1, %assert_fail191 ], [ -1, %assert_fail193 ], [ -1, %assert_fail197 ], [ -1, %assert_fail199 ], [ -1, %assert_fail201 ], [ -1, %assert_fail203 ], [ -1, %assert_fail205 ], [ -1, %assert_fail209 ], [ -1, %assert_fail211 ], [ -1, %assert_fail213 ], [ -1, %assert_fail215 ], [ -1, %assert_fail217 ], [ -1, %assert_fail221 ], [ -1, %assert_fail223 ], [ -1, %assert_fail225 ], [ -1, %assert_fail227 ], [ %542, %handle_init ], [ %545, %handle_init_end ], [ %common.ret.op.i, %backward_op_compute_.exit ]
  ret i32 %common.ret.op, !dbg !18

assert_fail:                                      ; preds = %entry
  %5 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %5(ptr nonnull @.str), !dbg !18
  br label %common.ret

assert_end:                                       ; preds = %entry
  %dO_handle.code = load i32, ptr %arg_type_ids, align 4, !dbg !18, !tbaa !23
  %6 = getelementptr inbounds i32, ptr %arg_type_ids, i64 1, !dbg !18
  %Q_handle.code = load i32, ptr %6, align 4, !dbg !18, !tbaa !34
  %7 = getelementptr inbounds i32, ptr %arg_type_ids, i64 2, !dbg !18
  %K_handle.code = load i32, ptr %7, align 4, !dbg !18, !tbaa !36
  %8 = getelementptr inbounds i32, ptr %arg_type_ids, i64 3, !dbg !18
  %V_handle.code = load i32, ptr %8, align 4, !dbg !18, !tbaa !39
  %9 = getelementptr inbounds i32, ptr %arg_type_ids, i64 4, !dbg !18
  %O_handle.code = load i32, ptr %9, align 4, !dbg !18, !tbaa !41
  %10 = getelementptr inbounds i32, ptr %arg_type_ids, i64 5, !dbg !18
  %L_handle.code = load i32, ptr %10, align 4, !dbg !18, !tbaa !45
  %11 = getelementptr inbounds i32, ptr %arg_type_ids, i64 6, !dbg !18
  %dQ_handle.code = load i32, ptr %11, align 4, !dbg !18, !tbaa !47
  %12 = getelementptr inbounds i32, ptr %arg_type_ids, i64 7, !dbg !18
  %dK_handle.code = load i32, ptr %12, align 4, !dbg !18, !tbaa !50
  %13 = getelementptr inbounds i32, ptr %arg_type_ids, i64 8, !dbg !18
  %dV_handle.code = load i32, ptr %13, align 4, !dbg !18, !tbaa !52
  %dO_handle = load ptr, ptr %args, align 8, !dbg !18
  %14 = getelementptr inbounds %0, ptr %args, i64 1, !dbg !18
  %Q_handle = load ptr, ptr %14, align 8, !dbg !18
  %15 = getelementptr inbounds %0, ptr %args, i64 2, !dbg !18
  %K_handle = load ptr, ptr %15, align 8, !dbg !18
  %16 = getelementptr inbounds %0, ptr %args, i64 3, !dbg !18
  %V_handle = load ptr, ptr %16, align 8, !dbg !18
  %17 = getelementptr inbounds %0, ptr %args, i64 4, !dbg !18
  %O_handle = load ptr, ptr %17, align 8, !dbg !18
  %18 = getelementptr inbounds %0, ptr %args, i64 5, !dbg !18
  %L_handle = load ptr, ptr %18, align 8, !dbg !18
  %19 = getelementptr inbounds %0, ptr %args, i64 6, !dbg !18
  %dQ_handle = load ptr, ptr %19, align 8, !dbg !18
  %20 = getelementptr inbounds %0, ptr %args, i64 7, !dbg !18
  %dK_handle = load ptr, ptr %20, align 8, !dbg !18
  %21 = getelementptr inbounds %0, ptr %args, i64 8, !dbg !18
  %dV_handle = load ptr, ptr %21, align 8, !dbg !18
  %dO = load ptr, ptr %dO_handle, align 8, !dbg !18
  %22 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 4, !dbg !18
  %backward_op.dO_handle.shape = load ptr, ptr %22, align 8, !dbg !18
  %23 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 5, !dbg !18
  %backward_op.dO_handle.strides = load ptr, ptr %23, align 8, !dbg !18
  %24 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 1, i32 1, !dbg !18
  %dev_id = load i32, ptr %24, align 4, !dbg !18
  %25 = sext i32 %dev_id to i64, !dbg !18
  %26 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 4, !dbg !18
  %27 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 5, !dbg !18
  %28 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 4, !dbg !18
  %29 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 5, !dbg !18
  %30 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 4, !dbg !18
  %31 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 5, !dbg !18
  %32 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 4, !dbg !18
  %33 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 5, !dbg !18
  %Q = load ptr, ptr %Q_handle, align 8, !dbg !18
  %backward_op.Q_handle.shape = load ptr, ptr %26, align 8, !dbg !18
  %backward_op.Q_handle.strides = load ptr, ptr %27, align 8, !dbg !18
  %K = load ptr, ptr %K_handle, align 8, !dbg !18
  %backward_op.K_handle.shape = load ptr, ptr %28, align 8, !dbg !18
  %backward_op.K_handle.strides = load ptr, ptr %29, align 8, !dbg !18
  %V = load ptr, ptr %V_handle, align 8, !dbg !18
  %backward_op.V_handle.shape = load ptr, ptr %30, align 8, !dbg !18
  %backward_op.V_handle.strides = load ptr, ptr %31, align 8, !dbg !18
  %O = load ptr, ptr %O_handle, align 8, !dbg !18
  %backward_op.O_handle.shape = load ptr, ptr %32, align 8, !dbg !18
  %backward_op.O_handle.strides = load ptr, ptr %33, align 8, !dbg !18
  %L = load ptr, ptr %L_handle, align 8, !dbg !18
  %34 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 4, !dbg !18
  %backward_op.L_handle.shape = load ptr, ptr %34, align 8, !dbg !18
  %35 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 5, !dbg !18
  %backward_op.L_handle.strides = load ptr, ptr %35, align 8, !dbg !18
  %dQ = load ptr, ptr %dQ_handle, align 8, !dbg !18
  %36 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 4, !dbg !18
  %backward_op.dQ_handle.shape = load ptr, ptr %36, align 8, !dbg !18
  %37 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 5, !dbg !18
  %backward_op.dQ_handle.strides = load ptr, ptr %37, align 8, !dbg !18
  %dK = load ptr, ptr %dK_handle, align 8, !dbg !18
  %38 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 4, !dbg !18
  %backward_op.dK_handle.shape = load ptr, ptr %38, align 8, !dbg !18
  %39 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 5, !dbg !18
  %backward_op.dK_handle.strides = load ptr, ptr %39, align 8, !dbg !18
  %dV = load ptr, ptr %dV_handle, align 8, !dbg !18
  %40 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 4, !dbg !18
  %backward_op.dV_handle.shape = load ptr, ptr %40, align 8, !dbg !18
  %41 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 5, !dbg !18
  %backward_op.dV_handle.strides = load ptr, ptr %41, align 8, !dbg !18
  switch i32 %dO_handle.code, label %assert_fail1 [
    i32 13, label %assert_end2
    i32 7, label %assert_end2
    i32 4, label %assert_end2
    i32 3, label %assert_end2
  ], !dbg !18

assert_fail1:                                     ; preds = %assert_end
  %42 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %42(ptr nonnull @.str.1), !dbg !18
  br label %common.ret

assert_end2:                                      ; preds = %assert_end, %assert_end, %assert_end, %assert_end
  switch i32 %Q_handle.code, label %assert_fail3 [
    i32 13, label %assert_end4
    i32 7, label %assert_end4
    i32 4, label %assert_end4
    i32 3, label %assert_end4
  ], !dbg !18

assert_fail3:                                     ; preds = %assert_end2
  %43 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %43(ptr nonnull @.str.2), !dbg !18
  br label %common.ret

assert_end4:                                      ; preds = %assert_end2, %assert_end2, %assert_end2, %assert_end2
  switch i32 %K_handle.code, label %assert_fail5 [
    i32 13, label %assert_end6
    i32 7, label %assert_end6
    i32 4, label %assert_end6
    i32 3, label %assert_end6
  ], !dbg !18

assert_fail5:                                     ; preds = %assert_end4
  %44 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %44(ptr nonnull @.str.3), !dbg !18
  br label %common.ret

assert_end6:                                      ; preds = %assert_end4, %assert_end4, %assert_end4, %assert_end4
  switch i32 %V_handle.code, label %assert_fail7 [
    i32 13, label %assert_end8
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 3, label %assert_end8
  ], !dbg !18

assert_fail7:                                     ; preds = %assert_end6
  %45 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %45(ptr nonnull @.str.4), !dbg !18
  br label %common.ret

assert_end8:                                      ; preds = %assert_end6, %assert_end6, %assert_end6, %assert_end6
  switch i32 %O_handle.code, label %assert_fail9 [
    i32 13, label %assert_end10
    i32 7, label %assert_end10
    i32 4, label %assert_end10
    i32 3, label %assert_end10
  ], !dbg !18

assert_fail9:                                     ; preds = %assert_end8
  %46 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %46(ptr nonnull @.str.5), !dbg !18
  br label %common.ret

assert_end10:                                     ; preds = %assert_end8, %assert_end8, %assert_end8, %assert_end8
  switch i32 %L_handle.code, label %assert_fail11 [
    i32 13, label %assert_end12
    i32 7, label %assert_end12
    i32 4, label %assert_end12
    i32 3, label %assert_end12
  ], !dbg !18

assert_fail11:                                    ; preds = %assert_end10
  %47 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %47(ptr nonnull @.str.6), !dbg !18
  br label %common.ret

assert_end12:                                     ; preds = %assert_end10, %assert_end10, %assert_end10, %assert_end10
  switch i32 %dQ_handle.code, label %assert_fail13 [
    i32 13, label %assert_end14
    i32 7, label %assert_end14
    i32 4, label %assert_end14
    i32 3, label %assert_end14
  ], !dbg !18

assert_fail13:                                    ; preds = %assert_end12
  %48 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %48(ptr nonnull @.str.7), !dbg !18
  br label %common.ret

assert_end14:                                     ; preds = %assert_end12, %assert_end12, %assert_end12, %assert_end12
  switch i32 %dK_handle.code, label %assert_fail15 [
    i32 13, label %assert_end16
    i32 7, label %assert_end16
    i32 4, label %assert_end16
    i32 3, label %assert_end16
  ], !dbg !18

assert_fail15:                                    ; preds = %assert_end14
  %49 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %49(ptr nonnull @.str.8), !dbg !18
  br label %common.ret

assert_end16:                                     ; preds = %assert_end14, %assert_end14, %assert_end14, %assert_end14
  switch i32 %dV_handle.code, label %assert_fail17 [
    i32 13, label %assert_end18
    i32 7, label %assert_end18
    i32 4, label %assert_end18
    i32 3, label %assert_end18
  ], !dbg !18

assert_fail17:                                    ; preds = %assert_end16
  %50 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %50(ptr nonnull @.str.9), !dbg !18
  br label %common.ret

assert_end18:                                     ; preds = %assert_end16, %assert_end16, %assert_end16, %assert_end16
  %51 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 2, !dbg !18
  %52 = load i32, ptr %51, align 4, !dbg !18
  %53 = icmp eq i32 %52, 4, !dbg !18
  br i1 %53, label %assert_end22, label %assert_fail19, !dbg !18, !prof !19

assert_fail19:                                    ; preds = %assert_end18
  %54 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %54(ptr nonnull @.str.10), !dbg !18
  br label %common.ret

assert_end22:                                     ; preds = %assert_end18
  %55 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 3, i32 0, !dbg !18
  %56 = load i8, ptr %55, align 1, !dbg !18
  %57 = icmp eq i8 %56, 2, !dbg !18
  %58 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 3, i32 1, !dbg !18
  %59 = load i8, ptr %58, align 1, !dbg !18
  %60 = icmp eq i8 %59, 16, !dbg !18
  %61 = and i1 %57, %60, !dbg !18
  %62 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 3, i32 2, !dbg !18
  %63 = load i16, ptr %62, align 2, !dbg !18
  %64 = icmp eq i16 %63, 1, !dbg !18
  %65 = and i1 %61, %64, !dbg !18
  br i1 %65, label %assert_end24, label %assert_fail23, !dbg !18, !prof !19

assert_fail23:                                    ; preds = %assert_end22
  %66 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %66(ptr nonnull @.str.11), !dbg !18
  br label %common.ret

assert_end24:                                     ; preds = %assert_end22
  %67 = load i64, ptr %backward_op.dO_handle.shape, align 8, !dbg !18, !tbaa !57
  %68 = and i64 %67, 4294967295, !dbg !18
  %69 = icmp eq i64 %68, 6, !dbg !18
  br i1 %69, label %assert_end26, label %assert_fail25, !dbg !18, !prof !19

assert_fail25:                                    ; preds = %assert_end24
  %70 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %70(ptr nonnull @.str.12), !dbg !18
  br label %common.ret

assert_end26:                                     ; preds = %assert_end24
  %71 = getelementptr inbounds i64, ptr %backward_op.dO_handle.shape, i64 1, !dbg !18
  %72 = load i64, ptr %71, align 8, !dbg !18, !tbaa !57
  %73 = and i64 %72, 4294967295, !dbg !18
  %74 = icmp eq i64 %73, 32, !dbg !18
  br i1 %74, label %assert_end28, label %assert_fail27, !dbg !18, !prof !19

assert_fail27:                                    ; preds = %assert_end26
  %75 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %75(ptr nonnull @.str.13), !dbg !18
  br label %common.ret

assert_end28:                                     ; preds = %assert_end26
  %76 = getelementptr inbounds i64, ptr %backward_op.dO_handle.shape, i64 2, !dbg !18
  %77 = load i64, ptr %76, align 8, !dbg !18, !tbaa !57
  %78 = and i64 %77, 4294967295, !dbg !18
  %79 = icmp eq i64 %78, 512, !dbg !18
  br i1 %79, label %assert_end30, label %assert_fail29, !dbg !18, !prof !19

assert_fail29:                                    ; preds = %assert_end28
  %80 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %80(ptr nonnull @.str.14), !dbg !18
  br label %common.ret

assert_end30:                                     ; preds = %assert_end28
  %81 = getelementptr inbounds i64, ptr %backward_op.dO_handle.shape, i64 3, !dbg !18
  %82 = load i64, ptr %81, align 8, !dbg !18, !tbaa !57
  %83 = and i64 %82, 4294967295, !dbg !18
  %84 = icmp eq i64 %83, 128, !dbg !18
  br i1 %84, label %assert_end32, label %assert_fail31, !dbg !18, !prof !19

assert_fail31:                                    ; preds = %assert_end30
  %85 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %85(ptr nonnull @.str.15), !dbg !18
  br label %common.ret

assert_end32:                                     ; preds = %assert_end30
  %.not = icmp eq ptr %backward_op.dO_handle.strides, null, !dbg !18
  br i1 %.not, label %if_end, label %if_then, !dbg !18, !prof !59

if_then:                                          ; preds = %assert_end32
  %86 = load <4 x i64>, ptr %backward_op.dO_handle.strides, align 8, !dbg !18, !tbaa !57
  %87 = and <4 x i64> %86, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %88 = icmp ne <4 x i64> %87, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %89 = bitcast <4 x i1> %88 to i4, !dbg !18
  %90 = icmp eq i4 %89, 0, !dbg !18
  br i1 %90, label %if_end, label %assert_fail33, !dbg !18, !prof !19

if_end:                                           ; preds = %if_then, %assert_end32
  %91 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 6, !dbg !18
  %92 = load i64, ptr %91, align 8, !dbg !18
  %93 = icmp eq i64 %92, 0, !dbg !18
  br i1 %93, label %assert_end36, label %assert_fail35, !dbg !18, !prof !19

assert_fail33:                                    ; preds = %if_then
  %94 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %94(ptr nonnull @.str.16), !dbg !18
  br label %common.ret

assert_fail35:                                    ; preds = %if_end
  %95 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %95(ptr nonnull @.str.17), !dbg !18
  br label %common.ret

assert_end36:                                     ; preds = %if_end
  %96 = getelementptr inbounds %1, ptr %dO_handle, i64 0, i32 1, i32 0, !dbg !18
  %97 = load i32, ptr %96, align 4, !dbg !18
  %98 = icmp eq i32 %97, 8, !dbg !18
  br i1 %98, label %assert_end38, label %assert_fail37, !dbg !18, !prof !19

assert_fail37:                                    ; preds = %assert_end36
  %99 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %99(ptr nonnull @.str.18), !dbg !18
  br label %common.ret

assert_end38:                                     ; preds = %assert_end36
  %100 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 2, !dbg !18
  %101 = load i32, ptr %100, align 4, !dbg !18
  %102 = icmp eq i32 %101, 4, !dbg !18
  br i1 %102, label %assert_end42, label %assert_fail39, !dbg !18, !prof !19

assert_fail39:                                    ; preds = %assert_end38
  %103 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %103(ptr nonnull @.str.19), !dbg !18
  br label %common.ret

assert_end42:                                     ; preds = %assert_end38
  %104 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 0, !dbg !18
  %105 = load i8, ptr %104, align 1, !dbg !18
  %106 = icmp eq i8 %105, 2, !dbg !18
  %107 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 1, !dbg !18
  %108 = load i8, ptr %107, align 1, !dbg !18
  %109 = icmp eq i8 %108, 16, !dbg !18
  %110 = and i1 %106, %109, !dbg !18
  %111 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 2, !dbg !18
  %112 = load i16, ptr %111, align 2, !dbg !18
  %113 = icmp eq i16 %112, 1, !dbg !18
  %114 = and i1 %110, %113, !dbg !18
  br i1 %114, label %assert_end44, label %assert_fail43, !dbg !18, !prof !19

assert_fail43:                                    ; preds = %assert_end42
  %115 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %115(ptr nonnull @.str.20), !dbg !18
  br label %common.ret

assert_end44:                                     ; preds = %assert_end42
  %116 = load i64, ptr %backward_op.Q_handle.shape, align 8, !dbg !18, !tbaa !57
  %117 = and i64 %116, 4294967295, !dbg !18
  %118 = icmp eq i64 %117, 6, !dbg !18
  br i1 %118, label %assert_end46, label %assert_fail45, !dbg !18, !prof !19

assert_fail45:                                    ; preds = %assert_end44
  %119 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %119(ptr nonnull @.str.21), !dbg !18
  br label %common.ret

assert_end46:                                     ; preds = %assert_end44
  %120 = getelementptr inbounds i64, ptr %backward_op.Q_handle.shape, i64 1, !dbg !18
  %121 = load i64, ptr %120, align 8, !dbg !18, !tbaa !57
  %122 = and i64 %121, 4294967295, !dbg !18
  %123 = icmp eq i64 %122, 32, !dbg !18
  br i1 %123, label %assert_end48, label %assert_fail47, !dbg !18, !prof !19

assert_fail47:                                    ; preds = %assert_end46
  %124 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %124(ptr nonnull @.str.22), !dbg !18
  br label %common.ret

assert_end48:                                     ; preds = %assert_end46
  %125 = getelementptr inbounds i64, ptr %backward_op.Q_handle.shape, i64 2, !dbg !18
  %126 = load i64, ptr %125, align 8, !dbg !18, !tbaa !57
  %127 = and i64 %126, 4294967295, !dbg !18
  %128 = icmp eq i64 %127, 512, !dbg !18
  br i1 %128, label %assert_end50, label %assert_fail49, !dbg !18, !prof !19

assert_fail49:                                    ; preds = %assert_end48
  %129 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %129(ptr nonnull @.str.23), !dbg !18
  br label %common.ret

assert_end50:                                     ; preds = %assert_end48
  %130 = getelementptr inbounds i64, ptr %backward_op.Q_handle.shape, i64 3, !dbg !18
  %131 = load i64, ptr %130, align 8, !dbg !18, !tbaa !57
  %132 = and i64 %131, 4294967295, !dbg !18
  %133 = icmp eq i64 %132, 128, !dbg !18
  br i1 %133, label %assert_end52, label %assert_fail51, !dbg !18, !prof !19

assert_fail51:                                    ; preds = %assert_end50
  %134 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %134(ptr nonnull @.str.24), !dbg !18
  br label %common.ret

assert_end52:                                     ; preds = %assert_end50
  %.not237 = icmp eq ptr %backward_op.Q_handle.strides, null, !dbg !18
  br i1 %.not237, label %if_end54, label %if_then53, !dbg !18, !prof !59

if_then53:                                        ; preds = %assert_end52
  %135 = load <4 x i64>, ptr %backward_op.Q_handle.strides, align 8, !dbg !18, !tbaa !57
  %136 = and <4 x i64> %135, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %137 = icmp ne <4 x i64> %136, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %138 = bitcast <4 x i1> %137 to i4, !dbg !18
  %139 = icmp eq i4 %138, 0, !dbg !18
  br i1 %139, label %if_end54, label %assert_fail55, !dbg !18, !prof !19

if_end54:                                         ; preds = %if_then53, %assert_end52
  %140 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 6, !dbg !18
  %141 = load i64, ptr %140, align 8, !dbg !18
  %142 = icmp eq i64 %141, 0, !dbg !18
  br i1 %142, label %assert_end58, label %assert_fail57, !dbg !18, !prof !19

assert_fail55:                                    ; preds = %if_then53
  %143 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %143(ptr nonnull @.str.25), !dbg !18
  br label %common.ret

assert_fail57:                                    ; preds = %if_end54
  %144 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %144(ptr nonnull @.str.26), !dbg !18
  br label %common.ret

assert_end58:                                     ; preds = %if_end54
  %145 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 0, !dbg !18
  %146 = load i32, ptr %145, align 4, !dbg !18
  %147 = icmp eq i32 %146, 8, !dbg !18
  br i1 %147, label %assert_end60, label %assert_fail59, !dbg !18, !prof !19

assert_fail59:                                    ; preds = %assert_end58
  %148 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %148(ptr nonnull @.str.27), !dbg !18
  br label %common.ret

assert_end60:                                     ; preds = %assert_end58
  %149 = trunc i64 %25 to i32
  %150 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 1, !dbg !18
  %151 = load i32, ptr %150, align 4, !dbg !18
  %152 = icmp eq i32 %149, %151, !dbg !18
  br i1 %152, label %assert_end62, label %assert_fail61, !dbg !18, !prof !19

assert_fail61:                                    ; preds = %assert_end60
  %153 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %153(ptr nonnull @.str.28), !dbg !18
  br label %common.ret

assert_end62:                                     ; preds = %assert_end60
  %154 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 2, !dbg !18
  %155 = load i32, ptr %154, align 4, !dbg !18
  %156 = icmp eq i32 %155, 4, !dbg !18
  br i1 %156, label %assert_end66, label %assert_fail63, !dbg !18, !prof !19

assert_fail63:                                    ; preds = %assert_end62
  %157 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %157(ptr nonnull @.str.29), !dbg !18
  br label %common.ret

assert_end66:                                     ; preds = %assert_end62
  %158 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 0, !dbg !18
  %159 = load i8, ptr %158, align 1, !dbg !18
  %160 = icmp eq i8 %159, 2, !dbg !18
  %161 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 1, !dbg !18
  %162 = load i8, ptr %161, align 1, !dbg !18
  %163 = icmp eq i8 %162, 16, !dbg !18
  %164 = and i1 %160, %163, !dbg !18
  %165 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 2, !dbg !18
  %166 = load i16, ptr %165, align 2, !dbg !18
  %167 = icmp eq i16 %166, 1, !dbg !18
  %168 = and i1 %164, %167, !dbg !18
  br i1 %168, label %assert_end68, label %assert_fail67, !dbg !18, !prof !19

assert_fail67:                                    ; preds = %assert_end66
  %169 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %169(ptr nonnull @.str.30), !dbg !18
  br label %common.ret

assert_end68:                                     ; preds = %assert_end66
  %170 = load i64, ptr %backward_op.K_handle.shape, align 8, !dbg !18, !tbaa !57
  %171 = and i64 %170, 4294967295, !dbg !18
  %172 = icmp eq i64 %171, 6, !dbg !18
  br i1 %172, label %assert_end70, label %assert_fail69, !dbg !18, !prof !19

assert_fail69:                                    ; preds = %assert_end68
  %173 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %173(ptr nonnull @.str.31), !dbg !18
  br label %common.ret

assert_end70:                                     ; preds = %assert_end68
  %174 = getelementptr inbounds i64, ptr %backward_op.K_handle.shape, i64 1, !dbg !18
  %175 = load i64, ptr %174, align 8, !dbg !18, !tbaa !57
  %176 = and i64 %175, 4294967295, !dbg !18
  %177 = icmp eq i64 %176, 32, !dbg !18
  br i1 %177, label %assert_end72, label %assert_fail71, !dbg !18, !prof !19

assert_fail71:                                    ; preds = %assert_end70
  %178 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %178(ptr nonnull @.str.32), !dbg !18
  br label %common.ret

assert_end72:                                     ; preds = %assert_end70
  %179 = getelementptr inbounds i64, ptr %backward_op.K_handle.shape, i64 2, !dbg !18
  %180 = load i64, ptr %179, align 8, !dbg !18, !tbaa !57
  %181 = and i64 %180, 4294967295, !dbg !18
  %182 = icmp eq i64 %181, 512, !dbg !18
  br i1 %182, label %assert_end74, label %assert_fail73, !dbg !18, !prof !19

assert_fail73:                                    ; preds = %assert_end72
  %183 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %183(ptr nonnull @.str.33), !dbg !18
  br label %common.ret

assert_end74:                                     ; preds = %assert_end72
  %184 = getelementptr inbounds i64, ptr %backward_op.K_handle.shape, i64 3, !dbg !18
  %185 = load i64, ptr %184, align 8, !dbg !18, !tbaa !57
  %186 = and i64 %185, 4294967295, !dbg !18
  %187 = icmp eq i64 %186, 128, !dbg !18
  br i1 %187, label %assert_end76, label %assert_fail75, !dbg !18, !prof !19

assert_fail75:                                    ; preds = %assert_end74
  %188 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %188(ptr nonnull @.str.34), !dbg !18
  br label %common.ret

assert_end76:                                     ; preds = %assert_end74
  %.not238 = icmp eq ptr %backward_op.K_handle.strides, null, !dbg !18
  br i1 %.not238, label %if_end78, label %if_then77, !dbg !18, !prof !59

if_then77:                                        ; preds = %assert_end76
  %189 = load <4 x i64>, ptr %backward_op.K_handle.strides, align 8, !dbg !18, !tbaa !57
  %190 = and <4 x i64> %189, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %191 = icmp ne <4 x i64> %190, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %192 = bitcast <4 x i1> %191 to i4, !dbg !18
  %193 = icmp eq i4 %192, 0, !dbg !18
  br i1 %193, label %if_end78, label %assert_fail79, !dbg !18, !prof !19

if_end78:                                         ; preds = %if_then77, %assert_end76
  %194 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 6, !dbg !18
  %195 = load i64, ptr %194, align 8, !dbg !18
  %196 = icmp eq i64 %195, 0, !dbg !18
  br i1 %196, label %assert_end82, label %assert_fail81, !dbg !18, !prof !19

assert_fail79:                                    ; preds = %if_then77
  %197 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %197(ptr nonnull @.str.35), !dbg !18
  br label %common.ret

assert_fail81:                                    ; preds = %if_end78
  %198 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %198(ptr nonnull @.str.36), !dbg !18
  br label %common.ret

assert_end82:                                     ; preds = %if_end78
  %199 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 0, !dbg !18
  %200 = load i32, ptr %199, align 4, !dbg !18
  %201 = icmp eq i32 %200, 8, !dbg !18
  br i1 %201, label %assert_end84, label %assert_fail83, !dbg !18, !prof !19

assert_fail83:                                    ; preds = %assert_end82
  %202 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %202(ptr nonnull @.str.37), !dbg !18
  br label %common.ret

assert_end84:                                     ; preds = %assert_end82
  %203 = trunc i64 %25 to i32
  %204 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 1, !dbg !18
  %205 = load i32, ptr %204, align 4, !dbg !18
  %206 = icmp eq i32 %203, %205, !dbg !18
  br i1 %206, label %assert_end86, label %assert_fail85, !dbg !18, !prof !19

assert_fail85:                                    ; preds = %assert_end84
  %207 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %207(ptr nonnull @.str.38), !dbg !18
  br label %common.ret

assert_end86:                                     ; preds = %assert_end84
  %208 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 2, !dbg !18
  %209 = load i32, ptr %208, align 4, !dbg !18
  %210 = icmp eq i32 %209, 4, !dbg !18
  br i1 %210, label %assert_end90, label %assert_fail87, !dbg !18, !prof !19

assert_fail87:                                    ; preds = %assert_end86
  %211 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %211(ptr nonnull @.str.39), !dbg !18
  br label %common.ret

assert_end90:                                     ; preds = %assert_end86
  %212 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 0, !dbg !18
  %213 = load i8, ptr %212, align 1, !dbg !18
  %214 = icmp eq i8 %213, 2, !dbg !18
  %215 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 1, !dbg !18
  %216 = load i8, ptr %215, align 1, !dbg !18
  %217 = icmp eq i8 %216, 16, !dbg !18
  %218 = and i1 %214, %217, !dbg !18
  %219 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 2, !dbg !18
  %220 = load i16, ptr %219, align 2, !dbg !18
  %221 = icmp eq i16 %220, 1, !dbg !18
  %222 = and i1 %218, %221, !dbg !18
  br i1 %222, label %assert_end92, label %assert_fail91, !dbg !18, !prof !19

assert_fail91:                                    ; preds = %assert_end90
  %223 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %223(ptr nonnull @.str.40), !dbg !18
  br label %common.ret

assert_end92:                                     ; preds = %assert_end90
  %224 = load i64, ptr %backward_op.V_handle.shape, align 8, !dbg !18, !tbaa !57
  %225 = and i64 %224, 4294967295, !dbg !18
  %226 = icmp eq i64 %225, 6, !dbg !18
  br i1 %226, label %assert_end94, label %assert_fail93, !dbg !18, !prof !19

assert_fail93:                                    ; preds = %assert_end92
  %227 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %227(ptr nonnull @.str.41), !dbg !18
  br label %common.ret

assert_end94:                                     ; preds = %assert_end92
  %228 = getelementptr inbounds i64, ptr %backward_op.V_handle.shape, i64 1, !dbg !18
  %229 = load i64, ptr %228, align 8, !dbg !18, !tbaa !57
  %230 = and i64 %229, 4294967295, !dbg !18
  %231 = icmp eq i64 %230, 32, !dbg !18
  br i1 %231, label %assert_end96, label %assert_fail95, !dbg !18, !prof !19

assert_fail95:                                    ; preds = %assert_end94
  %232 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %232(ptr nonnull @.str.42), !dbg !18
  br label %common.ret

assert_end96:                                     ; preds = %assert_end94
  %233 = getelementptr inbounds i64, ptr %backward_op.V_handle.shape, i64 2, !dbg !18
  %234 = load i64, ptr %233, align 8, !dbg !18, !tbaa !57
  %235 = and i64 %234, 4294967295, !dbg !18
  %236 = icmp eq i64 %235, 512, !dbg !18
  br i1 %236, label %assert_end98, label %assert_fail97, !dbg !18, !prof !19

assert_fail97:                                    ; preds = %assert_end96
  %237 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %237(ptr nonnull @.str.43), !dbg !18
  br label %common.ret

assert_end98:                                     ; preds = %assert_end96
  %238 = getelementptr inbounds i64, ptr %backward_op.V_handle.shape, i64 3, !dbg !18
  %239 = load i64, ptr %238, align 8, !dbg !18, !tbaa !57
  %240 = and i64 %239, 4294967295, !dbg !18
  %241 = icmp eq i64 %240, 128, !dbg !18
  br i1 %241, label %assert_end100, label %assert_fail99, !dbg !18, !prof !19

assert_fail99:                                    ; preds = %assert_end98
  %242 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %242(ptr nonnull @.str.44), !dbg !18
  br label %common.ret

assert_end100:                                    ; preds = %assert_end98
  %.not239 = icmp eq ptr %backward_op.V_handle.strides, null, !dbg !18
  br i1 %.not239, label %if_end102, label %if_then101, !dbg !18, !prof !59

if_then101:                                       ; preds = %assert_end100
  %243 = load <4 x i64>, ptr %backward_op.V_handle.strides, align 8, !dbg !18, !tbaa !57
  %244 = and <4 x i64> %243, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %245 = icmp ne <4 x i64> %244, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %246 = bitcast <4 x i1> %245 to i4, !dbg !18
  %247 = icmp eq i4 %246, 0, !dbg !18
  br i1 %247, label %if_end102, label %assert_fail103, !dbg !18, !prof !19

if_end102:                                        ; preds = %if_then101, %assert_end100
  %248 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 6, !dbg !18
  %249 = load i64, ptr %248, align 8, !dbg !18
  %250 = icmp eq i64 %249, 0, !dbg !18
  br i1 %250, label %assert_end106, label %assert_fail105, !dbg !18, !prof !19

assert_fail103:                                   ; preds = %if_then101
  %251 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %251(ptr nonnull @.str.45), !dbg !18
  br label %common.ret

assert_fail105:                                   ; preds = %if_end102
  %252 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %252(ptr nonnull @.str.46), !dbg !18
  br label %common.ret

assert_end106:                                    ; preds = %if_end102
  %253 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 0, !dbg !18
  %254 = load i32, ptr %253, align 4, !dbg !18
  %255 = icmp eq i32 %254, 8, !dbg !18
  br i1 %255, label %assert_end108, label %assert_fail107, !dbg !18, !prof !19

assert_fail107:                                   ; preds = %assert_end106
  %256 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %256(ptr nonnull @.str.47), !dbg !18
  br label %common.ret

assert_end108:                                    ; preds = %assert_end106
  %257 = trunc i64 %25 to i32
  %258 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 1, !dbg !18
  %259 = load i32, ptr %258, align 4, !dbg !18
  %260 = icmp eq i32 %257, %259, !dbg !18
  br i1 %260, label %assert_end110, label %assert_fail109, !dbg !18, !prof !19

assert_fail109:                                   ; preds = %assert_end108
  %261 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %261(ptr nonnull @.str.48), !dbg !18
  br label %common.ret

assert_end110:                                    ; preds = %assert_end108
  %262 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 2, !dbg !18
  %263 = load i32, ptr %262, align 4, !dbg !18
  %264 = icmp eq i32 %263, 4, !dbg !18
  br i1 %264, label %assert_end114, label %assert_fail111, !dbg !18, !prof !19

assert_fail111:                                   ; preds = %assert_end110
  %265 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %265(ptr nonnull @.str.49), !dbg !18
  br label %common.ret

assert_end114:                                    ; preds = %assert_end110
  %266 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 0, !dbg !18
  %267 = load i8, ptr %266, align 1, !dbg !18
  %268 = icmp eq i8 %267, 2, !dbg !18
  %269 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 1, !dbg !18
  %270 = load i8, ptr %269, align 1, !dbg !18
  %271 = icmp eq i8 %270, 16, !dbg !18
  %272 = and i1 %268, %271, !dbg !18
  %273 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 2, !dbg !18
  %274 = load i16, ptr %273, align 2, !dbg !18
  %275 = icmp eq i16 %274, 1, !dbg !18
  %276 = and i1 %272, %275, !dbg !18
  br i1 %276, label %assert_end116, label %assert_fail115, !dbg !18, !prof !19

assert_fail115:                                   ; preds = %assert_end114
  %277 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %277(ptr nonnull @.str.50), !dbg !18
  br label %common.ret

assert_end116:                                    ; preds = %assert_end114
  %278 = load i64, ptr %backward_op.O_handle.shape, align 8, !dbg !18, !tbaa !57
  %279 = and i64 %278, 4294967295, !dbg !18
  %280 = icmp eq i64 %279, 6, !dbg !18
  br i1 %280, label %assert_end118, label %assert_fail117, !dbg !18, !prof !19

assert_fail117:                                   ; preds = %assert_end116
  %281 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %281(ptr nonnull @.str.51), !dbg !18
  br label %common.ret

assert_end118:                                    ; preds = %assert_end116
  %282 = getelementptr inbounds i64, ptr %backward_op.O_handle.shape, i64 1, !dbg !18
  %283 = load i64, ptr %282, align 8, !dbg !18, !tbaa !57
  %284 = and i64 %283, 4294967295, !dbg !18
  %285 = icmp eq i64 %284, 32, !dbg !18
  br i1 %285, label %assert_end120, label %assert_fail119, !dbg !18, !prof !19

assert_fail119:                                   ; preds = %assert_end118
  %286 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %286(ptr nonnull @.str.52), !dbg !18
  br label %common.ret

assert_end120:                                    ; preds = %assert_end118
  %287 = getelementptr inbounds i64, ptr %backward_op.O_handle.shape, i64 2, !dbg !18
  %288 = load i64, ptr %287, align 8, !dbg !18, !tbaa !57
  %289 = and i64 %288, 4294967295, !dbg !18
  %290 = icmp eq i64 %289, 512, !dbg !18
  br i1 %290, label %assert_end122, label %assert_fail121, !dbg !18, !prof !19

assert_fail121:                                   ; preds = %assert_end120
  %291 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %291(ptr nonnull @.str.53), !dbg !18
  br label %common.ret

assert_end122:                                    ; preds = %assert_end120
  %292 = getelementptr inbounds i64, ptr %backward_op.O_handle.shape, i64 3, !dbg !18
  %293 = load i64, ptr %292, align 8, !dbg !18, !tbaa !57
  %294 = and i64 %293, 4294967295, !dbg !18
  %295 = icmp eq i64 %294, 128, !dbg !18
  br i1 %295, label %assert_end124, label %assert_fail123, !dbg !18, !prof !19

assert_fail123:                                   ; preds = %assert_end122
  %296 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %296(ptr nonnull @.str.54), !dbg !18
  br label %common.ret

assert_end124:                                    ; preds = %assert_end122
  %.not240 = icmp eq ptr %backward_op.O_handle.strides, null, !dbg !18
  br i1 %.not240, label %if_end126, label %if_then125, !dbg !18, !prof !59

if_then125:                                       ; preds = %assert_end124
  %297 = load <4 x i64>, ptr %backward_op.O_handle.strides, align 8, !dbg !18, !tbaa !57
  %298 = and <4 x i64> %297, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %299 = icmp ne <4 x i64> %298, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %300 = bitcast <4 x i1> %299 to i4, !dbg !18
  %301 = icmp eq i4 %300, 0, !dbg !18
  br i1 %301, label %if_end126, label %assert_fail127, !dbg !18, !prof !19

if_end126:                                        ; preds = %if_then125, %assert_end124
  %302 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 6, !dbg !18
  %303 = load i64, ptr %302, align 8, !dbg !18
  %304 = icmp eq i64 %303, 0, !dbg !18
  br i1 %304, label %assert_end130, label %assert_fail129, !dbg !18, !prof !19

assert_fail127:                                   ; preds = %if_then125
  %305 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %305(ptr nonnull @.str.55), !dbg !18
  br label %common.ret

assert_fail129:                                   ; preds = %if_end126
  %306 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %306(ptr nonnull @.str.56), !dbg !18
  br label %common.ret

assert_end130:                                    ; preds = %if_end126
  %307 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 1, i32 0, !dbg !18
  %308 = load i32, ptr %307, align 4, !dbg !18
  %309 = icmp eq i32 %308, 8, !dbg !18
  br i1 %309, label %assert_end132, label %assert_fail131, !dbg !18, !prof !19

assert_fail131:                                   ; preds = %assert_end130
  %310 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %310(ptr nonnull @.str.57), !dbg !18
  br label %common.ret

assert_end132:                                    ; preds = %assert_end130
  %311 = trunc i64 %25 to i32
  %312 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 1, i32 1, !dbg !18
  %313 = load i32, ptr %312, align 4, !dbg !18
  %314 = icmp eq i32 %311, %313, !dbg !18
  br i1 %314, label %assert_end134, label %assert_fail133, !dbg !18, !prof !19

assert_fail133:                                   ; preds = %assert_end132
  %315 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %315(ptr nonnull @.str.58), !dbg !18
  br label %common.ret

assert_end134:                                    ; preds = %assert_end132
  %316 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 2, !dbg !18
  %317 = load i32, ptr %316, align 4, !dbg !18
  %318 = icmp eq i32 %317, 3, !dbg !18
  br i1 %318, label %assert_end138, label %assert_fail135, !dbg !18, !prof !19

assert_fail135:                                   ; preds = %assert_end134
  %319 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %319(ptr nonnull @.str.59), !dbg !18
  br label %common.ret

assert_end138:                                    ; preds = %assert_end134
  %320 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 0, !dbg !18
  %321 = load i8, ptr %320, align 1, !dbg !18
  %322 = icmp eq i8 %321, 2, !dbg !18
  %323 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 1, !dbg !18
  %324 = load i8, ptr %323, align 1, !dbg !18
  %325 = icmp eq i8 %324, 16, !dbg !18
  %326 = and i1 %322, %325, !dbg !18
  %327 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 2, !dbg !18
  %328 = load i16, ptr %327, align 2, !dbg !18
  %329 = icmp eq i16 %328, 1, !dbg !18
  %330 = and i1 %326, %329, !dbg !18
  br i1 %330, label %assert_end140, label %assert_fail139, !dbg !18, !prof !19

assert_fail139:                                   ; preds = %assert_end138
  %331 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %331(ptr nonnull @.str.60), !dbg !18
  br label %common.ret

assert_end140:                                    ; preds = %assert_end138
  %332 = load i64, ptr %backward_op.L_handle.shape, align 8, !dbg !18, !tbaa !57
  %333 = and i64 %332, 4294967295, !dbg !18
  %334 = icmp eq i64 %333, 6, !dbg !18
  br i1 %334, label %assert_end142, label %assert_fail141, !dbg !18, !prof !19

assert_fail141:                                   ; preds = %assert_end140
  %335 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %335(ptr nonnull @.str.61), !dbg !18
  br label %common.ret

assert_end142:                                    ; preds = %assert_end140
  %336 = getelementptr inbounds i64, ptr %backward_op.L_handle.shape, i64 1, !dbg !18
  %337 = load i64, ptr %336, align 8, !dbg !18, !tbaa !57
  %338 = and i64 %337, 4294967295, !dbg !18
  %339 = icmp eq i64 %338, 32, !dbg !18
  br i1 %339, label %assert_end144, label %assert_fail143, !dbg !18, !prof !19

assert_fail143:                                   ; preds = %assert_end142
  %340 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %340(ptr nonnull @.str.62), !dbg !18
  br label %common.ret

assert_end144:                                    ; preds = %assert_end142
  %341 = getelementptr inbounds i64, ptr %backward_op.L_handle.shape, i64 2, !dbg !18
  %342 = load i64, ptr %341, align 8, !dbg !18, !tbaa !57
  %343 = and i64 %342, 4294967295, !dbg !18
  %344 = icmp eq i64 %343, 512, !dbg !18
  br i1 %344, label %assert_end146, label %assert_fail145, !dbg !18, !prof !19

assert_fail145:                                   ; preds = %assert_end144
  %345 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %345(ptr nonnull @.str.63), !dbg !18
  br label %common.ret

assert_end146:                                    ; preds = %assert_end144
  %.not241 = icmp eq ptr %backward_op.L_handle.strides, null, !dbg !18
  br i1 %.not241, label %if_end148, label %if_then147, !dbg !18, !prof !59

if_then147:                                       ; preds = %assert_end146
  %346 = getelementptr inbounds i64, ptr %backward_op.L_handle.strides, i64 2, !dbg !18
  %347 = load i64, ptr %346, align 8, !dbg !18, !tbaa !57
  %348 = and i64 %347, 4294967295, !dbg !18
  %349 = icmp eq i64 %348, 1, !dbg !18
  %350 = getelementptr inbounds i64, ptr %backward_op.L_handle.strides, i64 1, !dbg !18
  %351 = load i64, ptr %350, align 8, !dbg !18, !tbaa !57
  %352 = and i64 %351, 4294967295, !dbg !18
  %353 = icmp eq i64 %352, 512, !dbg !18
  %354 = and i1 %349, %353, !dbg !18
  %355 = load i64, ptr %backward_op.L_handle.strides, align 8, !dbg !18, !tbaa !57
  %356 = and i64 %355, 4294967295, !dbg !18
  %357 = icmp eq i64 %356, 16384, !dbg !18
  %358 = and i1 %354, %357, !dbg !18
  br i1 %358, label %if_end148, label %assert_fail149, !dbg !18, !prof !19

if_end148:                                        ; preds = %if_then147, %assert_end146
  %359 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 6, !dbg !18
  %360 = load i64, ptr %359, align 8, !dbg !18
  %361 = icmp eq i64 %360, 0, !dbg !18
  br i1 %361, label %assert_end152, label %assert_fail151, !dbg !18, !prof !19

assert_fail149:                                   ; preds = %if_then147
  %362 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %362(ptr nonnull @.str.64), !dbg !18
  br label %common.ret

assert_fail151:                                   ; preds = %if_end148
  %363 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %363(ptr nonnull @.str.65), !dbg !18
  br label %common.ret

assert_end152:                                    ; preds = %if_end148
  %364 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 1, i32 0, !dbg !18
  %365 = load i32, ptr %364, align 4, !dbg !18
  %366 = icmp eq i32 %365, 8, !dbg !18
  br i1 %366, label %assert_end154, label %assert_fail153, !dbg !18, !prof !19

assert_fail153:                                   ; preds = %assert_end152
  %367 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %367(ptr nonnull @.str.66), !dbg !18
  br label %common.ret

assert_end154:                                    ; preds = %assert_end152
  %368 = trunc i64 %25 to i32
  %369 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 1, i32 1, !dbg !18
  %370 = load i32, ptr %369, align 4, !dbg !18
  %371 = icmp eq i32 %368, %370, !dbg !18
  br i1 %371, label %assert_end156, label %assert_fail155, !dbg !18, !prof !19

assert_fail155:                                   ; preds = %assert_end154
  %372 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %372(ptr nonnull @.str.67), !dbg !18
  br label %common.ret

assert_end156:                                    ; preds = %assert_end154
  %373 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 2, !dbg !18
  %374 = load i32, ptr %373, align 4, !dbg !18
  %375 = icmp eq i32 %374, 4, !dbg !18
  br i1 %375, label %assert_end160, label %assert_fail157, !dbg !18, !prof !19

assert_fail157:                                   ; preds = %assert_end156
  %376 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %376(ptr nonnull @.str.68), !dbg !18
  br label %common.ret

assert_end160:                                    ; preds = %assert_end156
  %377 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 3, i32 0, !dbg !18
  %378 = load i8, ptr %377, align 1, !dbg !18
  %379 = icmp eq i8 %378, 2, !dbg !18
  %380 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 3, i32 1, !dbg !18
  %381 = load i8, ptr %380, align 1, !dbg !18
  %382 = icmp eq i8 %381, 16, !dbg !18
  %383 = and i1 %379, %382, !dbg !18
  %384 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 3, i32 2, !dbg !18
  %385 = load i16, ptr %384, align 2, !dbg !18
  %386 = icmp eq i16 %385, 1, !dbg !18
  %387 = and i1 %383, %386, !dbg !18
  br i1 %387, label %assert_end162, label %assert_fail161, !dbg !18, !prof !19

assert_fail161:                                   ; preds = %assert_end160
  %388 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %388(ptr nonnull @.str.69), !dbg !18
  br label %common.ret

assert_end162:                                    ; preds = %assert_end160
  %389 = load i64, ptr %backward_op.dQ_handle.shape, align 8, !dbg !18, !tbaa !57
  %390 = and i64 %389, 4294967295, !dbg !18
  %391 = icmp eq i64 %390, 6, !dbg !18
  br i1 %391, label %assert_end164, label %assert_fail163, !dbg !18, !prof !19

assert_fail163:                                   ; preds = %assert_end162
  %392 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %392(ptr nonnull @.str.70), !dbg !18
  br label %common.ret

assert_end164:                                    ; preds = %assert_end162
  %393 = getelementptr inbounds i64, ptr %backward_op.dQ_handle.shape, i64 1, !dbg !18
  %394 = load i64, ptr %393, align 8, !dbg !18, !tbaa !57
  %395 = and i64 %394, 4294967295, !dbg !18
  %396 = icmp eq i64 %395, 32, !dbg !18
  br i1 %396, label %assert_end166, label %assert_fail165, !dbg !18, !prof !19

assert_fail165:                                   ; preds = %assert_end164
  %397 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %397(ptr nonnull @.str.71), !dbg !18
  br label %common.ret

assert_end166:                                    ; preds = %assert_end164
  %398 = getelementptr inbounds i64, ptr %backward_op.dQ_handle.shape, i64 2, !dbg !18
  %399 = load i64, ptr %398, align 8, !dbg !18, !tbaa !57
  %400 = and i64 %399, 4294967295, !dbg !18
  %401 = icmp eq i64 %400, 512, !dbg !18
  br i1 %401, label %assert_end168, label %assert_fail167, !dbg !18, !prof !19

assert_fail167:                                   ; preds = %assert_end166
  %402 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %402(ptr nonnull @.str.72), !dbg !18
  br label %common.ret

assert_end168:                                    ; preds = %assert_end166
  %403 = getelementptr inbounds i64, ptr %backward_op.dQ_handle.shape, i64 3, !dbg !18
  %404 = load i64, ptr %403, align 8, !dbg !18, !tbaa !57
  %405 = and i64 %404, 4294967295, !dbg !18
  %406 = icmp eq i64 %405, 128, !dbg !18
  br i1 %406, label %assert_end170, label %assert_fail169, !dbg !18, !prof !19

assert_fail169:                                   ; preds = %assert_end168
  %407 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %407(ptr nonnull @.str.73), !dbg !18
  br label %common.ret

assert_end170:                                    ; preds = %assert_end168
  %.not242 = icmp eq ptr %backward_op.dQ_handle.strides, null, !dbg !18
  br i1 %.not242, label %if_end172, label %if_then171, !dbg !18, !prof !59

if_then171:                                       ; preds = %assert_end170
  %408 = load <4 x i64>, ptr %backward_op.dQ_handle.strides, align 8, !dbg !18, !tbaa !57
  %409 = and <4 x i64> %408, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %410 = icmp ne <4 x i64> %409, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %411 = bitcast <4 x i1> %410 to i4, !dbg !18
  %412 = icmp eq i4 %411, 0, !dbg !18
  br i1 %412, label %if_end172, label %assert_fail173, !dbg !18, !prof !19

if_end172:                                        ; preds = %if_then171, %assert_end170
  %413 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 6, !dbg !18
  %414 = load i64, ptr %413, align 8, !dbg !18
  %415 = icmp eq i64 %414, 0, !dbg !18
  br i1 %415, label %assert_end176, label %assert_fail175, !dbg !18, !prof !19

assert_fail173:                                   ; preds = %if_then171
  %416 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %416(ptr nonnull @.str.74), !dbg !18
  br label %common.ret

assert_fail175:                                   ; preds = %if_end172
  %417 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %417(ptr nonnull @.str.75), !dbg !18
  br label %common.ret

assert_end176:                                    ; preds = %if_end172
  %418 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 1, i32 0, !dbg !18
  %419 = load i32, ptr %418, align 4, !dbg !18
  %420 = icmp eq i32 %419, 8, !dbg !18
  br i1 %420, label %assert_end178, label %assert_fail177, !dbg !18, !prof !19

assert_fail177:                                   ; preds = %assert_end176
  %421 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %421(ptr nonnull @.str.76), !dbg !18
  br label %common.ret

assert_end178:                                    ; preds = %assert_end176
  %422 = trunc i64 %25 to i32
  %423 = getelementptr inbounds %1, ptr %dQ_handle, i64 0, i32 1, i32 1, !dbg !18
  %424 = load i32, ptr %423, align 4, !dbg !18
  %425 = icmp eq i32 %422, %424, !dbg !18
  br i1 %425, label %assert_end180, label %assert_fail179, !dbg !18, !prof !19

assert_fail179:                                   ; preds = %assert_end178
  %426 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %426(ptr nonnull @.str.77), !dbg !18
  br label %common.ret

assert_end180:                                    ; preds = %assert_end178
  %427 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 2, !dbg !18
  %428 = load i32, ptr %427, align 4, !dbg !18
  %429 = icmp eq i32 %428, 4, !dbg !18
  br i1 %429, label %assert_end184, label %assert_fail181, !dbg !18, !prof !19

assert_fail181:                                   ; preds = %assert_end180
  %430 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %430(ptr nonnull @.str.78), !dbg !18
  br label %common.ret

assert_end184:                                    ; preds = %assert_end180
  %431 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 3, i32 0, !dbg !18
  %432 = load i8, ptr %431, align 1, !dbg !18
  %433 = icmp eq i8 %432, 2, !dbg !18
  %434 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 3, i32 1, !dbg !18
  %435 = load i8, ptr %434, align 1, !dbg !18
  %436 = icmp eq i8 %435, 16, !dbg !18
  %437 = and i1 %433, %436, !dbg !18
  %438 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 3, i32 2, !dbg !18
  %439 = load i16, ptr %438, align 2, !dbg !18
  %440 = icmp eq i16 %439, 1, !dbg !18
  %441 = and i1 %437, %440, !dbg !18
  br i1 %441, label %assert_end186, label %assert_fail185, !dbg !18, !prof !19

assert_fail185:                                   ; preds = %assert_end184
  %442 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %442(ptr nonnull @.str.79), !dbg !18
  br label %common.ret

assert_end186:                                    ; preds = %assert_end184
  %443 = load i64, ptr %backward_op.dK_handle.shape, align 8, !dbg !18, !tbaa !57
  %444 = and i64 %443, 4294967295, !dbg !18
  %445 = icmp eq i64 %444, 6, !dbg !18
  br i1 %445, label %assert_end188, label %assert_fail187, !dbg !18, !prof !19

assert_fail187:                                   ; preds = %assert_end186
  %446 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %446(ptr nonnull @.str.80), !dbg !18
  br label %common.ret

assert_end188:                                    ; preds = %assert_end186
  %447 = getelementptr inbounds i64, ptr %backward_op.dK_handle.shape, i64 1, !dbg !18
  %448 = load i64, ptr %447, align 8, !dbg !18, !tbaa !57
  %449 = and i64 %448, 4294967295, !dbg !18
  %450 = icmp eq i64 %449, 32, !dbg !18
  br i1 %450, label %assert_end190, label %assert_fail189, !dbg !18, !prof !19

assert_fail189:                                   ; preds = %assert_end188
  %451 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %451(ptr nonnull @.str.81), !dbg !18
  br label %common.ret

assert_end190:                                    ; preds = %assert_end188
  %452 = getelementptr inbounds i64, ptr %backward_op.dK_handle.shape, i64 2, !dbg !18
  %453 = load i64, ptr %452, align 8, !dbg !18, !tbaa !57
  %454 = and i64 %453, 4294967295, !dbg !18
  %455 = icmp eq i64 %454, 512, !dbg !18
  br i1 %455, label %assert_end192, label %assert_fail191, !dbg !18, !prof !19

assert_fail191:                                   ; preds = %assert_end190
  %456 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %456(ptr nonnull @.str.82), !dbg !18
  br label %common.ret

assert_end192:                                    ; preds = %assert_end190
  %457 = getelementptr inbounds i64, ptr %backward_op.dK_handle.shape, i64 3, !dbg !18
  %458 = load i64, ptr %457, align 8, !dbg !18, !tbaa !57
  %459 = and i64 %458, 4294967295, !dbg !18
  %460 = icmp eq i64 %459, 128, !dbg !18
  br i1 %460, label %assert_end194, label %assert_fail193, !dbg !18, !prof !19

assert_fail193:                                   ; preds = %assert_end192
  %461 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %461(ptr nonnull @.str.83), !dbg !18
  br label %common.ret

assert_end194:                                    ; preds = %assert_end192
  %.not243 = icmp eq ptr %backward_op.dK_handle.strides, null, !dbg !18
  br i1 %.not243, label %if_end196, label %if_then195, !dbg !18, !prof !59

if_then195:                                       ; preds = %assert_end194
  %462 = load <4 x i64>, ptr %backward_op.dK_handle.strides, align 8, !dbg !18, !tbaa !57
  %463 = and <4 x i64> %462, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %464 = icmp ne <4 x i64> %463, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %465 = bitcast <4 x i1> %464 to i4, !dbg !18
  %466 = icmp eq i4 %465, 0, !dbg !18
  br i1 %466, label %if_end196, label %assert_fail197, !dbg !18, !prof !19

if_end196:                                        ; preds = %if_then195, %assert_end194
  %467 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 6, !dbg !18
  %468 = load i64, ptr %467, align 8, !dbg !18
  %469 = icmp eq i64 %468, 0, !dbg !18
  br i1 %469, label %assert_end200, label %assert_fail199, !dbg !18, !prof !19

assert_fail197:                                   ; preds = %if_then195
  %470 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %470(ptr nonnull @.str.84), !dbg !18
  br label %common.ret

assert_fail199:                                   ; preds = %if_end196
  %471 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %471(ptr nonnull @.str.85), !dbg !18
  br label %common.ret

assert_end200:                                    ; preds = %if_end196
  %472 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 1, i32 0, !dbg !18
  %473 = load i32, ptr %472, align 4, !dbg !18
  %474 = icmp eq i32 %473, 8, !dbg !18
  br i1 %474, label %assert_end202, label %assert_fail201, !dbg !18, !prof !19

assert_fail201:                                   ; preds = %assert_end200
  %475 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %475(ptr nonnull @.str.86), !dbg !18
  br label %common.ret

assert_end202:                                    ; preds = %assert_end200
  %476 = trunc i64 %25 to i32
  %477 = getelementptr inbounds %1, ptr %dK_handle, i64 0, i32 1, i32 1, !dbg !18
  %478 = load i32, ptr %477, align 4, !dbg !18
  %479 = icmp eq i32 %476, %478, !dbg !18
  br i1 %479, label %assert_end204, label %assert_fail203, !dbg !18, !prof !19

assert_fail203:                                   ; preds = %assert_end202
  %480 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %480(ptr nonnull @.str.87), !dbg !18
  br label %common.ret

assert_end204:                                    ; preds = %assert_end202
  %481 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 2, !dbg !18
  %482 = load i32, ptr %481, align 4, !dbg !18
  %483 = icmp eq i32 %482, 4, !dbg !18
  br i1 %483, label %assert_end208, label %assert_fail205, !dbg !18, !prof !19

assert_fail205:                                   ; preds = %assert_end204
  %484 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %484(ptr nonnull @.str.88), !dbg !18
  br label %common.ret

assert_end208:                                    ; preds = %assert_end204
  %485 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 3, i32 0, !dbg !18
  %486 = load i8, ptr %485, align 1, !dbg !18
  %487 = icmp eq i8 %486, 2, !dbg !18
  %488 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 3, i32 1, !dbg !18
  %489 = load i8, ptr %488, align 1, !dbg !18
  %490 = icmp eq i8 %489, 16, !dbg !18
  %491 = and i1 %487, %490, !dbg !18
  %492 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 3, i32 2, !dbg !18
  %493 = load i16, ptr %492, align 2, !dbg !18
  %494 = icmp eq i16 %493, 1, !dbg !18
  %495 = and i1 %491, %494, !dbg !18
  br i1 %495, label %assert_end210, label %assert_fail209, !dbg !18, !prof !19

assert_fail209:                                   ; preds = %assert_end208
  %496 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %496(ptr nonnull @.str.89), !dbg !18
  br label %common.ret

assert_end210:                                    ; preds = %assert_end208
  %497 = load i64, ptr %backward_op.dV_handle.shape, align 8, !dbg !18, !tbaa !57
  %498 = and i64 %497, 4294967295, !dbg !18
  %499 = icmp eq i64 %498, 6, !dbg !18
  br i1 %499, label %assert_end212, label %assert_fail211, !dbg !18, !prof !19

assert_fail211:                                   ; preds = %assert_end210
  %500 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %500(ptr nonnull @.str.90), !dbg !18
  br label %common.ret

assert_end212:                                    ; preds = %assert_end210
  %501 = getelementptr inbounds i64, ptr %backward_op.dV_handle.shape, i64 1, !dbg !18
  %502 = load i64, ptr %501, align 8, !dbg !18, !tbaa !57
  %503 = and i64 %502, 4294967295, !dbg !18
  %504 = icmp eq i64 %503, 32, !dbg !18
  br i1 %504, label %assert_end214, label %assert_fail213, !dbg !18, !prof !19

assert_fail213:                                   ; preds = %assert_end212
  %505 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %505(ptr nonnull @.str.91), !dbg !18
  br label %common.ret

assert_end214:                                    ; preds = %assert_end212
  %506 = getelementptr inbounds i64, ptr %backward_op.dV_handle.shape, i64 2, !dbg !18
  %507 = load i64, ptr %506, align 8, !dbg !18, !tbaa !57
  %508 = and i64 %507, 4294967295, !dbg !18
  %509 = icmp eq i64 %508, 512, !dbg !18
  br i1 %509, label %assert_end216, label %assert_fail215, !dbg !18, !prof !19

assert_fail215:                                   ; preds = %assert_end214
  %510 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %510(ptr nonnull @.str.92), !dbg !18
  br label %common.ret

assert_end216:                                    ; preds = %assert_end214
  %511 = getelementptr inbounds i64, ptr %backward_op.dV_handle.shape, i64 3, !dbg !18
  %512 = load i64, ptr %511, align 8, !dbg !18, !tbaa !57
  %513 = and i64 %512, 4294967295, !dbg !18
  %514 = icmp eq i64 %513, 128, !dbg !18
  br i1 %514, label %assert_end218, label %assert_fail217, !dbg !18, !prof !19

assert_fail217:                                   ; preds = %assert_end216
  %515 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %515(ptr nonnull @.str.93), !dbg !18
  br label %common.ret

assert_end218:                                    ; preds = %assert_end216
  %.not244 = icmp eq ptr %backward_op.dV_handle.strides, null, !dbg !18
  br i1 %.not244, label %if_end220, label %if_then219, !dbg !18, !prof !59

if_then219:                                       ; preds = %assert_end218
  %516 = load <4 x i64>, ptr %backward_op.dV_handle.strides, align 8, !dbg !18, !tbaa !57
  %517 = and <4 x i64> %516, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !18
  %518 = icmp ne <4 x i64> %517, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !18
  %519 = bitcast <4 x i1> %518 to i4, !dbg !18
  %520 = icmp eq i4 %519, 0, !dbg !18
  br i1 %520, label %if_end220, label %assert_fail221, !dbg !18, !prof !19

if_end220:                                        ; preds = %if_then219, %assert_end218
  %521 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 6, !dbg !18
  %522 = load i64, ptr %521, align 8, !dbg !18
  %523 = icmp eq i64 %522, 0, !dbg !18
  br i1 %523, label %assert_end224, label %assert_fail223, !dbg !18, !prof !19

assert_fail221:                                   ; preds = %if_then219
  %524 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %524(ptr nonnull @.str.94), !dbg !18
  br label %common.ret

assert_fail223:                                   ; preds = %if_end220
  %525 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %525(ptr nonnull @.str.95), !dbg !18
  br label %common.ret

assert_end224:                                    ; preds = %if_end220
  %526 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 1, i32 0, !dbg !18
  %527 = load i32, ptr %526, align 4, !dbg !18
  %528 = icmp eq i32 %527, 8, !dbg !18
  br i1 %528, label %assert_end226, label %assert_fail225, !dbg !18, !prof !19

assert_fail225:                                   ; preds = %assert_end224
  %529 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %529(ptr nonnull @.str.96), !dbg !18
  br label %common.ret

assert_end226:                                    ; preds = %assert_end224
  %530 = trunc i64 %25 to i32
  %531 = getelementptr inbounds %1, ptr %dV_handle, i64 0, i32 1, i32 1, !dbg !18
  %532 = load i32, ptr %531, align 4, !dbg !18
  %533 = icmp eq i32 %530, %532, !dbg !18
  br i1 %533, label %assert_end228, label %assert_fail227, !dbg !18, !prof !19

assert_fail227:                                   ; preds = %assert_end226
  %534 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %534(ptr nonnull @.str.97), !dbg !18
  br label %common.ret

assert_end228:                                    ; preds = %assert_end226
  store i64 8, ptr %stack_value235, align 8, !dbg !18
  %535 = getelementptr inbounds i64, ptr %stack_value235, i64 1, !dbg !18
  store i64 %25, ptr %535, align 8, !dbg !18
  store <2 x i32> zeroinitializer, ptr %stack_tcode236, align 8, !dbg !18, !tbaa !57
  %536 = getelementptr inbounds %0, ptr %stack_value235, i64 2, !dbg !18
  %537 = getelementptr inbounds i32, ptr %stack_tcode236, i64 2, !dbg !18
  %538 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !18, !tbaa !20
  %539 = load ptr, ptr @_MergedGlobals, align 8, !dbg !18
  %.not245 = icmp eq ptr %539, null, !dbg !18
  br i1 %.not245, label %handle_init, label %handle_init_end, !dbg !18, !prof !59

handle_init:                                      ; preds = %assert_end228
  %540 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !18, !tbaa !20
  %541 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !18, !tbaa !20
  %542 = call i32 %541(ptr %540, ptr nonnull @.str.98, ptr nonnull %3), !dbg !18
  %543 = icmp eq i32 %542, 0, !dbg !18
  br i1 %543, label %call_end, label %common.ret, !dbg !18, !prof !19

handle_init_end:                                  ; preds = %call_end, %assert_end228
  %544 = phi ptr [ %539, %assert_end228 ], [ %547, %call_end ], !dbg !18
  %545 = call i32 %538(ptr %544, ptr nonnull %stack_value235, ptr nonnull %stack_tcode236, i32 2, ptr nonnull %536, ptr nonnull %537), !dbg !18
  %546 = icmp eq i32 %545, 0, !dbg !18
  br i1 %546, label %call_end230, label %common.ret, !dbg !18, !prof !19

call_end:                                         ; preds = %handle_init
  %547 = load ptr, ptr %3, align 8, !dbg !18
  store ptr %547, ptr @_MergedGlobals, align 8, !dbg !18
  br label %handle_init_end, !dbg !18

call_end230:                                      ; preds = %handle_init_end
  %548 = trunc i64 %25 to i32
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0), !dbg !18
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1), !dbg !18
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2), !dbg !18
  %549 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !18, !tbaa !20
  %D.i = call ptr %549(i32 8, i32 %548, i64 196608, i32 2, i32 16), !dbg !18
  %550 = icmp eq ptr %D.i, null, !dbg !18
  br i1 %550, label %backward_op_compute_.exit, label %if_end.i, !dbg !18, !prof !19

if_end.i:                                         ; preds = %call_end230
  %551 = trunc i64 %25 to i32
  %552 = load ptr, ptr @__TVMBackendAllocWorkspace, align 8, !dbg !18, !tbaa !20
  %dQ_cache.i = call ptr %552(i32 8, i32 %551, i64 1610612736, i32 2, i32 16), !dbg !18
  %553 = icmp eq ptr %dQ_cache.i, null, !dbg !18
  br i1 %553, label %backward_op_compute_.exit, label %if_end2.i, !dbg !18, !prof !19

if_end2.i:                                        ; preds = %if_end.i
  store ptr %D.i, ptr %stack_value235, align 8, !dbg !18
  store i32 3, ptr %stack_tcode236, align 8, !dbg !18, !tbaa !57
  %sunkaddr = getelementptr inbounds i8, ptr %stack_value235, i64 8, !dbg !18
  store ptr %O, ptr %sunkaddr, align 8, !dbg !18
  %554 = icmp eq ptr %O, null, !dbg !18
  %spec.select.i = select i1 %554, i32 4, i32 3, !dbg !18
  %sunkaddr246 = getelementptr inbounds i8, ptr %stack_tcode236, i64 4, !dbg !18
  store i32 %spec.select.i, ptr %sunkaddr246, align 4, !dbg !18
  %sunkaddr247 = getelementptr inbounds i8, ptr %stack_value235, i64 16, !dbg !18
  store ptr %dO, ptr %sunkaddr247, align 8, !dbg !18
  %555 = icmp eq ptr %dO, null, !dbg !18
  %.sink84.i = select i1 %555, i32 4, i32 3, !dbg !18
  %sunkaddr248 = getelementptr inbounds i8, ptr %stack_tcode236, i64 8, !dbg !18
  store i32 %.sink84.i, ptr %sunkaddr248, align 8, !dbg !18
  %556 = getelementptr inbounds i64, ptr %stack_value235, i64 3, !dbg !18
  %557 = getelementptr inbounds i32, ptr %stack_tcode236, i64 3, !dbg !18
  %558 = getelementptr inbounds i64, ptr %stack_value235, i64 4, !dbg !18
  store <2 x i64> <i64 192, i64 128>, ptr %556, align 8, !dbg !18
  %559 = getelementptr inbounds i32, ptr %stack_tcode236, i64 4, !dbg !18
  store <2 x i32> zeroinitializer, ptr %557, align 4, !dbg !18, !tbaa !57
  %560 = getelementptr inbounds %0, ptr %stack_value235, i64 5, !dbg !18
  %561 = getelementptr inbounds i32, ptr %stack_tcode236, i64 5, !dbg !18
  %562 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !18, !tbaa !20
  %563 = load ptr, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 1), align 8, !dbg !18
  %.not.i = icmp eq ptr %563, null, !dbg !18
  br i1 %.not.i, label %handle_init.i, label %handle_init_end.i, !dbg !18, !prof !59

handle_init.i:                                    ; preds = %if_end2.i
  %564 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !18, !tbaa !20
  %565 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !18, !tbaa !20
  %566 = call i32 %565(ptr %564, ptr nonnull @.str.99, ptr nonnull %2), !dbg !18
  %567 = icmp eq i32 %566, 0, !dbg !18
  br i1 %567, label %call_end.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

handle_init_end.i:                                ; preds = %call_end.i, %if_end2.i
  %568 = phi ptr [ %563, %if_end2.i ], [ %571, %call_end.i ], !dbg !18
  %569 = call i32 %562(ptr %568, ptr nonnull %stack_value235, ptr nonnull %stack_tcode236, i32 5, ptr nonnull %560, ptr nonnull %561), !dbg !18
  %570 = icmp eq i32 %569, 0, !dbg !18
  br i1 %570, label %call_end13.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

call_end.i:                                       ; preds = %handle_init.i
  %571 = load ptr, ptr %2, align 8, !dbg !18
  store ptr %571, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 1), align 8, !dbg !18
  br label %handle_init_end.i, !dbg !18

call_end13.i:                                     ; preds = %handle_init_end.i
  store ptr %D.i, ptr %stack_value235, align 8, !dbg !18
  store i32 3, ptr %stack_tcode236, align 8, !dbg !18, !tbaa !57
  %sunkaddr249 = getelementptr inbounds i8, ptr %stack_value235, i64 8, !dbg !18
  store ptr %K, ptr %sunkaddr249, align 8, !dbg !18
  %572 = insertelement <4 x ptr> poison, ptr %K, i64 0, !dbg !18
  %573 = insertelement <4 x ptr> %572, ptr %L, i64 1, !dbg !18
  %574 = insertelement <4 x ptr> %573, ptr %Q, i64 2, !dbg !18
  %575 = insertelement <4 x ptr> %574, ptr %V, i64 3, !dbg !18
  %576 = icmp eq <4 x ptr> %575, zeroinitializer, !dbg !18
  %sunkaddr250 = getelementptr inbounds i8, ptr %stack_value235, i64 16, !dbg !18
  store ptr %L, ptr %sunkaddr250, align 8, !dbg !18
  %sunkaddr251 = getelementptr inbounds i8, ptr %stack_value235, i64 24, !dbg !18
  store ptr %Q, ptr %sunkaddr251, align 8, !dbg !18
  %sunkaddr252 = getelementptr inbounds i8, ptr %stack_value235, i64 32, !dbg !18
  store ptr %V, ptr %sunkaddr252, align 8, !dbg !18
  %577 = select <4 x i1> %576, <4 x i32> <i32 4, i32 4, i32 4, i32 4>, <4 x i32> <i32 3, i32 3, i32 3, i32 3>, !dbg !18
  %sunkaddr253 = getelementptr inbounds i8, ptr %stack_tcode236, i64 4, !dbg !18
  store <4 x i32> %577, ptr %sunkaddr253, align 4, !dbg !18
  %sunkaddr254 = getelementptr inbounds i8, ptr %stack_value235, i64 40, !dbg !18
  store ptr %dK, ptr %sunkaddr254, align 8, !dbg !18
  %578 = icmp eq ptr %dK, null, !dbg !18
  %.82.i = select i1 %578, i32 4, i32 3, !dbg !18
  %sunkaddr255 = getelementptr inbounds i8, ptr %stack_tcode236, i64 20, !dbg !18
  store i32 %.82.i, ptr %sunkaddr255, align 4, !dbg !18, !tbaa !57
  %579 = getelementptr inbounds %0, ptr %stack_value235, i64 6, !dbg !18
  store ptr %dO, ptr %579, align 8, !dbg !18
  %580 = getelementptr inbounds i32, ptr %stack_tcode236, i64 6, !dbg !18
  store i32 %.sink84.i, ptr %580, align 8, !dbg !18
  %581 = getelementptr inbounds %0, ptr %stack_value235, i64 7, !dbg !18
  store ptr %dQ_cache.i, ptr %581, align 8, !dbg !18
  %582 = getelementptr inbounds i32, ptr %stack_tcode236, i64 7, !dbg !18
  store i32 3, ptr %582, align 4, !dbg !18, !tbaa !57
  %583 = getelementptr inbounds %0, ptr %stack_value235, i64 8, !dbg !18
  store ptr %dV, ptr %583, align 8, !dbg !18
  %584 = icmp eq ptr %dV, null, !dbg !18
  %.sink88.i = select i1 %584, i32 4, i32 3, !dbg !18
  %585 = getelementptr inbounds i32, ptr %stack_tcode236, i64 8, !dbg !18
  store i32 %.sink88.i, ptr %585, align 8, !dbg !18
  %586 = getelementptr inbounds i64, ptr %stack_value235, i64 9, !dbg !18
  %587 = getelementptr inbounds i32, ptr %stack_tcode236, i64 9, !dbg !18
  store <2 x i64> <i64 6, i64 32>, ptr %586, align 8, !dbg !18
  %588 = getelementptr inbounds i64, ptr %stack_value235, i64 11, !dbg !18
  store <2 x i64> <i64 64, i64 128>, ptr %588, align 8, !dbg !18
  store <4 x i32> zeroinitializer, ptr %587, align 4, !dbg !18, !tbaa !57
  %589 = getelementptr inbounds %0, ptr %stack_value235, i64 13, !dbg !18
  %590 = getelementptr inbounds i32, ptr %stack_tcode236, i64 13, !dbg !18
  %591 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !18, !tbaa !20
  %592 = load ptr, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 2), align 8, !dbg !18
  %.not75.i = icmp eq ptr %592, null, !dbg !18
  br i1 %.not75.i, label %handle_init41.i, label %handle_init_end42.i, !dbg !18, !prof !59

handle_init41.i:                                  ; preds = %call_end13.i
  %593 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !18, !tbaa !20
  %594 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !18, !tbaa !20
  %595 = call i32 %594(ptr %593, ptr nonnull @.str.100, ptr nonnull %1), !dbg !18
  %596 = icmp eq i32 %595, 0, !dbg !18
  br i1 %596, label %call_end44.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

handle_init_end42.i:                              ; preds = %call_end44.i, %call_end13.i
  %597 = phi ptr [ %592, %call_end13.i ], [ %600, %call_end44.i ], !dbg !18
  %598 = call i32 %591(ptr %597, ptr nonnull %stack_value235, ptr nonnull %stack_tcode236, i32 13, ptr nonnull %589, ptr nonnull %590), !dbg !18
  %599 = icmp eq i32 %598, 0, !dbg !18
  br i1 %599, label %call_end46.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

call_end44.i:                                     ; preds = %handle_init41.i
  %600 = load ptr, ptr %1, align 8, !dbg !18
  store ptr %600, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 2), align 8, !dbg !18
  br label %handle_init_end42.i, !dbg !18

call_end46.i:                                     ; preds = %handle_init_end42.i
  store ptr %dQ, ptr %stack_value235, align 8, !dbg !18
  %601 = icmp eq ptr %dQ, null, !dbg !18
  %.83.i = select i1 %601, i32 4, i32 3, !dbg !18
  store i32 %.83.i, ptr %stack_tcode236, align 8, !dbg !18, !tbaa !57
  %sunkaddr256 = getelementptr inbounds i8, ptr %stack_value235, i64 8, !dbg !18
  store ptr %dQ_cache.i, ptr %sunkaddr256, align 8, !dbg !18
  %sunkaddr257 = getelementptr inbounds i8, ptr %stack_tcode236, i64 4, !dbg !18
  store <2 x i32> <i32 3, i32 0>, ptr %sunkaddr257, align 4, !dbg !18, !tbaa !57
  %sunkaddr258 = getelementptr inbounds i8, ptr %stack_value235, i64 16, !dbg !18
  store <2 x i64> <i64 98304, i64 128>, ptr %sunkaddr258, align 8, !dbg !18
  %sunkaddr259 = getelementptr inbounds i8, ptr %stack_tcode236, i64 12, !dbg !18
  store i32 0, ptr %sunkaddr259, align 4, !dbg !18, !tbaa !57
  %602 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !18, !tbaa !20
  %603 = load ptr, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 3), align 8, !dbg !18
  %.not77.i = icmp eq ptr %603, null, !dbg !18
  br i1 %.not77.i, label %handle_init53.i, label %handle_init_end54.i, !dbg !18, !prof !59

handle_init53.i:                                  ; preds = %call_end46.i
  %604 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !18, !tbaa !20
  %605 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !18, !tbaa !20
  %606 = call i32 %605(ptr %604, ptr nonnull @.str.101, ptr nonnull %0), !dbg !18
  %607 = icmp eq i32 %606, 0, !dbg !18
  br i1 %607, label %call_end56.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

handle_init_end54.i:                              ; preds = %call_end56.i, %call_end46.i
  %608 = phi ptr [ %603, %call_end46.i ], [ %611, %call_end56.i ], !dbg !18
  %609 = call i32 %602(ptr %608, ptr nonnull %stack_value235, ptr nonnull %stack_tcode236, i32 4, ptr nonnull %558, ptr nonnull %559), !dbg !18
  %610 = icmp eq i32 %609, 0, !dbg !18
  br i1 %610, label %call_end58.i, label %backward_op_compute_.exit, !dbg !18, !prof !19

call_end56.i:                                     ; preds = %handle_init53.i
  %611 = load ptr, ptr %0, align 8, !dbg !18
  store ptr %611, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 3), align 8, !dbg !18
  br label %handle_init_end54.i, !dbg !18

call_end58.i:                                     ; preds = %handle_init_end54.i
  %612 = trunc i64 %25 to i32
  %613 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !18, !tbaa !20
  %614 = call i32 %613(i32 8, i32 %612, ptr nonnull %dQ_cache.i), !dbg !18
  %.not78.i = icmp eq i32 %614, 0, !dbg !18
  br i1 %.not78.i, label %if_end60.i, label %backward_op_compute_.exit, !dbg !18, !prof !59

if_end60.i:                                       ; preds = %call_end58.i
  %615 = trunc i64 %25 to i32
  %616 = load ptr, ptr @__TVMBackendFreeWorkspace, align 8, !dbg !18, !tbaa !20
  %617 = call i32 %616(i32 8, i32 %615, ptr nonnull %D.i), !dbg !18
  %.not79.i = icmp ne i32 %617, 0, !dbg !18
  %..i = sext i1 %.not79.i to i32, !dbg !18
  br label %backward_op_compute_.exit, !dbg !18

backward_op_compute_.exit:                        ; preds = %call_end230, %if_end.i, %handle_init.i, %handle_init_end.i, %handle_init41.i, %handle_init_end42.i, %handle_init53.i, %handle_init_end54.i, %call_end58.i, %if_end60.i
  %common.ret.op.i = phi i32 [ -1, %call_end230 ], [ -1, %if_end.i ], [ %566, %handle_init.i ], [ %569, %handle_init_end.i ], [ %595, %handle_init41.i ], [ %598, %handle_init_end42.i ], [ %606, %handle_init53.i ], [ %609, %handle_init_end54.i ], [ -1, %call_end58.i ], [ %..i, %if_end60.i ], !dbg !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0), !dbg !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1), !dbg !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2), !dbg !18
  br label %common.ret
}

define dllexport i32 @forward_op(ptr nocapture readonly %args, ptr nocapture readonly %arg_type_ids, i32 %num_args, ptr nocapture readnone %out_ret_value, ptr nocapture readnone %out_ret_tcode, ptr nocapture readnone %resource_handle) local_unnamed_addr #0 !dbg !60 {
entry:
  %0 = alloca ptr, align 8
  call void @llvm.dbg.value(metadata ptr %args, metadata !62, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata ptr %arg_type_ids, metadata !63, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !64, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata ptr %out_ret_value, metadata !65, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata ptr %out_ret_tcode, metadata !66, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata ptr %resource_handle, metadata !67, metadata !DIExpression()), !dbg !68
  %1 = alloca ptr, align 8, !dbg !68
  %stack_value131 = alloca [9 x %0], align 8, !dbg !68
  %stack_tcode132 = alloca [9 x i32], align 16, !dbg !68
  %2 = icmp eq i32 %num_args, 5, !dbg !68
  br i1 %2, label %assert_end, label %assert_fail, !dbg !68, !prof !19

common.ret:                                       ; preds = %forward_op_compute_.exit, %handle_init_end, %handle_init, %assert_fail123, %assert_fail121, %assert_fail119, %assert_fail117, %assert_fail113, %assert_fail111, %assert_fail109, %assert_fail107, %assert_fail103, %assert_fail101, %assert_fail99, %assert_fail97, %assert_fail95, %assert_fail91, %assert_fail89, %assert_fail87, %assert_fail85, %assert_fail83, %assert_fail79, %assert_fail77, %assert_fail75, %assert_fail73, %assert_fail71, %assert_fail67, %assert_fail65, %assert_fail63, %assert_fail61, %assert_fail59, %assert_fail55, %assert_fail53, %assert_fail51, %assert_fail49, %assert_fail47, %assert_fail43, %assert_fail41, %assert_fail39, %assert_fail37, %assert_fail35, %assert_fail31, %assert_fail29, %assert_fail27, %assert_fail25, %assert_fail23, %assert_fail21, %assert_fail19, %assert_fail17, %assert_fail15, %assert_fail11, %assert_fail9, %assert_fail7, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail7 ], [ -1, %assert_fail9 ], [ -1, %assert_fail11 ], [ -1, %assert_fail15 ], [ -1, %assert_fail17 ], [ -1, %assert_fail19 ], [ -1, %assert_fail21 ], [ -1, %assert_fail23 ], [ -1, %assert_fail25 ], [ -1, %assert_fail27 ], [ -1, %assert_fail29 ], [ -1, %assert_fail31 ], [ -1, %assert_fail35 ], [ -1, %assert_fail37 ], [ -1, %assert_fail39 ], [ -1, %assert_fail41 ], [ -1, %assert_fail43 ], [ -1, %assert_fail47 ], [ -1, %assert_fail49 ], [ -1, %assert_fail51 ], [ -1, %assert_fail53 ], [ -1, %assert_fail55 ], [ -1, %assert_fail59 ], [ -1, %assert_fail61 ], [ -1, %assert_fail63 ], [ -1, %assert_fail65 ], [ -1, %assert_fail67 ], [ -1, %assert_fail71 ], [ -1, %assert_fail73 ], [ -1, %assert_fail75 ], [ -1, %assert_fail77 ], [ -1, %assert_fail79 ], [ -1, %assert_fail83 ], [ -1, %assert_fail85 ], [ -1, %assert_fail87 ], [ -1, %assert_fail89 ], [ -1, %assert_fail91 ], [ -1, %assert_fail95 ], [ -1, %assert_fail97 ], [ -1, %assert_fail99 ], [ -1, %assert_fail101 ], [ -1, %assert_fail103 ], [ -1, %assert_fail107 ], [ -1, %assert_fail109 ], [ -1, %assert_fail111 ], [ -1, %assert_fail113 ], [ -1, %assert_fail117 ], [ -1, %assert_fail119 ], [ -1, %assert_fail121 ], [ -1, %assert_fail123 ], [ %304, %handle_init ], [ %307, %handle_init_end ], [ %common.ret.op.i, %forward_op_compute_.exit ]
  ret i32 %common.ret.op, !dbg !68

assert_fail:                                      ; preds = %entry
  %3 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %3(ptr nonnull @.str.102), !dbg !68
  br label %common.ret

assert_end:                                       ; preds = %entry
  %Q_handle.code = load i32, ptr %arg_type_ids, align 4, !dbg !68, !tbaa !69
  %4 = getelementptr inbounds i32, ptr %arg_type_ids, i64 1, !dbg !68
  %K_handle.code = load i32, ptr %4, align 4, !dbg !68, !tbaa !80
  %5 = getelementptr inbounds i32, ptr %arg_type_ids, i64 2, !dbg !68
  %V_handle.code = load i32, ptr %5, align 4, !dbg !68, !tbaa !82
  %6 = getelementptr inbounds i32, ptr %arg_type_ids, i64 3, !dbg !68
  %O_handle.code = load i32, ptr %6, align 4, !dbg !68, !tbaa !85
  %7 = getelementptr inbounds i32, ptr %arg_type_ids, i64 4, !dbg !68
  %L_handle.code = load i32, ptr %7, align 4, !dbg !68, !tbaa !87
  %Q_handle = load ptr, ptr %args, align 8, !dbg !68
  %8 = getelementptr inbounds %0, ptr %args, i64 1, !dbg !68
  %K_handle = load ptr, ptr %8, align 8, !dbg !68
  %9 = getelementptr inbounds %0, ptr %args, i64 2, !dbg !68
  %V_handle = load ptr, ptr %9, align 8, !dbg !68
  %10 = getelementptr inbounds %0, ptr %args, i64 3, !dbg !68
  %O_handle = load ptr, ptr %10, align 8, !dbg !68
  %11 = getelementptr inbounds %0, ptr %args, i64 4, !dbg !68
  %L_handle = load ptr, ptr %11, align 8, !dbg !68
  %12 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 4, !dbg !68
  %13 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 5, !dbg !68
  %14 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 1, !dbg !68
  %15 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 4, !dbg !68
  %16 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 5, !dbg !68
  %17 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 4, !dbg !68
  %18 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 5, !dbg !68
  %19 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 4, !dbg !68
  %20 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 5, !dbg !68
  %Q = load ptr, ptr %Q_handle, align 8, !dbg !68
  %forward_op.Q_handle.shape = load ptr, ptr %12, align 8, !dbg !68
  %forward_op.Q_handle.strides = load ptr, ptr %13, align 8, !dbg !68
  %dev_id = load i32, ptr %14, align 4, !dbg !68
  %21 = sext i32 %dev_id to i64, !dbg !68
  %K = load ptr, ptr %K_handle, align 8, !dbg !68
  %forward_op.K_handle.shape = load ptr, ptr %15, align 8, !dbg !68
  %forward_op.K_handle.strides = load ptr, ptr %16, align 8, !dbg !68
  %V = load ptr, ptr %V_handle, align 8, !dbg !68
  %forward_op.V_handle.shape = load ptr, ptr %17, align 8, !dbg !68
  %forward_op.V_handle.strides = load ptr, ptr %18, align 8, !dbg !68
  %O = load ptr, ptr %O_handle, align 8, !dbg !68
  %forward_op.O_handle.shape = load ptr, ptr %19, align 8, !dbg !68
  %forward_op.O_handle.strides = load ptr, ptr %20, align 8, !dbg !68
  %L = load ptr, ptr %L_handle, align 8, !dbg !68
  %22 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 4, !dbg !68
  %forward_op.L_handle.shape = load ptr, ptr %22, align 8, !dbg !68
  %23 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 5, !dbg !68
  %forward_op.L_handle.strides = load ptr, ptr %23, align 8, !dbg !68
  switch i32 %Q_handle.code, label %assert_fail1 [
    i32 13, label %assert_end2
    i32 7, label %assert_end2
    i32 4, label %assert_end2
    i32 3, label %assert_end2
  ], !dbg !68

assert_fail1:                                     ; preds = %assert_end
  %24 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %24(ptr nonnull @.str.103), !dbg !68
  br label %common.ret

assert_end2:                                      ; preds = %assert_end, %assert_end, %assert_end, %assert_end
  switch i32 %K_handle.code, label %assert_fail3 [
    i32 13, label %assert_end4
    i32 7, label %assert_end4
    i32 4, label %assert_end4
    i32 3, label %assert_end4
  ], !dbg !68

assert_fail3:                                     ; preds = %assert_end2
  %25 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %25(ptr nonnull @.str.104), !dbg !68
  br label %common.ret

assert_end4:                                      ; preds = %assert_end2, %assert_end2, %assert_end2, %assert_end2
  switch i32 %V_handle.code, label %assert_fail5 [
    i32 13, label %assert_end6
    i32 7, label %assert_end6
    i32 4, label %assert_end6
    i32 3, label %assert_end6
  ], !dbg !68

assert_fail5:                                     ; preds = %assert_end4
  %26 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %26(ptr nonnull @.str.105), !dbg !68
  br label %common.ret

assert_end6:                                      ; preds = %assert_end4, %assert_end4, %assert_end4, %assert_end4
  switch i32 %O_handle.code, label %assert_fail7 [
    i32 13, label %assert_end8
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 3, label %assert_end8
  ], !dbg !68

assert_fail7:                                     ; preds = %assert_end6
  %27 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %27(ptr nonnull @.str.106), !dbg !68
  br label %common.ret

assert_end8:                                      ; preds = %assert_end6, %assert_end6, %assert_end6, %assert_end6
  switch i32 %L_handle.code, label %assert_fail9 [
    i32 13, label %assert_end10
    i32 7, label %assert_end10
    i32 4, label %assert_end10
    i32 3, label %assert_end10
  ], !dbg !68

assert_fail9:                                     ; preds = %assert_end8
  %28 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %28(ptr nonnull @.str.107), !dbg !68
  br label %common.ret

assert_end10:                                     ; preds = %assert_end8, %assert_end8, %assert_end8, %assert_end8
  %29 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 2, !dbg !68
  %30 = load i32, ptr %29, align 4, !dbg !68
  %31 = icmp eq i32 %30, 4, !dbg !68
  br i1 %31, label %assert_end14, label %assert_fail11, !dbg !68, !prof !19

assert_fail11:                                    ; preds = %assert_end10
  %32 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %32(ptr nonnull @.str.108), !dbg !68
  br label %common.ret

assert_end14:                                     ; preds = %assert_end10
  %33 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 0, !dbg !68
  %34 = load i8, ptr %33, align 1, !dbg !68
  %35 = icmp eq i8 %34, 2, !dbg !68
  %36 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 1, !dbg !68
  %37 = load i8, ptr %36, align 1, !dbg !68
  %38 = icmp eq i8 %37, 16, !dbg !68
  %39 = and i1 %35, %38, !dbg !68
  %40 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 2, !dbg !68
  %41 = load i16, ptr %40, align 2, !dbg !68
  %42 = icmp eq i16 %41, 1, !dbg !68
  %43 = and i1 %39, %42, !dbg !68
  br i1 %43, label %assert_end16, label %assert_fail15, !dbg !68, !prof !19

assert_fail15:                                    ; preds = %assert_end14
  %44 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %44(ptr nonnull @.str.109), !dbg !68
  br label %common.ret

assert_end16:                                     ; preds = %assert_end14
  %45 = load i64, ptr %forward_op.Q_handle.shape, align 8, !dbg !68, !tbaa !57
  %46 = and i64 %45, 4294967295, !dbg !68
  %47 = icmp eq i64 %46, 6, !dbg !68
  br i1 %47, label %assert_end18, label %assert_fail17, !dbg !68, !prof !19

assert_fail17:                                    ; preds = %assert_end16
  %48 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %48(ptr nonnull @.str.110), !dbg !68
  br label %common.ret

assert_end18:                                     ; preds = %assert_end16
  %49 = getelementptr inbounds i64, ptr %forward_op.Q_handle.shape, i64 1, !dbg !68
  %50 = load i64, ptr %49, align 8, !dbg !68, !tbaa !57
  %51 = and i64 %50, 4294967295, !dbg !68
  %52 = icmp eq i64 %51, 32, !dbg !68
  br i1 %52, label %assert_end20, label %assert_fail19, !dbg !68, !prof !19

assert_fail19:                                    ; preds = %assert_end18
  %53 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %53(ptr nonnull @.str.111), !dbg !68
  br label %common.ret

assert_end20:                                     ; preds = %assert_end18
  %54 = getelementptr inbounds i64, ptr %forward_op.Q_handle.shape, i64 2, !dbg !68
  %55 = load i64, ptr %54, align 8, !dbg !68, !tbaa !57
  %56 = and i64 %55, 4294967295, !dbg !68
  %57 = icmp eq i64 %56, 512, !dbg !68
  br i1 %57, label %assert_end22, label %assert_fail21, !dbg !68, !prof !19

assert_fail21:                                    ; preds = %assert_end20
  %58 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %58(ptr nonnull @.str.112), !dbg !68
  br label %common.ret

assert_end22:                                     ; preds = %assert_end20
  %59 = getelementptr inbounds i64, ptr %forward_op.Q_handle.shape, i64 3, !dbg !68
  %60 = load i64, ptr %59, align 8, !dbg !68, !tbaa !57
  %61 = and i64 %60, 4294967295, !dbg !68
  %62 = icmp eq i64 %61, 128, !dbg !68
  br i1 %62, label %assert_end24, label %assert_fail23, !dbg !68, !prof !19

assert_fail23:                                    ; preds = %assert_end22
  %63 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %63(ptr nonnull @.str.113), !dbg !68
  br label %common.ret

assert_end24:                                     ; preds = %assert_end22
  %.not = icmp eq ptr %forward_op.Q_handle.strides, null, !dbg !68
  br i1 %.not, label %if_end, label %if_then, !dbg !68, !prof !59

if_then:                                          ; preds = %assert_end24
  %64 = load <4 x i64>, ptr %forward_op.Q_handle.strides, align 8, !dbg !68, !tbaa !57
  %65 = and <4 x i64> %64, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !68
  %66 = icmp ne <4 x i64> %65, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !68
  %67 = bitcast <4 x i1> %66 to i4, !dbg !68
  %68 = icmp eq i4 %67, 0, !dbg !68
  br i1 %68, label %if_end, label %assert_fail25, !dbg !68, !prof !19

if_end:                                           ; preds = %if_then, %assert_end24
  %69 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 6, !dbg !68
  %70 = load i64, ptr %69, align 8, !dbg !68
  %71 = icmp eq i64 %70, 0, !dbg !68
  br i1 %71, label %assert_end28, label %assert_fail27, !dbg !68, !prof !19

assert_fail25:                                    ; preds = %if_then
  %72 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %72(ptr nonnull @.str.114), !dbg !68
  br label %common.ret

assert_fail27:                                    ; preds = %if_end
  %73 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %73(ptr nonnull @.str.115), !dbg !68
  br label %common.ret

assert_end28:                                     ; preds = %if_end
  %74 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 0, !dbg !68
  %75 = load i32, ptr %74, align 4, !dbg !68
  %76 = icmp eq i32 %75, 8, !dbg !68
  br i1 %76, label %assert_end30, label %assert_fail29, !dbg !68, !prof !19

assert_fail29:                                    ; preds = %assert_end28
  %77 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %77(ptr nonnull @.str.116), !dbg !68
  br label %common.ret

assert_end30:                                     ; preds = %assert_end28
  %78 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 2, !dbg !68
  %79 = load i32, ptr %78, align 4, !dbg !68
  %80 = icmp eq i32 %79, 4, !dbg !68
  br i1 %80, label %assert_end34, label %assert_fail31, !dbg !68, !prof !19

assert_fail31:                                    ; preds = %assert_end30
  %81 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %81(ptr nonnull @.str.117), !dbg !68
  br label %common.ret

assert_end34:                                     ; preds = %assert_end30
  %82 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 0, !dbg !68
  %83 = load i8, ptr %82, align 1, !dbg !68
  %84 = icmp eq i8 %83, 2, !dbg !68
  %85 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 1, !dbg !68
  %86 = load i8, ptr %85, align 1, !dbg !68
  %87 = icmp eq i8 %86, 16, !dbg !68
  %88 = and i1 %84, %87, !dbg !68
  %89 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 2, !dbg !68
  %90 = load i16, ptr %89, align 2, !dbg !68
  %91 = icmp eq i16 %90, 1, !dbg !68
  %92 = and i1 %88, %91, !dbg !68
  br i1 %92, label %assert_end36, label %assert_fail35, !dbg !68, !prof !19

assert_fail35:                                    ; preds = %assert_end34
  %93 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %93(ptr nonnull @.str.118), !dbg !68
  br label %common.ret

assert_end36:                                     ; preds = %assert_end34
  %94 = load i64, ptr %forward_op.K_handle.shape, align 8, !dbg !68, !tbaa !57
  %95 = and i64 %94, 4294967295, !dbg !68
  %96 = icmp eq i64 %95, 6, !dbg !68
  br i1 %96, label %assert_end38, label %assert_fail37, !dbg !68, !prof !19

assert_fail37:                                    ; preds = %assert_end36
  %97 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %97(ptr nonnull @.str.119), !dbg !68
  br label %common.ret

assert_end38:                                     ; preds = %assert_end36
  %98 = getelementptr inbounds i64, ptr %forward_op.K_handle.shape, i64 1, !dbg !68
  %99 = load i64, ptr %98, align 8, !dbg !68, !tbaa !57
  %100 = and i64 %99, 4294967295, !dbg !68
  %101 = icmp eq i64 %100, 32, !dbg !68
  br i1 %101, label %assert_end40, label %assert_fail39, !dbg !68, !prof !19

assert_fail39:                                    ; preds = %assert_end38
  %102 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %102(ptr nonnull @.str.120), !dbg !68
  br label %common.ret

assert_end40:                                     ; preds = %assert_end38
  %103 = getelementptr inbounds i64, ptr %forward_op.K_handle.shape, i64 2, !dbg !68
  %104 = load i64, ptr %103, align 8, !dbg !68, !tbaa !57
  %105 = and i64 %104, 4294967295, !dbg !68
  %106 = icmp eq i64 %105, 512, !dbg !68
  br i1 %106, label %assert_end42, label %assert_fail41, !dbg !68, !prof !19

assert_fail41:                                    ; preds = %assert_end40
  %107 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %107(ptr nonnull @.str.121), !dbg !68
  br label %common.ret

assert_end42:                                     ; preds = %assert_end40
  %108 = getelementptr inbounds i64, ptr %forward_op.K_handle.shape, i64 3, !dbg !68
  %109 = load i64, ptr %108, align 8, !dbg !68, !tbaa !57
  %110 = and i64 %109, 4294967295, !dbg !68
  %111 = icmp eq i64 %110, 128, !dbg !68
  br i1 %111, label %assert_end44, label %assert_fail43, !dbg !68, !prof !19

assert_fail43:                                    ; preds = %assert_end42
  %112 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %112(ptr nonnull @.str.122), !dbg !68
  br label %common.ret

assert_end44:                                     ; preds = %assert_end42
  %.not133 = icmp eq ptr %forward_op.K_handle.strides, null, !dbg !68
  br i1 %.not133, label %if_end46, label %if_then45, !dbg !68, !prof !59

if_then45:                                        ; preds = %assert_end44
  %113 = load <4 x i64>, ptr %forward_op.K_handle.strides, align 8, !dbg !68, !tbaa !57
  %114 = and <4 x i64> %113, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !68
  %115 = icmp ne <4 x i64> %114, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !68
  %116 = bitcast <4 x i1> %115 to i4, !dbg !68
  %117 = icmp eq i4 %116, 0, !dbg !68
  br i1 %117, label %if_end46, label %assert_fail47, !dbg !68, !prof !19

if_end46:                                         ; preds = %if_then45, %assert_end44
  %118 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 6, !dbg !68
  %119 = load i64, ptr %118, align 8, !dbg !68
  %120 = icmp eq i64 %119, 0, !dbg !68
  br i1 %120, label %assert_end50, label %assert_fail49, !dbg !68, !prof !19

assert_fail47:                                    ; preds = %if_then45
  %121 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %121(ptr nonnull @.str.123), !dbg !68
  br label %common.ret

assert_fail49:                                    ; preds = %if_end46
  %122 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %122(ptr nonnull @.str.124), !dbg !68
  br label %common.ret

assert_end50:                                     ; preds = %if_end46
  %123 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 0, !dbg !68
  %124 = load i32, ptr %123, align 4, !dbg !68
  %125 = icmp eq i32 %124, 8, !dbg !68
  br i1 %125, label %assert_end52, label %assert_fail51, !dbg !68, !prof !19

assert_fail51:                                    ; preds = %assert_end50
  %126 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %126(ptr nonnull @.str.125), !dbg !68
  br label %common.ret

assert_end52:                                     ; preds = %assert_end50
  %127 = trunc i64 %21 to i32
  %128 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 1, !dbg !68
  %129 = load i32, ptr %128, align 4, !dbg !68
  %130 = icmp eq i32 %127, %129, !dbg !68
  br i1 %130, label %assert_end54, label %assert_fail53, !dbg !68, !prof !19

assert_fail53:                                    ; preds = %assert_end52
  %131 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %131(ptr nonnull @.str.126), !dbg !68
  br label %common.ret

assert_end54:                                     ; preds = %assert_end52
  %132 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 2, !dbg !68
  %133 = load i32, ptr %132, align 4, !dbg !68
  %134 = icmp eq i32 %133, 4, !dbg !68
  br i1 %134, label %assert_end58, label %assert_fail55, !dbg !68, !prof !19

assert_fail55:                                    ; preds = %assert_end54
  %135 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %135(ptr nonnull @.str.127), !dbg !68
  br label %common.ret

assert_end58:                                     ; preds = %assert_end54
  %136 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 0, !dbg !68
  %137 = load i8, ptr %136, align 1, !dbg !68
  %138 = icmp eq i8 %137, 2, !dbg !68
  %139 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 1, !dbg !68
  %140 = load i8, ptr %139, align 1, !dbg !68
  %141 = icmp eq i8 %140, 16, !dbg !68
  %142 = and i1 %138, %141, !dbg !68
  %143 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 2, !dbg !68
  %144 = load i16, ptr %143, align 2, !dbg !68
  %145 = icmp eq i16 %144, 1, !dbg !68
  %146 = and i1 %142, %145, !dbg !68
  br i1 %146, label %assert_end60, label %assert_fail59, !dbg !68, !prof !19

assert_fail59:                                    ; preds = %assert_end58
  %147 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %147(ptr nonnull @.str.128), !dbg !68
  br label %common.ret

assert_end60:                                     ; preds = %assert_end58
  %148 = load i64, ptr %forward_op.V_handle.shape, align 8, !dbg !68, !tbaa !57
  %149 = and i64 %148, 4294967295, !dbg !68
  %150 = icmp eq i64 %149, 6, !dbg !68
  br i1 %150, label %assert_end62, label %assert_fail61, !dbg !68, !prof !19

assert_fail61:                                    ; preds = %assert_end60
  %151 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %151(ptr nonnull @.str.129), !dbg !68
  br label %common.ret

assert_end62:                                     ; preds = %assert_end60
  %152 = getelementptr inbounds i64, ptr %forward_op.V_handle.shape, i64 1, !dbg !68
  %153 = load i64, ptr %152, align 8, !dbg !68, !tbaa !57
  %154 = and i64 %153, 4294967295, !dbg !68
  %155 = icmp eq i64 %154, 32, !dbg !68
  br i1 %155, label %assert_end64, label %assert_fail63, !dbg !68, !prof !19

assert_fail63:                                    ; preds = %assert_end62
  %156 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %156(ptr nonnull @.str.130), !dbg !68
  br label %common.ret

assert_end64:                                     ; preds = %assert_end62
  %157 = getelementptr inbounds i64, ptr %forward_op.V_handle.shape, i64 2, !dbg !68
  %158 = load i64, ptr %157, align 8, !dbg !68, !tbaa !57
  %159 = and i64 %158, 4294967295, !dbg !68
  %160 = icmp eq i64 %159, 512, !dbg !68
  br i1 %160, label %assert_end66, label %assert_fail65, !dbg !68, !prof !19

assert_fail65:                                    ; preds = %assert_end64
  %161 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %161(ptr nonnull @.str.131), !dbg !68
  br label %common.ret

assert_end66:                                     ; preds = %assert_end64
  %162 = getelementptr inbounds i64, ptr %forward_op.V_handle.shape, i64 3, !dbg !68
  %163 = load i64, ptr %162, align 8, !dbg !68, !tbaa !57
  %164 = and i64 %163, 4294967295, !dbg !68
  %165 = icmp eq i64 %164, 128, !dbg !68
  br i1 %165, label %assert_end68, label %assert_fail67, !dbg !68, !prof !19

assert_fail67:                                    ; preds = %assert_end66
  %166 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %166(ptr nonnull @.str.132), !dbg !68
  br label %common.ret

assert_end68:                                     ; preds = %assert_end66
  %.not134 = icmp eq ptr %forward_op.V_handle.strides, null, !dbg !68
  br i1 %.not134, label %if_end70, label %if_then69, !dbg !68, !prof !59

if_then69:                                        ; preds = %assert_end68
  %167 = load <4 x i64>, ptr %forward_op.V_handle.strides, align 8, !dbg !68, !tbaa !57
  %168 = and <4 x i64> %167, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !68
  %169 = icmp ne <4 x i64> %168, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !68
  %170 = bitcast <4 x i1> %169 to i4, !dbg !68
  %171 = icmp eq i4 %170, 0, !dbg !68
  br i1 %171, label %if_end70, label %assert_fail71, !dbg !68, !prof !19

if_end70:                                         ; preds = %if_then69, %assert_end68
  %172 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 6, !dbg !68
  %173 = load i64, ptr %172, align 8, !dbg !68
  %174 = icmp eq i64 %173, 0, !dbg !68
  br i1 %174, label %assert_end74, label %assert_fail73, !dbg !68, !prof !19

assert_fail71:                                    ; preds = %if_then69
  %175 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %175(ptr nonnull @.str.133), !dbg !68
  br label %common.ret

assert_fail73:                                    ; preds = %if_end70
  %176 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %176(ptr nonnull @.str.134), !dbg !68
  br label %common.ret

assert_end74:                                     ; preds = %if_end70
  %177 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 0, !dbg !68
  %178 = load i32, ptr %177, align 4, !dbg !68
  %179 = icmp eq i32 %178, 8, !dbg !68
  br i1 %179, label %assert_end76, label %assert_fail75, !dbg !68, !prof !19

assert_fail75:                                    ; preds = %assert_end74
  %180 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %180(ptr nonnull @.str.135), !dbg !68
  br label %common.ret

assert_end76:                                     ; preds = %assert_end74
  %181 = trunc i64 %21 to i32
  %182 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 1, !dbg !68
  %183 = load i32, ptr %182, align 4, !dbg !68
  %184 = icmp eq i32 %181, %183, !dbg !68
  br i1 %184, label %assert_end78, label %assert_fail77, !dbg !68, !prof !19

assert_fail77:                                    ; preds = %assert_end76
  %185 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %185(ptr nonnull @.str.136), !dbg !68
  br label %common.ret

assert_end78:                                     ; preds = %assert_end76
  %186 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 2, !dbg !68
  %187 = load i32, ptr %186, align 4, !dbg !68
  %188 = icmp eq i32 %187, 4, !dbg !68
  br i1 %188, label %assert_end82, label %assert_fail79, !dbg !68, !prof !19

assert_fail79:                                    ; preds = %assert_end78
  %189 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %189(ptr nonnull @.str.137), !dbg !68
  br label %common.ret

assert_end82:                                     ; preds = %assert_end78
  %190 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 0, !dbg !68
  %191 = load i8, ptr %190, align 1, !dbg !68
  %192 = icmp eq i8 %191, 2, !dbg !68
  %193 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 1, !dbg !68
  %194 = load i8, ptr %193, align 1, !dbg !68
  %195 = icmp eq i8 %194, 16, !dbg !68
  %196 = and i1 %192, %195, !dbg !68
  %197 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 3, i32 2, !dbg !68
  %198 = load i16, ptr %197, align 2, !dbg !68
  %199 = icmp eq i16 %198, 1, !dbg !68
  %200 = and i1 %196, %199, !dbg !68
  br i1 %200, label %assert_end84, label %assert_fail83, !dbg !68, !prof !19

assert_fail83:                                    ; preds = %assert_end82
  %201 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %201(ptr nonnull @.str.138), !dbg !68
  br label %common.ret

assert_end84:                                     ; preds = %assert_end82
  %202 = load i64, ptr %forward_op.O_handle.shape, align 8, !dbg !68, !tbaa !57
  %203 = and i64 %202, 4294967295, !dbg !68
  %204 = icmp eq i64 %203, 6, !dbg !68
  br i1 %204, label %assert_end86, label %assert_fail85, !dbg !68, !prof !19

assert_fail85:                                    ; preds = %assert_end84
  %205 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %205(ptr nonnull @.str.139), !dbg !68
  br label %common.ret

assert_end86:                                     ; preds = %assert_end84
  %206 = getelementptr inbounds i64, ptr %forward_op.O_handle.shape, i64 1, !dbg !68
  %207 = load i64, ptr %206, align 8, !dbg !68, !tbaa !57
  %208 = and i64 %207, 4294967295, !dbg !68
  %209 = icmp eq i64 %208, 32, !dbg !68
  br i1 %209, label %assert_end88, label %assert_fail87, !dbg !68, !prof !19

assert_fail87:                                    ; preds = %assert_end86
  %210 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %210(ptr nonnull @.str.140), !dbg !68
  br label %common.ret

assert_end88:                                     ; preds = %assert_end86
  %211 = getelementptr inbounds i64, ptr %forward_op.O_handle.shape, i64 2, !dbg !68
  %212 = load i64, ptr %211, align 8, !dbg !68, !tbaa !57
  %213 = and i64 %212, 4294967295, !dbg !68
  %214 = icmp eq i64 %213, 512, !dbg !68
  br i1 %214, label %assert_end90, label %assert_fail89, !dbg !68, !prof !19

assert_fail89:                                    ; preds = %assert_end88
  %215 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %215(ptr nonnull @.str.141), !dbg !68
  br label %common.ret

assert_end90:                                     ; preds = %assert_end88
  %216 = getelementptr inbounds i64, ptr %forward_op.O_handle.shape, i64 3, !dbg !68
  %217 = load i64, ptr %216, align 8, !dbg !68, !tbaa !57
  %218 = and i64 %217, 4294967295, !dbg !68
  %219 = icmp eq i64 %218, 128, !dbg !68
  br i1 %219, label %assert_end92, label %assert_fail91, !dbg !68, !prof !19

assert_fail91:                                    ; preds = %assert_end90
  %220 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %220(ptr nonnull @.str.142), !dbg !68
  br label %common.ret

assert_end92:                                     ; preds = %assert_end90
  %.not135 = icmp eq ptr %forward_op.O_handle.strides, null, !dbg !68
  br i1 %.not135, label %if_end94, label %if_then93, !dbg !68, !prof !59

if_then93:                                        ; preds = %assert_end92
  %221 = load <4 x i64>, ptr %forward_op.O_handle.strides, align 8, !dbg !68, !tbaa !57
  %222 = and <4 x i64> %221, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>, !dbg !68
  %223 = icmp ne <4 x i64> %222, <i64 2097152, i64 65536, i64 128, i64 1>, !dbg !68
  %224 = bitcast <4 x i1> %223 to i4, !dbg !68
  %225 = icmp eq i4 %224, 0, !dbg !68
  br i1 %225, label %if_end94, label %assert_fail95, !dbg !68, !prof !19

if_end94:                                         ; preds = %if_then93, %assert_end92
  %226 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 6, !dbg !68
  %227 = load i64, ptr %226, align 8, !dbg !68
  %228 = icmp eq i64 %227, 0, !dbg !68
  br i1 %228, label %assert_end98, label %assert_fail97, !dbg !68, !prof !19

assert_fail95:                                    ; preds = %if_then93
  %229 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %229(ptr nonnull @.str.143), !dbg !68
  br label %common.ret

assert_fail97:                                    ; preds = %if_end94
  %230 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %230(ptr nonnull @.str.144), !dbg !68
  br label %common.ret

assert_end98:                                     ; preds = %if_end94
  %231 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 1, i32 0, !dbg !68
  %232 = load i32, ptr %231, align 4, !dbg !68
  %233 = icmp eq i32 %232, 8, !dbg !68
  br i1 %233, label %assert_end100, label %assert_fail99, !dbg !68, !prof !19

assert_fail99:                                    ; preds = %assert_end98
  %234 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %234(ptr nonnull @.str.145), !dbg !68
  br label %common.ret

assert_end100:                                    ; preds = %assert_end98
  %235 = trunc i64 %21 to i32
  %236 = getelementptr inbounds %1, ptr %O_handle, i64 0, i32 1, i32 1, !dbg !68
  %237 = load i32, ptr %236, align 4, !dbg !68
  %238 = icmp eq i32 %235, %237, !dbg !68
  br i1 %238, label %assert_end102, label %assert_fail101, !dbg !68, !prof !19

assert_fail101:                                   ; preds = %assert_end100
  %239 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %239(ptr nonnull @.str.146), !dbg !68
  br label %common.ret

assert_end102:                                    ; preds = %assert_end100
  %240 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 2, !dbg !68
  %241 = load i32, ptr %240, align 4, !dbg !68
  %242 = icmp eq i32 %241, 3, !dbg !68
  br i1 %242, label %assert_end106, label %assert_fail103, !dbg !68, !prof !19

assert_fail103:                                   ; preds = %assert_end102
  %243 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %243(ptr nonnull @.str.147), !dbg !68
  br label %common.ret

assert_end106:                                    ; preds = %assert_end102
  %244 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 0, !dbg !68
  %245 = load i8, ptr %244, align 1, !dbg !68
  %246 = icmp eq i8 %245, 2, !dbg !68
  %247 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 1, !dbg !68
  %248 = load i8, ptr %247, align 1, !dbg !68
  %249 = icmp eq i8 %248, 16, !dbg !68
  %250 = and i1 %246, %249, !dbg !68
  %251 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 3, i32 2, !dbg !68
  %252 = load i16, ptr %251, align 2, !dbg !68
  %253 = icmp eq i16 %252, 1, !dbg !68
  %254 = and i1 %250, %253, !dbg !68
  br i1 %254, label %assert_end108, label %assert_fail107, !dbg !68, !prof !19

assert_fail107:                                   ; preds = %assert_end106
  %255 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %255(ptr nonnull @.str.148), !dbg !68
  br label %common.ret

assert_end108:                                    ; preds = %assert_end106
  %256 = load i64, ptr %forward_op.L_handle.shape, align 8, !dbg !68, !tbaa !57
  %257 = and i64 %256, 4294967295, !dbg !68
  %258 = icmp eq i64 %257, 6, !dbg !68
  br i1 %258, label %assert_end110, label %assert_fail109, !dbg !68, !prof !19

assert_fail109:                                   ; preds = %assert_end108
  %259 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %259(ptr nonnull @.str.149), !dbg !68
  br label %common.ret

assert_end110:                                    ; preds = %assert_end108
  %260 = getelementptr inbounds i64, ptr %forward_op.L_handle.shape, i64 1, !dbg !68
  %261 = load i64, ptr %260, align 8, !dbg !68, !tbaa !57
  %262 = and i64 %261, 4294967295, !dbg !68
  %263 = icmp eq i64 %262, 32, !dbg !68
  br i1 %263, label %assert_end112, label %assert_fail111, !dbg !68, !prof !19

assert_fail111:                                   ; preds = %assert_end110
  %264 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %264(ptr nonnull @.str.150), !dbg !68
  br label %common.ret

assert_end112:                                    ; preds = %assert_end110
  %265 = getelementptr inbounds i64, ptr %forward_op.L_handle.shape, i64 2, !dbg !68
  %266 = load i64, ptr %265, align 8, !dbg !68, !tbaa !57
  %267 = and i64 %266, 4294967295, !dbg !68
  %268 = icmp eq i64 %267, 512, !dbg !68
  br i1 %268, label %assert_end114, label %assert_fail113, !dbg !68, !prof !19

assert_fail113:                                   ; preds = %assert_end112
  %269 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %269(ptr nonnull @.str.151), !dbg !68
  br label %common.ret

assert_end114:                                    ; preds = %assert_end112
  %.not136 = icmp eq ptr %forward_op.L_handle.strides, null, !dbg !68
  br i1 %.not136, label %if_end116, label %if_then115, !dbg !68, !prof !59

if_then115:                                       ; preds = %assert_end114
  %270 = getelementptr inbounds i64, ptr %forward_op.L_handle.strides, i64 2, !dbg !68
  %271 = load i64, ptr %270, align 8, !dbg !68, !tbaa !57
  %272 = and i64 %271, 4294967295, !dbg !68
  %273 = icmp eq i64 %272, 1, !dbg !68
  %274 = getelementptr inbounds i64, ptr %forward_op.L_handle.strides, i64 1, !dbg !68
  %275 = load i64, ptr %274, align 8, !dbg !68, !tbaa !57
  %276 = and i64 %275, 4294967295, !dbg !68
  %277 = icmp eq i64 %276, 512, !dbg !68
  %278 = and i1 %273, %277, !dbg !68
  %279 = load i64, ptr %forward_op.L_handle.strides, align 8, !dbg !68, !tbaa !57
  %280 = and i64 %279, 4294967295, !dbg !68
  %281 = icmp eq i64 %280, 16384, !dbg !68
  %282 = and i1 %278, %281, !dbg !68
  br i1 %282, label %if_end116, label %assert_fail117, !dbg !68, !prof !19

if_end116:                                        ; preds = %if_then115, %assert_end114
  %283 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 6, !dbg !68
  %284 = load i64, ptr %283, align 8, !dbg !68
  %285 = icmp eq i64 %284, 0, !dbg !68
  br i1 %285, label %assert_end120, label %assert_fail119, !dbg !68, !prof !19

assert_fail117:                                   ; preds = %if_then115
  %286 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %286(ptr nonnull @.str.152), !dbg !68
  br label %common.ret

assert_fail119:                                   ; preds = %if_end116
  %287 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %287(ptr nonnull @.str.153), !dbg !68
  br label %common.ret

assert_end120:                                    ; preds = %if_end116
  %288 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 1, i32 0, !dbg !68
  %289 = load i32, ptr %288, align 4, !dbg !68
  %290 = icmp eq i32 %289, 8, !dbg !68
  br i1 %290, label %assert_end122, label %assert_fail121, !dbg !68, !prof !19

assert_fail121:                                   ; preds = %assert_end120
  %291 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %291(ptr nonnull @.str.154), !dbg !68
  br label %common.ret

assert_end122:                                    ; preds = %assert_end120
  %292 = trunc i64 %21 to i32
  %293 = getelementptr inbounds %1, ptr %L_handle, i64 0, i32 1, i32 1, !dbg !68
  %294 = load i32, ptr %293, align 4, !dbg !68
  %295 = icmp eq i32 %292, %294, !dbg !68
  br i1 %295, label %assert_end124, label %assert_fail123, !dbg !68, !prof !19

assert_fail123:                                   ; preds = %assert_end122
  %296 = load ptr, ptr @__TVMAPISetLastError, align 8, !dbg !68, !tbaa !20
  tail call void %296(ptr nonnull @.str.155), !dbg !68
  br label %common.ret

assert_end124:                                    ; preds = %assert_end122
  store i64 8, ptr %stack_value131, align 8, !dbg !68
  %297 = getelementptr inbounds i64, ptr %stack_value131, i64 1, !dbg !68
  store i64 %21, ptr %297, align 8, !dbg !68
  store <2 x i32> zeroinitializer, ptr %stack_tcode132, align 16, !dbg !68, !tbaa !57
  %298 = getelementptr inbounds %0, ptr %stack_value131, i64 2, !dbg !68
  %299 = getelementptr inbounds i32, ptr %stack_tcode132, i64 2, !dbg !68
  %300 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !68, !tbaa !20
  %301 = load ptr, ptr @_MergedGlobals, align 8, !dbg !68
  %.not137 = icmp eq ptr %301, null, !dbg !68
  br i1 %.not137, label %handle_init, label %handle_init_end, !dbg !68, !prof !59

handle_init:                                      ; preds = %assert_end124
  %302 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !68, !tbaa !20
  %303 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !68, !tbaa !20
  %304 = call i32 %303(ptr %302, ptr nonnull @.str.98, ptr nonnull %1), !dbg !68
  %305 = icmp eq i32 %304, 0, !dbg !68
  br i1 %305, label %call_end, label %common.ret, !dbg !68, !prof !19

handle_init_end:                                  ; preds = %call_end, %assert_end124
  %306 = phi ptr [ %301, %assert_end124 ], [ %309, %call_end ], !dbg !68
  %307 = call i32 %300(ptr %306, ptr nonnull %stack_value131, ptr nonnull %stack_tcode132, i32 2, ptr nonnull %298, ptr nonnull %299), !dbg !68
  %308 = icmp eq i32 %307, 0, !dbg !68
  br i1 %308, label %call_end126, label %common.ret, !dbg !68, !prof !19

call_end:                                         ; preds = %handle_init
  %309 = load ptr, ptr %1, align 8, !dbg !68
  store ptr %309, ptr @_MergedGlobals, align 8, !dbg !68
  br label %handle_init_end, !dbg !68

call_end126:                                      ; preds = %handle_init_end
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0), !dbg !68
  store ptr %K, ptr %stack_value131, align 8, !dbg !68
  %310 = insertelement <4 x ptr> poison, ptr %K, i64 0, !dbg !68
  %311 = insertelement <4 x ptr> %310, ptr %L, i64 1, !dbg !68
  %312 = insertelement <4 x ptr> %311, ptr %O, i64 2, !dbg !68
  %313 = insertelement <4 x ptr> %312, ptr %Q, i64 3, !dbg !68
  %314 = icmp eq <4 x ptr> %313, zeroinitializer, !dbg !68
  %sunkaddr = getelementptr inbounds i8, ptr %stack_value131, i64 8, !dbg !68
  store ptr %L, ptr %sunkaddr, align 8, !dbg !68
  %sunkaddr138 = getelementptr inbounds i8, ptr %stack_value131, i64 16, !dbg !68
  store ptr %O, ptr %sunkaddr138, align 8, !dbg !68
  %315 = getelementptr inbounds %0, ptr %stack_value131, i64 3, !dbg !68
  store ptr %Q, ptr %315, align 8, !dbg !68
  %316 = select <4 x i1> %314, <4 x i32> <i32 4, i32 4, i32 4, i32 4>, <4 x i32> <i32 3, i32 3, i32 3, i32 3>, !dbg !68
  store <4 x i32> %316, ptr %stack_tcode132, align 16, !dbg !68
  %317 = getelementptr inbounds %0, ptr %stack_value131, i64 4, !dbg !68
  store ptr %V, ptr %317, align 8, !dbg !68
  %318 = icmp eq ptr %V, null, !dbg !68
  %.sink18.i = select i1 %318, i32 4, i32 3, !dbg !68
  %319 = getelementptr inbounds i32, ptr %stack_tcode132, i64 4, !dbg !68
  store i32 %.sink18.i, ptr %319, align 16, !dbg !68
  %320 = getelementptr inbounds i64, ptr %stack_value131, i64 5, !dbg !68
  %321 = getelementptr inbounds i32, ptr %stack_tcode132, i64 5, !dbg !68
  store <2 x i64> <i64 12288, i64 8>, ptr %320, align 8, !dbg !68
  store <2 x i32> zeroinitializer, ptr %321, align 4, !dbg !68, !tbaa !57
  %322 = getelementptr inbounds i64, ptr %stack_value131, i64 7, !dbg !68
  store i64 16, ptr %322, align 8, !dbg !68
  %323 = getelementptr inbounds i32, ptr %stack_tcode132, i64 7, !dbg !68
  store i32 0, ptr %323, align 4, !dbg !68, !tbaa !57
  %324 = getelementptr inbounds %0, ptr %stack_value131, i64 8, !dbg !68
  %325 = getelementptr inbounds i32, ptr %stack_tcode132, i64 8, !dbg !68
  %326 = load ptr, ptr @__TVMFuncCall, align 8, !dbg !68, !tbaa !20
  %327 = load ptr, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 4), align 8, !dbg !68
  %.not.i = icmp eq ptr %327, null, !dbg !68
  br i1 %.not.i, label %handle_init.i, label %handle_init_end.i, !dbg !68, !prof !59

handle_init.i:                                    ; preds = %call_end126
  %328 = load ptr, ptr @__tvm_module_ctx, align 8, !dbg !68, !tbaa !20
  %329 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !68, !tbaa !20
  %330 = call i32 %329(ptr %328, ptr nonnull @.str.156, ptr nonnull %0), !dbg !68
  %331 = icmp eq i32 %330, 0, !dbg !68
  br i1 %331, label %call_end.i, label %forward_op_compute_.exit, !dbg !68, !prof !19

handle_init_end.i:                                ; preds = %call_end.i, %call_end126
  %332 = phi ptr [ %327, %call_end126 ], [ %334, %call_end.i ], !dbg !68
  %333 = call i32 %326(ptr %332, ptr nonnull %stack_value131, ptr nonnull %stack_tcode132, i32 8, ptr nonnull %324, ptr nonnull %325), !dbg !68
  br label %forward_op_compute_.exit, !dbg !68

call_end.i:                                       ; preds = %handle_init.i
  %334 = load ptr, ptr %0, align 8, !dbg !68
  store ptr %334, ptr getelementptr inbounds (<{ ptr, ptr, ptr, ptr, ptr }>, ptr @_MergedGlobals, i32 0, i32 4), align 8, !dbg !68
  br label %handle_init_end.i, !dbg !68

forward_op_compute_.exit:                         ; preds = %handle_init.i, %handle_init_end.i
  %common.ret.op.i = phi i32 [ %330, %handle_init.i ], [ %333, %handle_init_end.i ], !dbg !68
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0), !dbg !68
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #2 {
b0:
  %v0 = bitcast float %a0 to i32
  %v1 = and i32 %v0, 2147483647
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %const = bitcast i32 -114688 to i32
  %v5 = lshr i32 %v0, 13
  %v6 = and i32 %v5, 65535
  %v7 = add nuw nsw i32 %v6, %const
  %v8 = and i32 %v0, 8191
  %v9 = icmp ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %const_mat = add i32 %const, 1
  %v10 = add nuw nsw i32 %v6, %const_mat
  br label %b13

b3:                                               ; preds = %b1
  %v11 = icmp eq i32 %v8, 4096
  br i1 %v11, label %b4, label %b13

b4:                                               ; preds = %b3
  %0 = lshr i32 %v0, 13
  %v12 = and i32 %v7, 65535
  %v13 = and i32 %0, 1
  %v14 = add nuw nsw i32 %v12, %v13
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  br i1 %v35, label %b12, label %b13

b12:                                              ; preds = %b11
  %1 = lshr i32 %v28, 13
  %v36 = and i32 %1, 1
  %v37 = add nuw nsw i32 %v36, %v30
  br label %b13

b13:                                              ; preds = %b12, %b11, %b10, %b8, %b7, %b6, %b4, %b3, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ], [ %v30, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  ret i16 %vlast
}

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #2 {
b0:
  %0 = zext i16 %a0 to i32
  %v1 = and i32 %0, 32767
  %v3 = add nsw i32 %v1, -1024
  %v4 = icmp ult i32 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v1, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp ugt i32 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v1, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i32 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp ult i32 %v1, 256
  %v12 = lshr i32 %v1, 8
  %v13 = select i1 %v11, i32 %v1, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v1, %v29
  %v31 = xor i32 %v30, 8388608
  %v32 = shl i32 %v28, 23
  %v33 = sub i32 1124073472, %v32
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i32 %0, 32768
  %v38 = shl nuw i32 %v36, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #4

attributes #0 = { "target-cpu"="apple-latest" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nofree nosync nounwind memory(none) "target-cpu"="apple-latest" "target-features" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.tir", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=arm64-apple-macos -mcpu=apple-latest"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 2}
!5 = distinct !DISubprogram(name: "main.tir", scope: !1, file: !1, type: !6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !10, !8, !9, !10, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !DILocalVariable(name: "arg1", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "arg2", arg: 2, scope: !5, file: !1, type: !10)
!14 = !DILocalVariable(name: "arg3", arg: 3, scope: !5, file: !1, type: !8)
!15 = !DILocalVariable(name: "arg4", arg: 4, scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "arg5", arg: 5, scope: !5, file: !1, type: !10)
!17 = !DILocalVariable(name: "arg6", arg: 6, scope: !5, file: !1, type: !9)
!18 = !DILocation(line: 0, scope: !5)
!19 = !{!"branch_weights", i32 1048576, i32 1}
!20 = !{!21, !21, i64 0}
!21 = !{!"ctx_ptr", !22, i64 0}
!22 = !{!"tvm-tbaa"}
!23 = !{!24, !24, i64 0}
!24 = !{!"0x60000105fa20.w4.b0", !25, i64 0}
!25 = !{!"0x60000105fa20.w8.b0", !26, i64 0}
!26 = !{!"0x60000105fa20.w16.b0", !27, i64 0}
!27 = !{!"0x60000105fa20.w32.b0", !28, i64 0}
!28 = !{!"0x60000105fa20.w64.b0", !29, i64 0}
!29 = !{!"0x60000105fa20.w128.b0", !30, i64 0}
!30 = !{!"0x60000105fa20.w256.b0", !31, i64 0}
!31 = !{!"0x60000105fa20.w512.b0", !32, i64 0}
!32 = !{!"0x60000105fa20.w1024.b0", !33, i64 0}
!33 = !{!"0x60000105fa20", !22, i64 0}
!34 = !{!35, !35, i64 0}
!35 = !{!"0x60000105fa20.w4.b4", !25, i64 0}
!36 = !{!37, !37, i64 0}
!37 = !{!"0x60000105fa20.w4.b8", !38, i64 0}
!38 = !{!"0x60000105fa20.w8.b8", !26, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"0x60000105fa20.w4.b12", !38, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"0x60000105fa20.w4.b16", !43, i64 0}
!43 = !{!"0x60000105fa20.w8.b16", !44, i64 0}
!44 = !{!"0x60000105fa20.w16.b16", !27, i64 0}
!45 = !{!46, !46, i64 0}
!46 = !{!"0x60000105fa20.w4.b20", !43, i64 0}
!47 = !{!48, !48, i64 0}
!48 = !{!"0x60000105fa20.w4.b24", !49, i64 0}
!49 = !{!"0x60000105fa20.w8.b24", !44, i64 0}
!50 = !{!51, !51, i64 0}
!51 = !{!"0x60000105fa20.w4.b28", !49, i64 0}
!52 = !{!53, !53, i64 0}
!53 = !{!"0x60000105fa20.w4.b32", !54, i64 0}
!54 = !{!"0x60000105fa20.w8.b32", !55, i64 0}
!55 = !{!"0x60000105fa20.w16.b32", !56, i64 0}
!56 = !{!"0x60000105fa20.w32.b32", !28, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"tvm-alias", !22}
!59 = !{!"branch_weights", i32 1, i32 1048576}
!60 = distinct !DISubprogram(name: "main.tir", scope: !1, file: !1, type: !6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !61)
!61 = !{!62, !63, !64, !65, !66, !67}
!62 = !DILocalVariable(name: "arg1", arg: 1, scope: !60, file: !1, type: !9)
!63 = !DILocalVariable(name: "arg2", arg: 2, scope: !60, file: !1, type: !10)
!64 = !DILocalVariable(name: "arg3", arg: 3, scope: !60, file: !1, type: !8)
!65 = !DILocalVariable(name: "arg4", arg: 4, scope: !60, file: !1, type: !9)
!66 = !DILocalVariable(name: "arg5", arg: 5, scope: !60, file: !1, type: !10)
!67 = !DILocalVariable(name: "arg6", arg: 6, scope: !60, file: !1, type: !9)
!68 = !DILocation(line: 0, scope: !60)
!69 = !{!70, !70, i64 0}
!70 = !{!"0x600001034270.w4.b0", !71, i64 0}
!71 = !{!"0x600001034270.w8.b0", !72, i64 0}
!72 = !{!"0x600001034270.w16.b0", !73, i64 0}
!73 = !{!"0x600001034270.w32.b0", !74, i64 0}
!74 = !{!"0x600001034270.w64.b0", !75, i64 0}
!75 = !{!"0x600001034270.w128.b0", !76, i64 0}
!76 = !{!"0x600001034270.w256.b0", !77, i64 0}
!77 = !{!"0x600001034270.w512.b0", !78, i64 0}
!78 = !{!"0x600001034270.w1024.b0", !79, i64 0}
!79 = !{!"0x600001034270", !22, i64 0}
!80 = !{!81, !81, i64 0}
!81 = !{!"0x600001034270.w4.b4", !71, i64 0}
!82 = !{!83, !83, i64 0}
!83 = !{!"0x600001034270.w4.b8", !84, i64 0}
!84 = !{!"0x600001034270.w8.b8", !72, i64 0}
!85 = !{!86, !86, i64 0}
!86 = !{!"0x600001034270.w4.b12", !84, i64 0}
!87 = !{!88, !88, i64 0}
!88 = !{!"0x600001034270.w4.b16", !89, i64 0}
!89 = !{!"0x600001034270.w8.b16", !90, i64 0}
!90 = !{!"0x600001034270.w16.b16", !73, i64 0}

