from __future__ import annotations

import time

import math
import numpy as np
import tvm
from tvm import relax
from tvm import relay, te, tir, topi
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.relax import ExternFunc, ShapeExpr, Tuple
from tvm.relax.testing import dump_ast, nn
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.function import PrimFunc

from utils import LowerToTensorIRPass

from dataloader import *

np.random.seed(1)

def build_lstm_mod(steps_num, in_size, hidden_size, out_size, batch_size=1):
    """
        inputs: x_t, y (label), C_init, H_init, Wh_{}, Wx_{}, B_{}, Wh_q, B_q
    """
    dtype = relax.DynTensorType(dtype="float32")

    inputs_list = []
    x_list = [relax.Var("x_" + str(i), [batch_size, in_size], dtype) for i in range(steps_num)]
    inputs_list += x_list
    y = relax.Var("y", [batch_size, out_size], dtype)
    inputs_list.append(y)
    
    C = relax.Var("C", [batch_size, hidden_size], dtype)
    H = relax.Var("H", [batch_size, hidden_size], dtype)
    inputs_list.append(C)
    inputs_list.append(H)

    params = {}

    for suffix in ["f", "i", "c", "o"]:
        params["Wh_" + suffix] = relax.Var("Wh_" + suffix, [hidden_size, hidden_size], dtype)
        params["Wx_" + suffix] = relax.Var("Wx_" + suffix, [in_size, hidden_size], dtype)
        params["B_" + suffix]  = relax.Var("B_" + suffix, [hidden_size], dtype)
        inputs_list += [params["Wh_" + suffix], params["Wx_" + suffix], params["B_" + suffix]]
    params["Wh_q"] = relax.Var("Wh_q", [hidden_size, out_size], dtype)
    params["B_q"] = relax.Var("B_q", [out_size], dtype)
    inputs_list += [params["Wh_q"], params["B_q"]]
    
    bb = relax.BlockBuilder()
    with bb.function("LSTM", inputs_list):
        with bb.dataflow():
            for i in range(steps_num):

                F = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_f"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_f"])
                        ),
                        params["B_f"]
                    )
                ))

                I = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_i"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_i"])
                        ),
                        params["B_i"]
                    )
                ))

                C_tilde = bb.emit(relax.op.nn.tanh(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_c"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_c"])
                        ),
                        params["B_c"]
                    )
                ))

                O = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_o"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_o"])
                        ),
                        params["B_o"]
                    )
                ))

                C = bb.emit(relax.op.add(relax.op.multiply(F, C), relax.op.multiply(I, C_tilde)))
                H = bb.emit(relax.op.multiply(O, relax.op.nn.tanh(C)))
                
            lv0 = bb.emit(relax.op.nn.matmul(H, params["Wh_q"]))
            out = bb.emit(relax.op.add(lv0, params["B_q"]))
            loss = bb.emit_output(relax.op.nn.softmax_cross_entropy(out, y))
        bb.emit_func_output(loss)
    mod = relax.transform.Normalize()(bb.get())
    return relax.transform.SimpleAD(mod.get_global_var("LSTM"), require_grads=list(params.values()))(mod)


def train_lstm(): # in dataset

    # x_t, y (label), C_init, H_init
    def make_inputs(steps_num, hidden_size, data_entities, data_label):
        inputs = [0 for _ in range(steps_num)]

        for i in range(steps_num):
            rev_idx = steps_num - 1 - i
            inputs[rev_idx] = data_entities[:, i, :].astype(np.float32)
        inputs.append(data_label.astype(np.float32))
        inputs.append(np.zeros((data_entities.shape[0], hidden_size)).astype(np.float32)) # C_init
        inputs.append(np.zeros((data_entities.shape[0], hidden_size)).astype(np.float32)) # H_init
        return inputs

    # Wh_{}, Wx_{}, B_{}, Wh_q, B_q
    def make_params(hidden_size, out_size):
        def get_init(size1, size2):
            a = math.sqrt(6.0 / (size1 + size2))
            return a * np.random.uniform(-1.0, 1.0, (size1, size2)).astype(np.float32)

        ret = []
        for _ in range(4): # f, i, c, o
            ret.append(get_init(hidden_size, hidden_size)) # Wh
            ret.append(get_init(in_size, hidden_size)) # Wx
            ret.append(0.1 * np.random.uniform(-1.0, 1.0, (hidden_size, )).astype(np.float32)) # B
        ret.append(get_init(hidden_size, out_size)) # Wq
        ret.append(0.1 * np.random.uniform(-1.0, 1.0, (out_size, )).astype(np.float32)) # Bq
        return ret


    """
        Task: character classification
    """

    epochs = 15
    max_len = 10
    hidden_size = 50
    batch_size = 64
    dataset = "data/smaller"
    init_lr = 0.07

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size,
                               max_len, len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, batch_size,
                               max_len, len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, batch_size,
                               max_len, len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print("building lstm...")
    
    in_size = mb_train.num_chars
    out_size = mb_train.num_labels

    print(max_len, in_size, hidden_size, out_size, batch_size)

    mod = build_lstm_mod(max_len, in_size, hidden_size, out_size, batch_size)
    # mod.show()
    TIRModule = LowerToTensorIRPass()(mod)
    ex = relax.vm.build(TIRModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    print("done")

    # train
    print("training...")
    # get default data and params
    lr = init_lr
    
    """
    Every word is padding to max_len
    Each step in LSTM, we can feed one character to the network, and 
    this may help explore the connection between characters in one word.
    """
    params = make_params(hidden_size, out_size)
    for i in range(epochs):
        total_loss = 0
        batch_num = 0
        for (idxs, e, l) in mb_train:
            batch_num += 1
            if len(idxs) != batch_size:
                continue

            inputs = make_inputs(max_len, hidden_size, e, l) + params
            inputs_tvm = [tvm.nd.array(i) for i in inputs]
            assert len(inputs) == len(mod["LSTM"].params)
            
            # for j in range(len(inputs)):
            #     print("arg_name={}, inputs_shape={}, mod_arg_shape={}".format(
            #         mod["LSTM"].params[j].name_hint, 
            #         inputs[j].shape, 
            #         mod["LSTM"].params[j].shape
            #     ))
            
            loss, grads = vm["LSTM"](*inputs_tvm)
            assert len(params) == len(grads)    
            for j in range(len(params)):
                params[j] -= lr * grads[j].numpy()
            total_loss += loss.numpy()

        print("epoch = {}, loss = {}".format(str(i), total_loss / batch_num))

        # validate
        min_valid_loss = 99999
        best_model = []
        valid_loss = 0
        valid_batch_num = 0
        for (idxs, e, l) in mb_valid:
            if len(idxs) != batch_size:
                continue
            valid_batch_num += 1
            inputs = make_inputs(max_len, hidden_size, e, l) + params
            inputs_tvm = [tvm.nd.array(i) for i in inputs]
            loss, grads = vm["LSTM"](*inputs_tvm)
            valid_loss += int(loss.numpy())
        valid_loss /= valid_batch_num
        print("valid_loss = " + str(valid_loss))
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_model = [i.copy() for i in params]
        
    print("done")
    print("best model: ", best_model)
    # np.save(train_loss_file, train_loss)

    # for (idxs, e, l) in mb_test:
    #     inputs = make_inputs(max_len, hidden_size, e, l) + best_model
    #     inputs_tvm = [tvm.nd.array(i) for i in inputs]
    #     loss, grads = vm["LSTM"](*inputs_tvm)
    #     print("test_loss = " + str(int(loss.numpy())))
    #     pred = np.argmax(value_dict['output'], axis=1)
    #     true = np.argmax(l, axis=1)
    #     print("pred_rate = " + str((pred == true).sum() / batch_size))

train_lstm()