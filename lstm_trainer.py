import tvm
from tvm import relax
from tvm.relax.training import Trainer, SGD, MomentumSGD
from tvm.script._parser import ir as I, relax as R, tir as T

from dataloader import *

def build_lstm_mod(steps_num, in_size, hidden_size, out_size, batch_size=1):
    """
        inputs: x_t, C_init, H_init, Wh_{}, Wx_{}, B_{}, Wh_q, B_q
    """
    dtype = relax.DynTensorType(dtype="float32")

    inputs_list = []
    x_list = [relax.Var("x_" + str(i), [batch_size, in_size], dtype) for i in range(steps_num)]
    inputs_list += x_list

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

                F = bb.emit(R.sigmoid(
                    R.add(
                        R.add(
                            R.matmul(H, params["Wh_f"]),
                            R.matmul(x_list[i], params["Wx_f"])
                        ),
                        params["B_f"]
                    )
                ))

                I = bb.emit(R.sigmoid(
                    R.add(
                        R.add(
                            R.matmul(H, params["Wh_i"]),
                            R.matmul(x_list[i], params["Wx_i"])
                        ),
                        params["B_i"]
                    )
                ))

                C_tilde = bb.emit(R.tanh(
                    R.add(
                        R.add(
                            R.matmul(H, params["Wh_c"]),
                            R.matmul(x_list[i], params["Wx_c"])
                        ),
                        params["B_c"]
                    )
                ))

                O = bb.emit(R.sigmoid(
                    R.add(
                        R.add(
                            R.matmul(H, params["Wh_o"]),
                            R.matmul(x_list[i], params["Wx_o"])
                        ),
                        params["B_o"]
                    )
                ))

                C = bb.emit(R.add(R.multiply(F, C), R.multiply(I, C_tilde)))
                H = bb.emit(R.multiply(O, R.tanh(C)))

            lv0 = bb.emit(R.matmul(H, params["Wh_q"]))
            out = bb.emit_output(R.add(lv0, params["B_q"]))
        bb.emit_func_output(out)
    return relax.transform.Normalize()(bb.get())


def train_lstm(): # in dataset

    # x_t, y (label), C_init, H_init
    def make_inputs(steps_num, hidden_size, data_entities, data_label):
        inputs = [0 for _ in range(steps_num)]

        for i in range(steps_num):
            rev_idx = steps_num - 1 - i
            inputs[rev_idx] = data_entities[:, i, :].astype(np.float32)
        inputs.append(np.zeros((data_entities.shape[0], hidden_size)).astype(np.float32)) # C_init
        inputs.append(np.zeros((data_entities.shape[0], hidden_size)).astype(np.float32)) # H_init
        inputs.append(data_label.astype(np.float32))
        return inputs

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
    trainer = Trainer(backbone=mod, func_name="LSTM", parameters_indices=range(max_len+2, len(mod["LSTM"].params)))
    trainer.prepare("relax.nn.softmax_cross_entropy", SGD(None, 0.01))
    trainer.set_vm_config(target="llvm", device=tvm.cpu())
    trainer.setup()

    # train
    print("training...")

    """
    Every word is padding to max_len
    Each step in LSTM, we can feed one character to the network, and
    this may help explore the connection between characters in one word.
    """
    trainer.rand_init_params()
    for i in range(epochs):
        total_loss = 0
        batch_num = 0

        for (idxs, e, l) in mb_train:
            batch_num += 1
            if len(idxs) != batch_size:
                continue

            inputs = make_inputs(max_len, hidden_size, e, l)
            total_loss += trainer.backward(*inputs)

        print("epoch = {}, loss = {}".format(str(i), total_loss / batch_num))

    print("done")

train_lstm()
