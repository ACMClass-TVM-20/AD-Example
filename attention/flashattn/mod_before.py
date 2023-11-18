# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def backward_op(dO: T.Buffer((6, 32, 512, 128), "float16"), Q: T.Buffer((6, 32, 512, 128), "float16"), K: T.Buffer((6, 32, 512, 128), "float16"), V: T.Buffer((6, 32, 512, 128), "float16"), O: T.Buffer((6, 32, 512, 128), "float16"), L: T.Buffer((6, 32, 512), "float16"), dQ: T.Buffer((6, 32, 512, 128), "float16"), dK: T.Buffer((6, 32, 512, 128), "float16"), dV: T.Buffer((6, 32, 512, 128), "float16")):
        # with T.block("root"):
        Qi = T.alloc_buffer((8, 128), "float16", scope="shared")
        Kj = T.alloc_buffer((8, 128), "float16", scope="shared")
        Vj = T.alloc_buffer((8, 128), "float16", scope="shared")
        Li = T.alloc_buffer((8,), "float16", scope="local")
        Di = T.alloc_buffer((8,), "float16", scope="local")
        dOi = T.alloc_buffer((8, 128), "float16", scope="shared")
        dQi = T.alloc_buffer((8, 128), scope="shared")
        dKj = T.alloc_buffer((8, 128), scope="shared")
        dVj = T.alloc_buffer((8, 128), scope="shared")
        Sij = T.alloc_buffer((8, 8), scope="shared")
        Pij = T.alloc_buffer((8, 8), scope="shared")
        dSij = T.alloc_buffer((8, 8), scope="shared")
        dPij = T.alloc_buffer((8, 8), scope="shared")
        D = T.alloc_buffer((6, 32, 512), "float16")
        dQ_cache = T.alloc_buffer((6, 32, 64, 512, 128), "float16")
        accumulator = T.alloc_buffer((1,), scope="local")
        for b, nh, seq, head_dim in T.grid(6, 32, 512, 128):
            with T.block("compute_D"):
                ab, anh, aseq, ahead_dim = T.axis.remap("SSSR", [b, nh, seq, head_dim])
                T.reads(dO[ab, anh, aseq, ahead_dim], O[ab, anh, aseq, ahead_dim])
                T.writes(D[ab, anh, aseq])
                with T.init():
                    D[ab, anh, aseq] = T.float16(0)
                D[ab, anh, aseq] = D[ab, anh, aseq] + dO[ab, anh, aseq, ahead_dim] * O[ab, anh, aseq, ahead_dim]
        for b, nh, j in T.grid(6, 32, 64):
            with T.block("outer_compute"):
                ab, anh, aj = T.axis.remap("SSS", [b, nh, j])
                T.reads(K[ab, anh, aj * 8:aj * 8 + 8, 0:128], V[ab, anh, aj * 8:aj * 8 + 8, 0:128], Q[ab, anh, 0:512, 0:128], dO[ab, anh, 0:512, 0:128], L[ab, anh, 0:512], D[ab, anh, 0:512], Qi[0:8, 0:128], Kj[0:8, 0:128], Sij[0:8, 0:8], Li[0:8], dVj[0:8, 0:128], Pij[0:8, 0:8], dOi[0:8, 0:128], Vj[0:8, 0:128], dPij[0:8, 0:8], Di[0:8], dSij[0:8, 0:8], dQi[0:8, 0:128], dKj[0:8, 0:128])
                T.writes(Kj[0:8, 0:128], Vj[0:8, 0:128], dKj[0:8, 0:128], dVj[0:8, 0:128], Qi[0:8, 0:128], dOi[0:8, 0:128], Li[0:8], Di[0:8], Sij[0:8, 0:8], Pij[0:8, 0:8], dPij[0:8, 0:8], dSij[0:8, 0:8], dQi[0:8, 0:128], dQ_cache[ab, anh, aj, 0:512, 0:128], dK[ab, anh, aj * 8:aj * 8 + 8, 0:128], dV[ab, anh, aj * 8:aj * 8 + 8, 0:128])
                for v0, v1 in T.grid(8, 128):
                    with T.block("load_Kj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(K[ab, anh, aj * 8 + av0, av1])
                        T.writes(Kj[av0, av1])
                        Kj[av0, av1] = K[ab, anh, aj * 8 + av0, av1]
                for v0, v1 in T.grid(8, 128):
                    with T.block("load_Vj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(V[ab, anh, aj * 8 + av0, av1])
                        T.writes(Vj[av0, av1])
                        Vj[av0, av1] = V[ab, anh, aj * 8 + av0, av1]
                for v0, v1 in T.grid(8, 128):
                    with T.block("init_dKj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads()
                        T.writes(dKj[av0, av1])
                        dKj[av0, av1] = T.float32(0)
                for v0, v1 in T.grid(8, 128):
                    with T.block("init_dVj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads()
                        T.writes(dVj[av0, av1])
                        dVj[av0, av1] = T.float32(0)
                for i in range(64):
                    with T.block("inner_compute"):
                        ai = T.axis.reduce(64, i)
                        T.reads(Q[ab, anh, ai * 8:ai * 8 + 8, 0:128], dO[ab, anh, ai * 8:ai * 8 + 8, 0:128], L[ab, anh, ai * 8:ai * 8 + 8], D[ab, anh, ai * 8:ai * 8 + 8], Qi[0:8, 0:128], Kj[0:8, 0:128], Sij[0:8, 0:8], Li[0:8], dVj[0:8, 0:128], Pij[0:8, 0:8], dOi[0:8, 0:128], Vj[0:8, 0:128], dPij[0:8, 0:8], Di[0:8], dSij[0:8, 0:8], dQi[0:8, 0:128], dKj[0:8, 0:128])
                        T.writes(Qi[0:8, 0:128], dOi[0:8, 0:128], Li[0:8], Di[0:8], Sij[0:8, 0:8], Pij[0:8, 0:8], dVj[0:8, 0:128], dPij[0:8, 0:8], dSij[0:8, 0:8], dQi[0:8, 0:128], dQ_cache[ab, anh, aj, ai * 8:ai * 8 + 8, 0:128], dKj[0:8, 0:128])
                        for v0, v1 in T.grid(8, 128):
                            with T.block("load_Qi"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(Q[ab, anh, ai * 8 + av0, av1])
                                T.writes(Qi[av0, av1])
                                Qi[av0, av1] = Q[ab, anh, ai * 8 + av0, av1]
                        for v0, v1 in T.grid(8, 128):
                            with T.block("load_dOi"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(dO[ab, anh, ai * 8 + av0, av1])
                                T.writes(dOi[av0, av1])
                                dOi[av0, av1] = dO[ab, anh, ai * 8 + av0, av1]
                        for v0 in range(8):
                            with T.block("load_Li"):
                                av0 = T.axis.spatial(8, v0)
                                T.reads(L[ab, anh, ai * 8 + av0])
                                T.writes(Li[av0])
                                Li[av0] = L[ab, anh, ai * 8 + av0]
                        for v0 in range(8):
                            with T.block("load_Di"):
                                av0 = T.axis.spatial(8, v0)
                                T.reads(D[ab, anh, ai * 8 + av0])
                                T.writes(Di[av0])
                                Di[av0] = D[ab, anh, ai * 8 + av0]
                        for v0, v1, v2 in T.grid(8, 8, 128):
                            with T.block("compute_Sij"):
                                av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                T.reads(Qi[av0, av2], Kj[av1, av2])
                                T.writes(Sij[av0, av1])
                                with T.init():
                                    Sij[av0, av1] = T.float32(0)
                                Sij[av0, av1] = T.Cast("float32", Sij[av0, av1] + T.Cast("float32", Qi[av0, av2] * Kj[av1, av2] * T.float16(0.088388347648318433)))
                        for v0, v1 in T.grid(8, 8):
                            with T.block("compute_Pij"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(Sij[av0, av1], Li[av0])
                                T.writes(Pij[av0, av1])
                                Pij[av0, av1] = T.exp(Sij[av0, av1] - T.Cast("float32", Li[av0]))
                        for v0, v1, v2 in T.grid(8, 8, 128):
                            with T.block("compute_dVj"):
                                av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
                                T.reads(dVj[av0, av2], Pij[av1, av0], dOi[av1, av2])
                                T.writes(dVj[av0, av2])
                                dVj[av0, av2] = dVj[av0, av2] + Pij[av1, av0] * T.Cast("float32", dOi[av1, av2])
                        for v0, v1, v2 in T.grid(8, 8, 128):
                            with T.block("compute_dPij"):
                                av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                T.reads(dOi[av0, av2], Vj[av1, av2])
                                T.writes(dPij[av0, av1])
                                with T.init():
                                    dPij[av0, av1] = T.float32(0)
                                dPij[av0, av1] = dPij[av0, av1] + T.Cast("float32", dOi[av0, av2] * Vj[av1, av2])
                        for v0, v1 in T.grid(8, 8):
                            with T.block("compute_dSij"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(Pij[av0, av1], dPij[av0, av1], Di[av0])
                                T.writes(dSij[av0, av1])
                                dSij[av0, av1] = Pij[av0, av1] * (dPij[av0, av1] - T.Cast("float32", Di[av0]))
                        for v0, v1, v2 in T.grid(8, 128, 8):
                            with T.block("compute_dQi"):
                                av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                T.reads(dSij[av0, av2], Kj[av2, av1])
                                T.writes(dQi[av0, av1])
                                with T.init():
                                    dQi[av0, av1] = T.float32(0)
                                dQi[av0, av1] = dQi[av0, av1] + T.Cast("float32", T.float16(0.088388347648318433)) * dSij[av0, av2] * T.Cast("float32", Kj[av2, av1])
                        for v0, v1 in T.grid(8, 128):
                            with T.block("store_dQi_cache"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(dQi[av0, av1])
                                T.writes(dQ_cache[ab, anh, aj, ai * 8 + av0, av1])
                                dQ_cache[ab, anh, aj, ai * 8 + av0, av1] = T.Cast("float16", dQi[av0, av1])
                        for v0, v1, v2 in T.grid(8, 8, 128):
                            with T.block("compute_dKj"):
                                av0, av1, av2 = T.axis.remap("SRS", [v0, v1, v2])
                                T.reads(dKj[av0, av2], dSij[av1, av0], Qi[av1, av2])
                                T.writes(dKj[av0, av2])
                                dKj[av0, av2] = dKj[av0, av2] + T.Cast("float32", T.float16(0.088388347648318433)) * dSij[av1, av0] * T.Cast("float32", Qi[av1, av2])
                for v0, v1 in T.grid(8, 128):
                    with T.block("store_dKj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(dKj[av0, av1])
                        T.writes(dK[ab, anh, aj * 8 + av0, av1])
                        dK[ab, anh, aj * 8 + av0, av1] = T.Cast("float16", dKj[av0, av1])
                for v0, v1 in T.grid(8, 128):
                    with T.block("store_dVj"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(dVj[av0, av1])
                        T.writes(dV[ab, anh, aj * 8 + av0, av1])
                        dV[ab, anh, aj * 8 + av0, av1] = T.Cast("float16", dVj[av0, av1])
        for b, nh, v0, v1 in T.grid(6, 32, 512, 128):
            with T.block("store_dQ"):
                ab, anh, av0, av1 = T.axis.remap("SSSS", [b, nh, v0, v1])
                T.reads(dQ_cache[ab, anh, 0:64, av0, av1], accumulator[0])
                T.writes(accumulator[0], dQ[ab, anh, av0, av1])
                for j in range(64):
                    with T.block("combine_dQ"):
                        aj = T.axis.reduce(64, j)
                        T.reads(dQ_cache[ab, anh, aj, av0, av1])
                        T.writes(accumulator[0])
                        with T.init():
                            accumulator[0] = T.float32(0)
                        accumulator[0] = accumulator[0] + T.Cast("float32", dQ_cache[ab, anh, aj, av0, av1])
                dQ[ab, anh, av0, av1] = T.Cast("float16", accumulator[0])

    @T.prim_func
    def forward_op(Q: T.Buffer((6, 32, 512, 128), "float16"), K: T.Buffer((6, 32, 512, 128), "float16"), V: T.Buffer((6, 32, 512, 128), "float16"), O: T.Buffer((6, 32, 512, 128), "float16"), L: T.Buffer((6, 32, 512), "float16")):
        # with T.block("root"):
        Qi = T.alloc_buffer((8, 128), "float16", scope="shared")
        Kj = T.alloc_buffer((8, 128), "float16", scope="shared")
        Oi = T.alloc_buffer((8, 128), scope="shared")
        li = T.alloc_buffer((8,), scope="local")
        mi = T.alloc_buffer((8,), scope="local")
        mi_mid = T.alloc_buffer((8,), scope="local")
        mi_old = T.alloc_buffer((8,), scope="local")
        Sij = T.alloc_buffer((8, 8), scope="shared")
        Pij = T.alloc_buffer((8, 8), scope="shared")
        for b, nh, i in T.grid(6, 32, 64):
            with T.block("outer_compute"):
                ab, anh, ai = T.axis.remap("SSS", [b, nh, i])
                T.reads(Q[ab, anh, ai * 8:ai * 8 + 8, 0:128], K[ab, anh, 0:512, 0:128], Qi[0:8, 0:128], Kj[0:8, 0:128], mi[0:8], Sij[0:8, 0:8], mi_old[0:8], mi_mid[0:8], Pij[0:8, 0:8], V[ab, anh, 0:512, 0:128], Oi[0:8, 0:128], li[0:8])
                T.writes(Qi[0:8, 0:128], Oi[0:8, 0:128], li[0:8], mi[0:8], Kj[0:8, 0:128], Sij[0:8, 0:8], mi_old[0:8], mi_mid[0:8], Pij[0:8, 0:8], O[ab, anh, ai * 8:ai * 8 + 8, 0:128], L[ab, anh, ai * 8:ai * 8 + 8])
                for v0, v1 in T.grid(8, 128):
                    with T.block("load_Qi"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(Q[ab, anh, ai * 8 + av0, av1])
                        T.writes(Qi[av0, av1])
                        Qi[av0, av1] = Q[ab, anh, ai * 8 + av0, av1]
                for v0, v1 in T.grid(8, 128):
                    with T.block("init_Oi"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads()
                        T.writes(Oi[av0, av1])
                        Oi[av0, av1] = T.float32(0)
                for v0 in range(8):
                    with T.block("init_li"):
                        av0 = T.axis.spatial(8, v0)
                        T.reads()
                        T.writes(li[av0])
                        li[av0] = T.float32(0)
                for v0 in range(8):
                    with T.block("init_mi"):
                        av0 = T.axis.spatial(8, v0)
                        T.reads()
                        T.writes(mi[av0])
                        mi[av0] = T.float32(-3.4028234663852886e+38)
                for j in range(64):
                    with T.block("inner_compute"):
                        aj = T.axis.reduce(64, j)
                        T.reads(K[ab, anh, aj * 8:aj * 8 + 8, 0:128], Qi[0:8, 0:128], Kj[0:8, 0:128], mi[0:8], Sij[0:8, 0:8], mi_old[0:8], mi_mid[0:8], Pij[0:8, 0:8], V[ab, anh, aj * 8:aj * 8 + 8, 0:128])
                        T.writes(Kj[0:8, 0:128], Sij[0:8, 0:8], mi_old[0:8], mi_mid[0:8], mi[0:8], Pij[0:8, 0:8], li[0:8], Oi[0:8, 0:128])
                        for v0, v1 in T.grid(8, 128):
                            with T.block("load_Kj"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(K[ab, anh, aj * 8 + av0, av1])
                                T.writes(Kj[av0, av1])
                                Kj[av0, av1] = K[ab, anh, aj * 8 + av0, av1]
                        for v0, v1, v2 in T.grid(8, 8, 128):
                            with T.block("compute_Sij"):
                                av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                T.reads(Qi[av0, av2], Kj[av1, av2])
                                T.writes(Sij[av0, av1])
                                with T.init():
                                    Sij[av0, av1] = T.float32(0)
                                Sij[av0, av1] = Sij[av0, av1] + T.Cast("float32", Qi[av0, av2] * Kj[av1, av2] * T.float16(0.088388347648318433))
                        for v0 in range(8):
                            with T.block("store_mi_old"):
                                av0 = T.axis.spatial(8, v0)
                                T.reads(mi[av0])
                                T.writes(mi_old[av0])
                                mi_old[av0] = mi[av0]
                        for v0, v1 in T.grid(8, 8):
                            with T.block("compute_mi_mid"):
                                av0, av1 = T.axis.remap("SR", [v0, v1])
                                T.reads(Sij[av0, av1])
                                T.writes(mi_mid[av0])
                                with T.init():
                                    mi_mid[av0] = T.float32(-3.4028234663852886e+38)
                                mi_mid[av0] = T.max(mi_mid[av0], Sij[av0, av1])
                        for v0 in range(8):
                            with T.block("compute_mi"):
                                av0 = T.axis.spatial(8, v0)
                                T.reads(mi_old[av0], mi_mid[av0])
                                T.writes(mi[av0])
                                mi[av0] = T.max(mi_old[av0], mi_mid[av0])
                        for v0, v1 in T.grid(8, 8):
                            with T.block("compute_Pij"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(Sij[av0, av1], mi[av0])
                                T.writes(Pij[av0, av1])
                                Pij[av0, av1] = T.exp(Sij[av0, av1] - mi[av0])
                        for v0, v1 in T.grid(8, 8):
                            with T.block("compute_li"):
                                av0, av1 = T.axis.remap("SR", [v0, v1])
                                T.reads(mi_old[av0], mi[av0], Pij[av0, av1])
                                T.writes(li[av0])
                                with T.init():
                                    li[av0] = T.exp(mi_old[av0] - mi[av0]) * li[av0]
                                li[av0] = li[av0] + Pij[av0, av1]
                        for v0, v1 in T.grid(8, 128):
                            with T.block("load_Vj"):
                                av0, av1 = T.axis.remap("SS", [v0, v1])
                                T.reads(V[ab, anh, aj * 8 + av0, av1])
                                T.writes(Kj[av0, av1])
                                Kj[av0, av1] = V[ab, anh, aj * 8 + av0, av1]
                        for v0, v1, v2 in T.grid(8, 128, 8):
                            with T.block("compute_Oi"):
                                av0, av1, av2 = T.axis.remap("SSR", [v0, v1, v2])
                                T.reads(mi_old[av0], mi[av0], Pij[av0, av2], Kj[av2, av1])
                                T.writes(Oi[av0, av1])
                                with T.init():
                                    Oi[av0, av1] = T.exp(mi_old[av0] - mi[av0]) * Oi[av0, av1]
                                Oi[av0, av1] = Oi[av0, av1] + T.Cast("float32", Pij[av0, av2] * T.Cast("float32", Kj[av2, av1]))
                for v0, v1 in T.grid(8, 128):
                    with T.block("store_Oi"):
                        av0, av1 = T.axis.remap("SS", [v0, v1])
                        T.reads(Oi[av0, av1], li[av0])
                        T.writes(O[ab, anh, ai * 8 + av0, av1])
                        O[ab, anh, ai * 8 + av0, av1] = T.Cast("float16", Oi[av0, av1] / li[av0])
                for v0 in range(8):
                    with T.block("store_L"):
                        av0 = T.axis.spatial(8, v0)
                        T.reads(mi[av0], li[av0])
                        T.writes(L[ab, anh, ai * 8 + av0])
                        L[ab, anh, ai * 8 + av0] = T.Cast("float16", mi[av0] + T.log(li[av0]))

    @R.function
    def backward(dO: R.Tensor((6, 32, 512, 128), dtype="float16"), Q: R.Tensor((6, 32, 512, 128), dtype="float16"), K: R.Tensor((6, 32, 512, 128), dtype="float16"), V: R.Tensor((6, 32, 512, 128), dtype="float16"), O: R.Tensor((6, 32, 512, 128), dtype="float16"), L: R.Tensor((6, 32, 512), dtype="float16")) -> R.Tuple(R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512, 128), dtype="float16")):
        cls = Module
        out = R.call_tir(cls.backward_op, (dO, Q, K, V, O, L), out_sinfo=[R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512, 128), dtype="float16")])
        return out

    @R.function
    def forward(Q: R.Tensor((6, 32, 512, 128), dtype="float16"), K: R.Tensor((6, 32, 512, 128), dtype="float16"), V: R.Tensor((6, 32, 512, 128), dtype="float16")) -> R.Tuple(R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512), dtype="float16")):
        cls = Module
        out = R.call_tir(cls.forward_op, (Q, K, V), out_sinfo=[R.Tensor((6, 32, 512, 128), dtype="float16"), R.Tensor((6, 32, 512), dtype="float16")])
        return out
